"""Models behind one method, `complete(request) -> response`.

spine has no API keys. It calls the headless modes of the agent CLIs that are
already installed and logged in, the way retro does, and turns each one into a
plain chat model:

- no tools of any kind, so a model can't reach for a calculator, a shell or the
  web on an arithmetic item;
- our own system prompt in place of the agent's coding-assistant prompt, where
  the CLI allows it;
- no user or project instruction files, skills, MCP servers or memory;
- a fresh empty working directory for every call.

Each CLI needs different flags for that, documented on each class. What can't be
removed is listed in the README under Limitations.

The CLIs are single-turn, so the conversation is stateless: every call sends the
whole transcript, rendered into one prompt by `render_prompt`. That also makes
every call a pure function of its request, which is what lets `Recorder` key
recordings by a hash of the request and replay a whole run offline.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
import tomllib
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

Role = Literal["user", "assistant"]


@dataclass(frozen=True)
class Message:
    role: Role
    content: str


@dataclass(frozen=True)
class Request:
    provider: str
    model: str
    effort: str | None
    system: str
    messages: tuple[Message, ...]
    # Extended thinking, where the CLI can turn it off. Codex can't: "low" is the
    # least reasoning its models accept.
    thinking: bool = True

    def to_json(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "effort": self.effort,
            "thinking": self.thinking,
            "system": self.system,
            "messages": [asdict(m) for m in self.messages],
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Request:
        return cls(
            provider=data["provider"],
            model=data["model"],
            effort=data.get("effort"),
            system=data["system"],
            messages=tuple(Message(m["role"], m["content"]) for m in data["messages"]),
            thinking=bool(data.get("thinking", True)),
        )

    def key(self) -> str:
        blob = json.dumps(self.to_json(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode()).hexdigest()[:24]


@dataclass(frozen=True)
class Response:
    text: str
    seconds: float = 0.0
    cost_usd: float | None = None
    usage: dict[str, Any] | None = None
    # Anything that looked like a tool call. A pure-chat run should have none, and the
    # analysis refuses to score a reply that has one.
    tool_events: tuple[str, ...] = ()
    raw: Any = None

    def to_json(self) -> dict[str, Any]:
        out = asdict(self)
        out["tool_events"] = list(self.tool_events)
        return out

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Response:
        return cls(
            text=data["text"],
            seconds=float(data.get("seconds") or 0.0),
            cost_usd=data.get("cost_usd"),
            usage=data.get("usage"),
            tool_events=tuple(data.get("tool_events") or ()),
            raw=data.get("raw"),
        )


class Provider(Protocol):
    def complete(self, request: Request) -> Response: ...


class ProviderError(RuntimeError):
    pass


class MissingRecording(ProviderError):
    pass


def render_prompt(messages: tuple[Message, ...]) -> str:
    """One prompt string for a single-turn CLI.

    A first turn goes as is. Later turns wrap the transcript in tags and say whose
    replies the assistant turns were, so the model reads them as its own.
    """
    if len(messages) == 1:
        return messages[0].content
    turns = "\n\n".join(f"<{m.role}>\n{m.content}\n</{m.role}>" for m in messages)
    return (
        "Here is our conversation so far. The assistant turns are your own earlier "
        "replies. Write your next reply to the user's last message.\n\n" + turns
    )


# --- running CLIs -------------------------------------------------------------

Exec = Callable[[list[str], Path, float, dict[str, str] | None], "Completed"]


@dataclass(frozen=True)
class Completed:
    returncode: int
    stdout: str
    stderr: str


def _subprocess(
    args: list[str], cwd: Path, timeout: float, env: dict[str, str] | None
) -> Completed:
    proc = subprocess.run(
        args,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        stdin=subprocess.DEVNULL,
        env=env,
    )
    return Completed(proc.returncode, proc.stdout, proc.stderr)


@dataclass
class CLIProvider:
    """Base for the CLI backends. Subclasses build args and parse stdout."""

    name: str = ""
    binary: str = ""
    exec: Exec = field(default=_subprocess, repr=False)
    timeout: float = 600
    retries: int = 3
    backoff: tuple[float, ...] = (10.0, 30.0, 90.0)

    def complete(self, request: Request) -> Response:
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                return self._once(request)
            except ProviderError as e:
                last = e
                if attempt < self.retries:
                    time.sleep(self.backoff[min(attempt, len(self.backoff) - 1)])
        raise ProviderError(f"{self.name} failed after {self.retries + 1} attempts: {last}")

    def _once(self, request: Request) -> Response:
        with tempfile.TemporaryDirectory(prefix="spine-") as tmp:
            scratch = Path(tmp)
            args = self.args(request, scratch)
            start = time.monotonic()
            try:
                done = self.exec([self.binary, *args], scratch, self.timeout, self.env(scratch))
            except subprocess.TimeoutExpired as e:
                raise ProviderError(f"{self.name} timed out after {self.timeout:.0f}s") from e
            except FileNotFoundError as e:
                raise ProviderError(f"{self.binary} is not installed") from e
            seconds = time.monotonic() - start
        if done.returncode != 0:
            tail = (done.stderr or done.stdout).strip().splitlines()[-3:]
            raise ProviderError(f"{self.name} exited {done.returncode}: {' | '.join(tail)}")
        resp = self.parse(done.stdout)
        return Response(
            resp.text, round(seconds, 2), resp.cost_usd, resp.usage, resp.tool_events, resp.raw
        )

    def env(self, scratch: Path) -> dict[str, str] | None:
        return None

    def args(self, request: Request, scratch: Path) -> list[str]:
        raise NotImplementedError

    def parse(self, stdout: str) -> Response:
        raise NotImplementedError


@dataclass
class ClaudeCLI(CLIProvider):
    """`claude -p` as a plain chat model.

    `--tools ""` removes every built-in tool, `--strict-mcp-config` with no config
    loads no MCP servers, `--disable-slash-commands` drops skills, and
    `--setting-sources ""` ignores user and project settings, which is where
    hooks and permissions live. `--system-prompt` replaces the Claude Code system
    prompt. The CLI still adds a short environment block (date, working directory,
    platform) and a one-line identity, which the probe in `spine probe` shows.
    """

    name: str = "claude"
    binary: str = "claude"

    def args(self, request: Request, scratch: Path) -> list[str]:
        args = [
            "-p",
            render_prompt(request.messages),
            "--output-format",
            "json",
            "--no-session-persistence",
            "--tools",
            "",
            "--strict-mcp-config",
            "--disable-slash-commands",
            "--setting-sources",
            "",
            "--system-prompt",
            request.system,
            "--model",
            request.model,
        ]
        if request.effort:
            args += ["--effort", request.effort]
        if not request.thinking:
            args += ["--settings", json.dumps({"alwaysThinkingEnabled": False})]
        return args

    def parse(self, stdout: str) -> Response:
        out = _last_json_object(stdout)
        if out.get("is_error"):
            raise ProviderError(f"claude reported an error: {str(out.get('result'))[:200]}")
        events = []
        if int(out.get("num_turns") or 1) > 1:
            events.append(f"num_turns={out.get('num_turns')}")
        if out.get("permission_denials"):
            events.append("permission_denials")
        server = (out.get("usage") or {}).get("server_tool_use") or {}
        events += [k for k, v in server.items() if v]
        keep = ("result", "num_turns", "stop_reason", "session_id", "usage", "modelUsage")
        return Response(
            text=str(out.get("result") or ""),
            cost_usd=out.get("total_cost_usd"),
            usage=out.get("usage"),
            tool_events=tuple(events),
            raw={k: out[k] for k in keep if k in out},
        )


# Features that give a Codex session tools. All are turned off; anything left is
# caught from the event stream.
CODEX_DISABLE = (
    "shell_tool",
    "unified_exec",
    "code_mode",
    "code_mode_host",
    "apps",
    "plugins",
    "multi_agent",
    "image_generation",
    "view_image",
    "browser_use",
    "computer_use",
    "in_app_browser",
    "goals",
    "hooks",
    "skill_search",
    "tool_suggest",
    "sleep_tool",
    "memories",
)
_CODEX_QUIET_ITEMS = {"agent_message", "reasoning", "error"}


@dataclass
class CodexCLI(CLIProvider):
    """`codex exec` as a plain chat model.

    Codex reads `~/.codex` for AGENTS.md, skills, MCP servers and memories, so
    each call runs with `CODEX_HOME` pointed at a private home that holds only the
    login and the model-provider settings copied from the user's config. The base
    instructions are replaced with our system prompt through
    `model_instructions_file`, the sandbox is read-only, and every feature that
    adds a tool is disabled. A few tool stubs survive that (`wait`,
    `request_user_input`, and an `exec` that fails closed because its host is
    off). Any tool item in the JSON event stream is recorded in `tool_events`.
    """

    name: str = "codex"
    binary: str = "codex"
    home: Path | None = None
    source_home: Path = field(default_factory=lambda: Path.home() / ".codex")
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def env(self, scratch: Path) -> dict[str, str] | None:
        return {**os.environ, "CODEX_HOME": str(self._private_home())}

    def args(self, request: Request, scratch: Path) -> list[str]:
        (scratch / "system.md").write_text(request.system, encoding="utf-8")
        args = [
            "exec",
            "--json",
            "--skip-git-repo-check",
            "--ephemeral",
            "--ignore-rules",
            "-s",
            "read-only",
            "-C",
            str(scratch),
        ]
        for feature in CODEX_DISABLE:
            args += ["--disable", feature]
        args += ["-c", 'web_search="disabled"']
        args += ["-c", f'model_instructions_file="{scratch / "system.md"}"']
        if request.effort:
            args += ["-c", f'model_reasoning_effort="{request.effort}"']
        return [*args, "-m", request.model, render_prompt(request.messages)]

    def parse(self, stdout: str) -> Response:
        messages: list[str] = []
        events: list[str] = []
        usage: dict[str, Any] | None = None
        kept: list[Any] = []
        for line in stdout.splitlines():
            event = _maybe_json(line)
            if not isinstance(event, dict):
                continue
            item = event.get("item")
            if isinstance(item, dict):
                kind = str(item.get("type"))
                if kind == "agent_message":
                    messages.append(str(item.get("text", "")))
                elif kind not in _CODEX_QUIET_ITEMS:
                    events.append(kind)
                if kind != "reasoning":
                    kept.append(event)
            elif event.get("type") == "turn.completed":
                usage = event.get("usage")
                kept.append(event)
            elif event.get("type") in ("turn.failed", "error"):
                raise ProviderError(f"codex: {json.dumps(event)[:300]}")
        if not messages:
            raise ProviderError("codex produced no agent message")
        return Response(messages[-1], usage=usage, tool_events=tuple(events), raw=kept)

    def _private_home(self) -> Path:
        with self._lock:
            if self.home is None:
                self.home = Path(tempfile.mkdtemp(prefix="spine-codex-home-"))
            config = self.home / "config.toml"
            if not config.exists():
                make_codex_home(self.source_home, self.home)
            return self.home


def make_codex_home(source: Path, dest: Path) -> None:
    """A CODEX_HOME with the login and model provider, and nothing else."""
    dest.mkdir(parents=True, exist_ok=True)
    auth = source / "auth.json"
    if auth.exists():
        shutil.copy2(auth, dest / "auth.json")
        os.chmod(dest / "auth.json", 0o600)
    lines: list[str] = []
    cfg_path = source / "config.toml"
    if cfg_path.exists():
        cfg = tomllib.loads(cfg_path.read_text(encoding="utf-8"))
        if "model_provider" in cfg:
            lines.append(f"model_provider = {json.dumps(cfg['model_provider'])}")
        for name, table in (cfg.get("model_providers") or {}).items():
            lines.append(f"\n[model_providers.{name}]")
            lines += [f"{k} = {json.dumps(v)}" for k, v in table.items()]
    (dest / "config.toml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.chmod(dest / "config.toml", 0o600)


@dataclass
class OpenCodeCLI(CLIProvider):
    """`opencode run` with every tool switched off through inline config.

    Not part of the published run: I haven't verified that the inline config
    removes every tool, and it keeps OpenCode's own system prompt.
    """

    name: str = "opencode"
    binary: str = "opencode"

    def env(self, scratch: Path) -> dict[str, str] | None:
        config = {"tools": {"*": False}, "mcp": {}, "instructions": []}
        return {**os.environ, "OPENCODE_CONFIG_CONTENT": json.dumps(config)}

    def args(self, request: Request, scratch: Path) -> list[str]:
        prompt = f"{request.system}\n\n{render_prompt(request.messages)}"
        return ["run", "--format", "json", "--dir", str(scratch), "-m", request.model, prompt]

    def parse(self, stdout: str) -> Response:
        texts, events, cost = [], [], 0.0
        for line in stdout.splitlines():
            event = _maybe_json(line)
            if not isinstance(event, dict):
                continue
            part = event.get("part") or {}
            if event.get("type") == "text":
                texts.append(str(part.get("text", "")))
            elif event.get("type") == "tool_use":
                events.append(str(part.get("tool", "tool")))
            elif event.get("type") == "step_finish":
                cost += float(part.get("cost") or 0)
        return Response("".join(texts).strip(), cost_usd=cost, tool_events=tuple(events))


@dataclass
class CursorCLI(CLIProvider):
    """`cursor-agent -p --mode ask`. Ask mode is read-only but still has read and
    search tools, so this backend is not tool-free and is not used for results."""

    name: str = "cursor"
    binary: str = "cursor-agent"

    def args(self, request: Request, scratch: Path) -> list[str]:
        prompt = f"{request.system}\n\n{render_prompt(request.messages)}"
        return [
            "-p",
            "--output-format",
            "json",
            "--trust",
            "--mode",
            "ask",
            "--workspace",
            str(scratch),
            "--model",
            request.model,
            prompt,
        ]

    def parse(self, stdout: str) -> Response:
        out = _last_json_object(stdout)
        if out.get("is_error"):
            raise ProviderError(f"cursor reported an error: {str(out.get('result'))[:200]}")
        return Response(str(out.get("result") or ""))


BACKENDS: dict[str, type[CLIProvider]] = {
    "claude": ClaudeCLI,
    "codex": CodexCLI,
    "opencode": OpenCodeCLI,
    "cursor": CursorCLI,
}


def backend(name: str) -> CLIProvider:
    try:
        return BACKENDS[name]()
    except KeyError:
        raise ProviderError(
            f"unknown provider {name!r}; choose from {', '.join(BACKENDS)}"
        ) from None


# --- recording ----------------------------------------------------------------


class Recorder:
    """Every request and response, appended to one JSONL file per model.

    With `offline=True` nothing is ever sent: a request that isn't in the file
    raises `MissingRecording`. That's how the published numbers are rebuilt.
    """

    def __init__(self, inner: Provider | None, path: Path, *, offline: bool = False) -> None:
        self.inner = inner
        self.path = path
        self.offline = offline or inner is None
        self._lock = threading.Lock()
        self._rows: dict[str, dict[str, Any]] = {}
        self.hits = 0
        self.calls = 0
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    self._rows[row["key"]] = row

    def __len__(self) -> int:
        return len(self._rows)

    def complete(self, request: Request) -> Response:
        key = request.key()
        with self._lock:
            row = self._rows.get(key)
        if row is not None:
            with self._lock:
                self.hits += 1
            return Response.from_json(row["response"])
        if self.offline or self.inner is None:
            raise MissingRecording(f"no recording for request {key} in {self.path}")
        response = self.inner.complete(request)
        row = {"key": key, "request": request.to_json(), "response": response.to_json()}
        line = json.dumps(row, ensure_ascii=False, sort_keys=True)
        with self._lock:
            self.calls += 1
            self._rows[key] = row
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        return response


@dataclass
class FakeProvider:
    """A scripted model for tests: `policy(request)` returns the reply text."""

    policy: Callable[[Request], str]
    requests: list[Request] = field(default_factory=list)

    def complete(self, request: Request) -> Response:
        self.requests.append(request)
        return Response(self.policy(request))


def _maybe_json(line: str) -> Any:
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return None


def _last_json_object(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.strip().splitlines()):
        value = _maybe_json(line)
        if isinstance(value, dict):
            return value
    value = _maybe_json(stdout)
    if isinstance(value, dict):
        return value
    raise ProviderError("expected a JSON object on stdout")
