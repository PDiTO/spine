from __future__ import annotations

import json
from pathlib import Path

import pytest

from spine.providers import (
    ClaudeCLI,
    CodexCLI,
    Completed,
    CursorCLI,
    FakeProvider,
    Message,
    MissingRecording,
    OpenCodeCLI,
    ProviderError,
    Recorder,
    Request,
    Response,
    backend,
    make_codex_home,
    render_prompt,
)


def req(*texts: str, provider: str = "claude", thinking: bool = True) -> Request:
    roles = ["user", "assistant"]
    msgs = tuple(Message(roles[i % 2], t) for i, t in enumerate(texts))  # type: ignore[arg-type]
    return Request(provider, "m", "low", "SYSTEM", msgs, thinking)


class FakeExec:
    def __init__(self, stdout: str, returncode: int = 0, stderr: str = "") -> None:
        self.stdout, self.returncode, self.stderr = stdout, returncode, stderr
        self.calls: list[tuple[list[str], Path, dict[str, str] | None]] = []

    def __call__(
        self, args: list[str], cwd: Path, timeout: float, env: dict[str, str] | None
    ) -> Completed:
        self.calls.append((args, cwd, env))
        return Completed(self.returncode, self.stdout, self.stderr)


def test_render_prompt_first_turn_is_plain() -> None:
    assert render_prompt(req("What is 2+2?").messages) == "What is 2+2?"


def test_render_prompt_later_turns_carry_the_transcript() -> None:
    text = render_prompt(req("Q", "A <answer>4</answer>", "Are you sure?").messages)
    assert "your own earlier replies" in text
    assert text.index("<user>\nQ") < text.index("<assistant>\nA") < text.index("Are you sure?")


def test_request_key_is_stable_and_content_sensitive() -> None:
    assert req("a").key() == req("a").key()
    assert req("a").key() != req("b").key()
    assert req("a").key() != req("a", thinking=False).key()
    assert Request.from_json(req("a", "b", "c").to_json()) == req("a", "b", "c")


CLAUDE_OUT = json.dumps(
    {
        "type": "result",
        "is_error": False,
        "result": "Done.\n<answer>4</answer>",
        "num_turns": 1,
        "total_cost_usd": 0.001,
        "usage": {"output_tokens": 9, "server_tool_use": {"web_search_requests": 0}},
        "session_id": "s",
        "permission_denials": [],
    }
)


def test_claude_args_are_pure_chat() -> None:
    fake = FakeExec(CLAUDE_OUT)
    resp = ClaudeCLI(exec=fake, retries=0).complete(req("Q", thinking=False))
    args = fake.calls[0][0]
    assert args[0] == "claude"
    for flag in ("--tools", "--strict-mcp-config", "--disable-slash-commands", "--system-prompt"):
        assert flag in args
    assert args[args.index("--tools") + 1] == ""
    assert args[args.index("--setting-sources") + 1] == ""
    assert json.loads(args[args.index("--settings") + 1]) == {"alwaysThinkingEnabled": False}
    assert args[args.index("--system-prompt") + 1] == "SYSTEM"
    assert resp.text.endswith("<answer>4</answer>")
    assert resp.tool_events == ()
    assert resp.cost_usd == 0.001


def test_claude_flags_tool_use() -> None:
    out = json.loads(CLAUDE_OUT)
    out["num_turns"] = 3
    out["usage"]["server_tool_use"]["web_search_requests"] = 1
    resp = ClaudeCLI(exec=FakeExec(json.dumps(out)), retries=0).complete(req("Q"))
    assert "num_turns=3" in resp.tool_events
    assert "web_search_requests" in resp.tool_events


def test_claude_error_is_raised() -> None:
    out = json.dumps({"is_error": True, "result": "rate limited"})
    with pytest.raises(ProviderError, match="rate limited"):
        ClaudeCLI(exec=FakeExec(out), retries=0).complete(req("Q"))


def test_nonzero_exit_retries_then_fails() -> None:
    fake = FakeExec("", returncode=1, stderr="boom")
    cli = ClaudeCLI(exec=fake, retries=2, backoff=(0.0,))
    with pytest.raises(ProviderError, match="3 attempts"):
        cli.complete(req("Q"))
    assert len(fake.calls) == 3


CODEX_OUT = "\n".join(
    json.dumps(e)
    for e in [
        {"type": "thread.started", "thread_id": "t"},
        {"type": "item.completed", "item": {"type": "error", "message": "code mode off"}},
        {"type": "item.completed", "item": {"type": "reasoning", "text": "hmm"}},
        {"type": "item.completed", "item": {"type": "agent_message", "text": "<answer>4</answer>"}},
        {"type": "turn.completed", "usage": {"output_tokens": 5}},
    ]
)


def test_codex_args_and_private_home(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "auth.json").write_text("{}")
    (src / "AGENTS.md").write_text("be a coding agent")
    (src / "config.toml").write_text(
        'model = "x"\nmodel_provider = "p"\n[model_providers.p]\nname = "P"\n'
        'base_url = "https://example.invalid"\n[mcp_servers.browser]\ncommand = "b"\n'
    )
    fake = FakeExec(CODEX_OUT)
    cli = CodexCLI(exec=fake, retries=0, source_home=src, home=tmp_path / "home")
    resp = cli.complete(req("Q", provider="codex"))
    args, _cwd, env = fake.calls[0]
    assert args[:2] == ["codex", "exec"]
    assert "shell_tool" in args
    assert "code_mode_host" in args
    assert 'web_search="disabled"' in args
    assert any(a.startswith("model_instructions_file=") for a in args)
    assert args[-1] == "Q"
    assert env is not None
    home = Path(env["CODEX_HOME"])
    config = (home / "config.toml").read_text()
    assert "model_providers.p" in config
    assert "mcp_servers" not in config
    assert not (home / "AGENTS.md").exists()
    assert (home / "auth.json").exists()
    assert resp.text == "<answer>4</answer>"
    assert resp.usage == {"output_tokens": 5}
    assert resp.tool_events == ()


def test_codex_flags_tool_items() -> None:
    out = (
        CODEX_OUT
        + "\n"
        + json.dumps(
            {"type": "item.completed", "item": {"type": "command_execution", "command": "ls"}}
        )
    )
    resp = CodexCLI(exec=FakeExec(out), retries=0).parse(out)
    assert resp.tool_events == ("command_execution",)


def test_codex_turn_failure_raises() -> None:
    out = json.dumps({"type": "turn.failed", "error": {"message": "unsupported"}})
    with pytest.raises(ProviderError):
        CodexCLI().parse(out)
    with pytest.raises(ProviderError, match="no agent message"):
        CodexCLI().parse("")


def test_make_codex_home_without_config(tmp_path: Path) -> None:
    make_codex_home(tmp_path / "missing", tmp_path / "home")
    assert (tmp_path / "home" / "config.toml").read_text() == "\n"


def test_other_backends_parse() -> None:
    oc = "\n".join(
        json.dumps(e)
        for e in [
            {"type": "text", "part": {"text": "<answer>4</answer>"}},
            {"type": "tool_use", "part": {"tool": "bash"}},
            {"type": "step_finish", "part": {"cost": 0.5}},
        ]
    )
    resp = OpenCodeCLI().parse(oc)
    assert resp.text == "<answer>4</answer>"
    assert resp.tool_events == ("bash",)
    assert "OPENCODE_CONFIG_CONTENT" in (OpenCodeCLI().env(Path()) or {})
    assert OpenCodeCLI().args(req("Q"), Path("/tmp"))[0] == "run"
    assert CursorCLI().parse(json.dumps({"result": "hi"})).text == "hi"
    assert "--mode" in CursorCLI().args(req("Q"), Path("/tmp"))
    with pytest.raises(ProviderError):
        CursorCLI().parse(json.dumps({"is_error": True, "result": "x"}))


def test_backend_lookup() -> None:
    assert isinstance(backend("claude"), ClaudeCLI)
    with pytest.raises(ProviderError, match="unknown provider"):
        backend("nope")


def test_recorder_records_then_replays_offline(tmp_path: Path) -> None:
    path = tmp_path / "m.jsonl"
    fake = FakeProvider(lambda r: f"<answer>{len(r.messages)}</answer>")
    live = Recorder(fake, path)
    first = live.complete(req("Q"))
    again = live.complete(req("Q"))
    assert first == again
    assert len(fake.requests) == 1
    assert live.calls == 1
    assert live.hits == 1

    offline = Recorder(None, path)
    assert offline.offline
    assert len(offline) == 1
    assert offline.complete(req("Q")).text == "<answer>1</answer>"
    with pytest.raises(MissingRecording):
        offline.complete(req("Q", "A", "B"))


def test_response_round_trip() -> None:
    r = Response("t", 1.5, 0.1, {"a": 1}, ("x",), {"raw": True})
    assert Response.from_json(r.to_json()) == r
