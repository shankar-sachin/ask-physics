"""The shared look of the shell scripts (``scripts/lib.sh``): the helpers a script calls, the
pure-sh renderer that works with no Python, and the installer's copy of it.

Run for real with a pseudo-terminal where animation matters, and piped where plain lines must
come out with no escape codes.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / "scripts" / "lib.sh"
INSTALL = ROOT / "install.sh"
BEGIN, END = "# >>> renderer", "# <<< renderer"

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell scripts")


def _env(tmp_path: Path, **extra: str) -> dict[str, str]:
    env = {
        **os.environ,
        "NO_CAFFEINATE": "1",
        "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"],
        "PYTHONPATH": str(ROOT / "src"),
        "ASKPHYSICS_LOG_DIR": str(tmp_path / "logs"),
        "COLUMNS": "100",
    }
    for name in ("DRY_RUN", "NO_COLOR", "ASKPHYSICS_UI", "FORCE_COLOR"):
        env.pop(name, None)
    env.update(extra)
    return env


def _script(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "demo.sh"
    path.write_text(f'. "{LIB}"\n{body}\n')
    return path


def _sh(tmp_path: Path, body: str, **env: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["sh", str(_script(tmp_path, body))],
        env=_env(tmp_path, **env),
        capture_output=True,
        text=True,
        check=False,
    )


def _terminal(tmp_path: Path, body: str, **env: str) -> str:
    """The raw bytes a script writes to a 100-column terminal."""
    import fcntl
    import pty
    import struct
    import termios

    pid, fd = pty.fork()
    if pid == 0:  # pragma: no cover - the child process
        os.environ.update(
            _env(tmp_path, TERM="xterm-256color", COLORTERM="truecolor", LANG="C.UTF-8", **env)
        )
        os.execvp("sh", ["sh", str(_script(tmp_path, body))])
    fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", 32, 100, 0, 0))
    out = b""
    while True:
        try:
            chunk = os.read(fd, 65536)
        except OSError:
            break
        if not chunk:
            break
        out += chunk
    os.waitpid(pid, 0)
    return out.decode("utf-8", "replace")


def _text(raw: str) -> str:
    """What a terminal shows once carriage returns and escape codes are applied (one line each)."""
    lines = []
    for line in raw.replace("\r\n", "\n").split("\n"):
        line = line.split("\r")[-1]
        shown = ""
        for part in re.split(r"(\x1b\[[0-9;?]*[A-Za-z])", line):
            moved = re.fullmatch(r"\x1b\[(\d+)G", part)
            if moved:  # move to a column: pad out to it
                shown = shown.ljust(int(moved.group(1)) - 1)
            elif not part.startswith("\x1b["):
                shown += part
        lines.append(shown.rstrip())
    return "\n".join(lines)


def _times_blurred(text: str) -> str:
    return re.sub(r"\b\d+h \d\dm \d\ds\b|\b\d+m \d\ds\b|\b\d+s\b", "Ns", text)


DEMO = """
printf '#!/bin/sh\\necho working\\n' > "$TMP/ok.sh"
printf '#!/bin/sh\\necho one; echo two; exit 2\\n' > "$TMP/bad.sh"
ui_begin "Demo" "About it" 3 8
phase "One" sh "$TMP/ok.sh"
gate "Two" last sh "$TMP/bad.sh"
ui_result skip "Three" "not installed"
ui_end "All good" "Not good" "make x::do x"
"""


def test_off_a_terminal_every_helper_prints_plain_lines_with_no_escape_codes(
    tmp_path: Path,
) -> None:
    for mode in ("sh", ""):  # the pure-sh renderer, then the Python one
        result = _sh(tmp_path, DEMO, TMP=str(tmp_path), ASKPHYSICS_UI=mode)
        assert result.returncode == 1, result.stderr  # a gate failed
        assert "\x1b" not in result.stdout and "\x1b" not in result.stderr
        assert _times_blurred(result.stdout).splitlines() == [
            "Ask Physics: Demo (3 steps)",
            "  About it",
            "start: [1/3] One",
            "working",
            "done: [1/3] One (Ns)",
            "start: [2/3] Two",
            "one",
            "two",
            "failed: [2/3] Two (exit 2, Ns)",
            f"  command: sh {tmp_path}/bad.sh",
            "skipped: [3/3] Three - not installed",
            "failed: Not good (1 of 3 steps failed: Two; Ns)",
            "  next: make x  - do x",
        ]


def test_the_python_and_the_sh_renderers_print_the_same_plain_lines(tmp_path: Path) -> None:
    body = DEMO + "\n"
    both = [_sh(tmp_path, body, TMP=str(tmp_path), ASKPHYSICS_UI=mode) for mode in ("sh", "py")]
    assert both[0].stdout == both[1].stdout or _times_blurred(both[0].stdout) == _times_blurred(
        both[1].stdout
    )


def test_no_color_turns_off_colour_and_animation_even_on_a_terminal(tmp_path: Path) -> None:
    for mode in ("sh", ""):
        raw = _terminal(
            tmp_path, DEMO.replace("exit 2", "exit 0"), TMP=str(tmp_path), NO_COLOR="1",
            ASKPHYSICS_UI=mode,
        )  # fmt: skip
        assert "\x1b" not in raw and "start: [1/3] One" in raw


def test_a_terminal_gets_a_spinner_that_turns_into_a_tick_with_the_time(tmp_path: Path) -> None:
    body = """
printf '#!/bin/sh\\nsleep 1\\necho done\\n' > "$TMP/slow.sh"
ui_begin "Demo" "About it" 2 6
phase "Slow" sh "$TMP/slow.sh"
phase "Slow again" sh "$TMP/slow.sh"
ui_end "Fine"
"""
    for mode in ("sh", ""):
        raw = _terminal(tmp_path, body, TMP=str(tmp_path), ASKPHYSICS_UI=mode)
        assert any(frame in raw for frame in "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"), mode
        text = _text(raw)
        assert "◉ Ask Physics  ·  Demo" in text
        assert re.search(r"✓ 1/2 Slow\s+1s", text), (mode, text)
        assert re.search(r"✓ 2/2 Slow again\s+1s", text)
        assert "✓ Fine" in text and "all 2 steps" in text


def test_a_failed_step_shows_its_command_the_end_of_its_output_and_the_log(
    tmp_path: Path,
) -> None:
    body = """
printf '#!/bin/sh\\necho first; echo boom >&2; exit 3\\n' > "$TMP/bad.sh"
phase "Breaks" sh "$TMP/bad.sh"
"""
    for mode in ("sh", ""):
        raw = _terminal(tmp_path, body, TMP=str(tmp_path), ASKPHYSICS_UI=mode)
        text = _text(raw)
        assert "✗ Breaks" in text and "exit 3" in text, (mode, text)
        assert f"sh {tmp_path}/bad.sh" in text.replace("'", "")
        assert "boom" in text and "first" in text
        collapsed = re.sub(r"\s+", "", text)  # a long path may wrap across lines
        log = re.search(re.escape(str(tmp_path / "logs")) + r"\S*?/01-breaks\.log", collapsed)
        assert log and Path(log.group(0)).read_text() == "first\nboom\n"


def test_a_failed_step_stops_the_script_and_a_gate_lets_it_go_on(tmp_path: Path) -> None:
    stop = _sh(tmp_path, 'set -eu\nphase "Bad" false\necho after', ASKPHYSICS_UI="sh")
    assert stop.returncode == 1 and "after" not in stop.stdout
    go_on = _sh(
        tmp_path,
        'set -eu\ngate "Bad" - false\ngate "Good" - true\nui_end "ok" "bad"\necho after',
        ASKPHYSICS_UI="sh",
    )
    assert go_on.returncode == 1 and "after" not in go_on.stdout
    assert "failed: bad (1 of 0 steps failed" not in go_on.stdout
    assert "failed: bad (failed: Bad" in _times_blurred(go_on.stdout)


def test_dry_run_prints_the_step_title_and_the_command_and_never_runs_it(tmp_path: Path) -> None:
    result = _sh(
        tmp_path,
        'ui_begin "Demo" "About" 2\nphase "Do it" touch "$TMP/never"\nui_end "Done"',
        TMP=str(tmp_path),
        DRY_RUN="1",
    )
    assert result.returncode == 0 and not (tmp_path / "never").exists()
    assert result.stdout.splitlines()[2:4] == ["-- Do it", f"+ touch {tmp_path}/never"]


def test_the_ready_panel_is_the_same_width_in_python_and_in_sh(tmp_path: Path) -> None:
    body = (
        'ui_ready "Ask Physics is ready" "make ask::ask something" "sh scripts/check.sh::run gates"'
    )
    boxes = []
    for mode in ("sh", ""):
        text = _text(_terminal(tmp_path, body, ASKPHYSICS_UI=mode))
        panel = [
            line
            for line in text.splitlines()
            if line.startswith("  ╭") or "│" in line or "╰" in line
        ]
        assert panel[0].startswith("  ╭─ ✓ Ask Physics is ready") and panel[-1].startswith("  ╰")
        assert {len(line) for line in panel} == {100}
        boxes.append(panel)
    assert boxes[0] == boxes[1]


def test_info_warn_and_fail_are_plain_off_a_terminal_and_fail_exits(tmp_path: Path) -> None:
    result = _sh(tmp_path, 'info "hello"\nwarn "careful"\nfail "no way"\necho unreachable')
    assert result.stdout == "hello\n"
    assert result.stderr == "warning: careful\nerror: no way\n"
    assert result.returncode == 1


def test_a_terminal_styles_info_warn_and_fail_with_the_theme(tmp_path: Path) -> None:
    raw = _terminal(tmp_path, 'info "hello"\nwarn "careful"\nfail "no way"')
    assert "38;2;189;147;249m" in raw  # the accent arrow
    assert "38;2;241;250;140m" in raw  # the warning yellow
    assert "38;2;255;110;110m" in raw  # the failure red
    assert "\u203a hello" in _text(raw) and "✗ no way" in _text(raw)


def test_installers_copy_of_the_renderer_is_the_same_text_as_libs() -> None:
    def block(path: Path) -> str:
        text = path.read_text()
        return text[text.index(BEGIN) : text.index(END)]

    assert block(INSTALL) == block(LIB)


def test_the_renderer_works_with_no_python_at_all(tmp_path: Path) -> None:
    # The installer runs before there is anything installed: only sh and the basics.
    script = tmp_path / "bare.sh"
    lib = LIB.read_text()
    script.write_text(lib[lib.index(BEGIN) : lib.index(END)] + '\nfb_step "Hello" "" "" "" true\n')
    env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path)}
    result = subprocess.run(
        ["sh", str(script)], env=env, capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert _times_blurred(result.stdout).splitlines() == ["start: Hello", "done: Hello (Ns)"]


def test_build_site_prints_its_stages_as_plain_lines_without_askphysics(tmp_path: Path) -> None:
    env = _env(tmp_path, ASKPHYSICS_UI="sh")
    result = subprocess.run(
        ["sh", str(ROOT / "scripts" / "build_site.sh"), str(tmp_path / "site")],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    lines = _times_blurred(result.stdout).splitlines()
    assert lines[0] == "Ask Physics: Build site (3 steps)"
    assert "done: [1/3] Page and artwork (Ns)" in lines
    assert any(re.fullmatch(r"\d+ files written", line) for line in lines)
    assert any("files in askphysics.tar.gz" in line for line in lines)
    assert (tmp_path / "site" / "index.html").is_file()
