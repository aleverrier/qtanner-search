import subprocess
import sys


def test_progressive_help_aliases() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "qtanner.search", "progressive", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    output = (proc.stdout or "") + (proc.stderr or "")
    assert proc.returncode == 0
    assert "--classical-enum-kmax" in output
    assert "--kmax" in output
