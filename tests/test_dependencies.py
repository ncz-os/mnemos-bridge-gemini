import os
from pathlib import Path
import subprocess
import sys
import tomllib


REPO_ROOT = Path(__file__).resolve().parents[1]


def _core_floor(deps: list[str]) -> tuple[int, ...]:
    """Parsed lower bound of the mnemos-bridge-core pin.

    Asserting the exact pin string made every legitimate floor raise fail this
    test, which is backwards: the requirement is a MINIMUM, so a higher floor
    satisfies it by definition.
    """
    for dep in deps:
        if dep.replace(" ", "").startswith("mnemos-bridge-core>="):
            spec = dep.replace(" ", "").split(">=", 1)[1].split(",")[0]
            return tuple(int(p) for p in spec.split(".") if p.isdigit())
    raise AssertionError(f"no mnemos-bridge-core lower bound in {deps}")

def test_dependency_closure_keeps_server_compatible_mcp_and_corrected_core():
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())["project"]

    # The corrected core landed in 0.1.3; a higher floor also satisfies this.
    assert _core_floor(project["dependencies"]) >= (0, 1, 3)
    assert "mcp>=1.0,<2" in project["dependencies"]


def test_missing_mandatory_google_sdk_fails_package_import():
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    result = subprocess.run(
        [sys.executable, "-S", "-c", "import mnemos_bridge_gemini"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "google" in result.stderr
