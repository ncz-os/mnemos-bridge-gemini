import os
from pathlib import Path
import subprocess
import sys
import tomllib


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_dependency_closure_keeps_server_compatible_mcp_and_corrected_core():
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())["project"]

    assert "mnemos-bridge-core>=0.1.3" in project["dependencies"]
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
