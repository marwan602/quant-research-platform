import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_javascript_rebalance_suite():
    """
    Executes the pure JavaScript rebalance engine unit tests in Node.js.
    Guarantees that the frontend JavaScript implementation in docs/rebalance.js
    strictly passes all conservation, boundary, validation, and parsing tests.
    """
    js_test_file = PROJECT_ROOT / "tests" / "test_rebalance_js.js"
    assert js_test_file.exists(), f"Missing JS test file: {js_test_file}"

    proc = subprocess.run(
        ["node", str(js_test_file)],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, f"JS Rebalance Tests Failed:\n{proc.stdout}\n{proc.stderr}"
    assert "ALL JAVASCRIPT REBALANCE TESTS PASSED!" in proc.stdout
