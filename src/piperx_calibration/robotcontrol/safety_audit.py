"""Static deployment safety checks for the physical Piper boundary."""
from __future__ import annotations

import ast
from pathlib import Path


FORBIDDEN_PIPER_CALLS = frozenset({"disable", "reset", "electronic_emergency_stop"})


def forbidden_sdk_calls(path: Path | None = None) -> list[tuple[int, str]]:
    """Find torque-releasing/controller-reset calls in our hardware adapter.

    The vendored SDK includes these public APIs and its own tests exercise
    them.  We audit only the service-owned physical adapter: vendored tests are
    never imported by robotcontrol and must not make this check noisy.
    """
    path = path or Path(__file__).with_name("piper_x.py")
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in FORBIDDEN_PIPER_CALLS:
                found.append((node.lineno, node.func.attr))
    return found


def assert_deployment_safe() -> None:
    found = forbidden_sdk_calls()
    if found:
        details = ", ".join(f"line {line}: {name}()" for line, name in found)
        raise RuntimeError(f"forbidden pyAgxArm safety calls found: {details}")


def main() -> None:
    assert_deployment_safe()
    print("robotcontrol Piper-X safety audit OK: no disable/reset/electronic-emergency-stop calls")


if __name__ == "__main__":
    main()
