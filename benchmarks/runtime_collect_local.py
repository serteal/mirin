"""Run the local shared-runtime collection harness only."""

from __future__ import annotations

import sys
from contextlib import contextmanager

from testbed_collect import main as _collect_main


@contextmanager
def _patched_argv(argv: list[str]):
    original = sys.argv
    sys.argv = [original[0], *argv]
    try:
        yield
    finally:
        sys.argv = original


def main() -> int:
    with _patched_argv(list(sys.argv[1:])):
        return int(_collect_main())


if __name__ == "__main__":
    raise SystemExit(main())
