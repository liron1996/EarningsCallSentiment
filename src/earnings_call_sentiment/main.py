"""Top-level CLI: `python -m earnings_call_sentiment.main <command> [...args]`.

Dispatches to phase modules. Each subcommand forwards remaining argv to the module's
own argparse, so per-phase flags work the same as calling the module directly.
"""
from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print("Usage: python -m earnings_call_sentiment.main <parse|extract> [...args]")
        return 2

    command, rest = argv[0], argv[1:]

    if command == "parse":
        from . import parse
        return parse.main(rest)
    if command == "extract":
        from . import extract
        return extract.main(rest)
    if command in ("-h", "--help", "help"):
        print(__doc__)
        return 0

    print(f"Unknown command: {command!r}. Available: parse, extract")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
