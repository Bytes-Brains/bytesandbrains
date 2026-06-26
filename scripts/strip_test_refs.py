#!/usr/bin/env python3
"""
Strip the 3-line `#[cfg(test)] #[path = "..._tests.rs"] mod tests;`
reference blocks from .rs files in the public mirror snapshot.

The sibling `*_tests.rs` files are excluded from the mirror by rsync.
This script removes the references to them from production source so
the public crate doesn't `mod tests` a missing file.

Convention (see CLAUDE.md "Test layout"):
- Unit tests live in `src/<path>_tests.rs` next to `src/<path>.rs`.
- The production source references them via:
      #[cfg(test)]
      #[path = "<basename>_tests.rs"]
      mod tests;
- The mirror excludes `**/*_tests.rs` and runs this script to strip
  the 3-line reference.

Usage:
    python3 scripts/strip_test_refs.py <ROOT_DIR>

Idempotent. Exit 0 on success.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# 3-line block: cfg + path + mod tests;, with arbitrary leading
# whitespace preserved on each line for the match. Trailing newline
# eaten so we don't leave an orphan blank line.
BLOCK_RE = re.compile(
    r"[ \t]*#\[cfg\(test\)\][ \t]*\n"
    r"[ \t]*#\[path[ \t]*=[ \t]*\"[A-Za-z0-9_./]+_tests\.rs\"\][ \t]*\n"
    r"[ \t]*mod[ \t]+tests[ \t]*;[ \t]*\n?",
    re.MULTILINE,
)


def strip_file(path: Path) -> int:
    source = path.read_text(encoding="utf-8")
    new_source, n = BLOCK_RE.subn("", source)
    if n > 0:
        path.write_text(new_source, encoding="utf-8")
    return n


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <ROOT_DIR>", file=sys.stderr)
        return 1
    root = Path(sys.argv[1])
    if not root.is_dir():
        print(f"not a directory: {root}", file=sys.stderr)
        return 1

    total = 0
    files = 0
    for rs in sorted(root.rglob("*.rs")):
        if "target" in rs.parts:
            continue
        # Skip the test files themselves — they should be excluded by
        # rsync but if any sneak through, don't touch them.
        if rs.stem.endswith("_tests"):
            continue
        n = strip_file(rs)
        if n:
            files += 1
            total += n
            print(f"  stripped {n} test ref(s) from {rs}")
    print(f"\nTotal: stripped {total} reference(s) from {files} file(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
