#!/usr/bin/env python3
"""
One-time migration: extract `#[cfg(test)] mod tests { ... }` blocks
from .rs files into sibling `*_tests.rs` files.

For `src/foo/bar.rs` with an inline test block, the body is moved to
`src/foo/bar_tests.rs` and the original block becomes:

    #[cfg(test)]
    #[path = "bar_tests.rs"]
    mod tests;

The new sibling test files are then excluded from the public mirror
(via mirror-public.yml rsync rules), and the 3-line `#[path]` reference
is stripped from .rs files during the mirror snapshot build.

Local `cargo test` continues to work: the test module is still a child
of its parent and retains private-item visibility via `use super::*;`.

Usage:
    python3 scripts/extract_inline_tests.py <ROOT_DIR>

Walks ROOT_DIR for .rs files, extracts inline test modules in-place.
Refuses to overwrite an existing `_tests.rs` file. Idempotent — running
again after a successful run is a no-op because the inline blocks are
already gone.
"""

from __future__ import annotations

import re
import sys
import textwrap
from pathlib import Path

CFG_TEST_RE = re.compile(r"^[ \t]*#\[cfg\(test\)\][ \t]*$")
MOD_LINE_RE = re.compile(
    r"^[ \t]*(?:pub(?:\([^)]*\))?[ \t]+)?mod[ \t]+([A-Za-z_][A-Za-z0-9_]*)[ \t]*\{[ \t]*$"
)


def find_matching_brace(source: str, open_idx: int) -> int | None:
    assert source[open_idx] == "{"
    depth = 0
    i = open_idx
    n = len(source)
    while i < n:
        c = source[i]
        # Block comment.
        if c == "/" and i + 1 < n and source[i + 1] == "*":
            depth_block = 1
            i += 2
            while i < n and depth_block > 0:
                if source[i] == "/" and i + 1 < n and source[i + 1] == "*":
                    depth_block += 1
                    i += 2
                elif source[i] == "*" and i + 1 < n and source[i + 1] == "/":
                    depth_block -= 1
                    i += 2
                else:
                    i += 1
            continue
        # Line comment.
        if c == "/" and i + 1 < n and source[i + 1] == "/":
            nl = source.find("\n", i)
            i = n if nl == -1 else nl + 1
            continue
        # Raw string.
        if c == "r" and i + 1 < n and source[i + 1] in ('"', "#"):
            j = i + 1
            hashes = 0
            while j < n and source[j] == "#":
                hashes += 1
                j += 1
            if j < n and source[j] == '"':
                terminator = '"' + "#" * hashes
                end = source.find(terminator, j + 1)
                if end == -1:
                    return None
                i = end + len(terminator)
                continue
        # Byte-string prefix.
        if c == "b" and i + 1 < n and source[i + 1] in ('"', "r"):
            i += 1
            continue
        # Regular string.
        if c == '"':
            j = i + 1
            while j < n:
                if source[j] == "\\":
                    j += 2
                    continue
                if source[j] == '"':
                    break
                j += 1
            i = j + 1
            continue
        # Char literal / lifetime.
        if c == "'":
            if i + 1 < n and (source[i + 1].isalpha() or source[i + 1] == "_"):
                j = i + 1
                while j < n and (source[j].isalnum() or source[j] == "_"):
                    j += 1
                if j < n and source[j] == "'":
                    i = j + 1
                else:
                    i = j
                continue
            j = i + 1
            if j < n and source[j] == "\\":
                j += 2
                while j < n and source[j] != "'":
                    j += 1
            else:
                j = i + 2
            if j < n and source[j] == "'":
                i = j + 1
                continue
            i += 1
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return None


def extract_file(path: Path) -> tuple[bool, str | None]:
    """Extract the inline test module from `path`. Returns (changed, err).

    On success creates a sibling `<stem>_tests.rs` file containing the
    test body and rewrites `path` to point at it via `#[path = ...]`.
    """
    source = path.read_text(encoding="utf-8")
    lines = source.splitlines(keepends=True)

    byte_offsets: list[int] = []
    off = 0
    for ln in lines:
        byte_offsets.append(off)
        off += len(ln)

    cfg_indices = [i for i, ln in enumerate(lines) if CFG_TEST_RE.match(ln)]
    if not cfg_indices:
        return False, None
    if len(cfg_indices) > 1:
        return False, f"multiple #[cfg(test)] attributes — manual review"

    i = cfg_indices[0]
    j = i + 1
    while j < len(lines):
        stripped = lines[j].strip()
        if stripped == "" or stripped.startswith("//"):
            j += 1
            continue
        break
    if j >= len(lines):
        return False, "#[cfg(test)] at EOF"

    m = MOD_LINE_RE.match(lines[j])
    if not m:
        return False, f"unrecognized item after #[cfg(test)]: {lines[j].strip()[:60]!r}"

    mod_name = m.group(1)
    if mod_name != "tests":
        return False, f"#[cfg(test)] mod `{mod_name}` (expected `tests`) — manual review"

    mod_line_start = byte_offsets[j]
    brace_open = source.find("{", mod_line_start)
    if brace_open == -1:
        return False, "no `{` on mod tests line"
    brace_close = find_matching_brace(source, brace_open)
    if brace_close is None:
        return False, "no matching `}` for mod tests"

    # Body is everything between the `{` and `}` (exclusive). Dedent
    # by one level (4 spaces) so the extracted file isn't artificially
    # indented.
    body = source[brace_open + 1 : brace_close]
    # Trim a single leading newline.
    if body.startswith("\n"):
        body = body[1:]
    # Trim trailing whitespace on the final line.
    body = body.rstrip() + "\n"
    body = textwrap.dedent(body)

    # Determine sibling test-file name. For `src/foo/bar.rs` it's
    # `src/foo/bar_tests.rs`. For directory modules like
    # `src/foo/mod.rs`, the stem is "mod" — keep the suffix uniform.
    sibling = path.with_name(path.stem + "_tests.rs")
    if sibling.exists():
        return False, f"sibling test file already exists: {sibling}"

    # Synthesize the test file with a brief comment header.
    rel_src = path.name
    header = (
        f"//! Inline unit tests for `{rel_src}`, extracted into this\n"
        f"//! sibling file so they can be excluded from the public mirror.\n"
        f"//! Imported by `{rel_src}` via `#[cfg(test)] #[path = ...] mod tests;`.\n"
        f"//!\n"
        f"//! This file is gated by `#[cfg(test)]` at the parent module, so it\n"
        f"//! is only compiled during `cargo test`.\n"
        f"\n"
    )
    sibling.write_text(header + body, encoding="utf-8")

    # Rewrite the source: replace lines [i .. last line of `}`] with the
    # 3-line reference. Preserve the original cfg-line indent so the
    # surrounding style matches.
    indent_match = re.match(r"[ \t]*", lines[i])
    indent = indent_match.group(0) if indent_match else ""
    replacement = (
        f"{indent}#[cfg(test)]\n"
        f"{indent}#[path = \"{sibling.name}\"]\n"
        f"{indent}mod tests;\n"
    )

    strip_start = byte_offsets[i]
    strip_end = brace_close + 1
    if strip_end < len(source) and source[strip_end] == "\n":
        strip_end += 1

    new_source = source[:strip_start] + replacement + source[strip_end:]
    path.write_text(new_source, encoding="utf-8")

    return True, None


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <ROOT_DIR>", file=sys.stderr)
        return 1
    root = Path(sys.argv[1])
    if not root.is_dir():
        print(f"not a directory: {root}", file=sys.stderr)
        return 1

    extracted = 0
    skipped = 0
    errors: list[str] = []

    for rs in sorted(root.rglob("*.rs")):
        if "target" in rs.parts:
            continue
        # Don't try to extract from a freshly-extracted *_tests.rs (no
        # `cfg(test)` block, so it'd be a no-op — but skip just in case
        # the file name pattern matches accidentally).
        if rs.stem.endswith("_tests"):
            continue
        changed, err = extract_file(rs)
        if err:
            errors.append(f"{rs}: {err}")
            continue
        if changed:
            extracted += 1
            print(f"  extracted {rs} → {rs.with_name(rs.stem + '_tests.rs').name}")
        else:
            skipped += 1

    print(f"\nExtracted: {extracted}, skipped: {skipped}")
    if errors:
        print("\nErrors:", file=sys.stderr)
        for e in errors:
            print(f"  {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
