#!/usr/bin/env python3
"""Format markdown files by joining multi-line paragraphs into single lines."""

import re
import sys
from pathlib import Path


def is_standalone(stripped: str) -> bool:
    """Lines that are always emitted on their own, never merged."""
    if stripped == "":
        return True
    if stripped.startswith("#"):
        return True
    if stripped.startswith(">"):
        return True
    if re.match(r"!\[", stripped):
        return True
    if re.match(r"<(img|div|/div|picture|figure|/figure|source)", stripped, re.IGNORECASE):
        return True
    return False


def is_list_item(stripped: str) -> bool:
    """Numbered or bulleted list item that starts a new mergeable block."""
    return bool(re.match(r"\d+\.\s", stripped) or re.match(r"[-*+]\s", stripped))


def is_table_row(stripped: str) -> bool:
    """GFM/CommonMark pipe table row; must not be merged with adjacent prose."""
    if not stripped:
        return False
    # Leading | is the usual form; avoids merging | col | rows into paragraphs.
    if stripped.startswith("|"):
        return True
    # Separator-only rows occasionally omit the first pipe in some exporters.
    if "|" in stripped and re.match(r"^[\s|:\-]+$", stripped):
        return True
    return False


def merge_lines(buffer: list[str]) -> str:
    """Join buffered lines into a single line, handling word-break hyphens."""
    if not buffer:
        return ""
    result = buffer[0]
    for line in buffer[1:]:
        if result.endswith("-") and line and line[0].islower():
            result += line
        else:
            result += " " + line
    return result


def format_markdown(text: str) -> str:
    lines = text.split("\n")
    output: list[str] = []
    buffer: list[str] = []
    in_code_block = False
    in_math_block = False

    def flush():
        nonlocal buffer
        if buffer:
            output.append(merge_lines(buffer))
            buffer = []

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("```"):
            flush()
            in_code_block = not in_code_block
            output.append(line)
            continue

        if in_code_block:
            output.append(line)
            continue

        if not in_math_block and stripped.startswith("$$"):
            flush()
            if stripped == "$$" or not stripped.endswith("$$"):
                in_math_block = True
            elif stripped.count("$$") == 2:
                pass  # single-line $$ ... $$, no state change
            output.append(line)
            continue

        if in_math_block:
            output.append(line)
            if stripped.endswith("$$"):
                in_math_block = False
            continue

        if is_standalone(stripped) or is_table_row(stripped):
            flush()
            output.append(line)
        elif is_list_item(stripped):
            flush()
            buffer = [stripped]
        else:
            buffer.append(stripped)

    flush()
    return "\n".join(output)


def main():
    if len(sys.argv) < 2:
        print("Usage: python format_md.py <file.md> [--inplace]")
        sys.exit(1)

    filepath = Path(sys.argv[1])
    inplace = "--inplace" in sys.argv

    if not filepath.exists():
        print(f"Error: {filepath} not found")
        sys.exit(1)

    text = filepath.read_text(encoding="utf-8")
    result = format_markdown(text)

    if inplace:
        filepath.write_text(result, encoding="utf-8")
        print(f"Formatted: {filepath}")
    else:
        print(result)


if __name__ == "__main__":
    main()
