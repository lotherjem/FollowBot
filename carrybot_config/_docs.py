"""Pull the ``#:`` field comments out of a dataclass's source.

The generated config file documents every key with the comment that sits above
that field in the code, so the two cannot drift apart: change the comment, and
the next ``python -m carrybot_config --write`` carries it.
"""

from __future__ import annotations

import inspect
import re
import textwrap
from typing import Dict, List, Tuple

_FIELD = re.compile(r"^(\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:\s*[^=]+?(=.*)?$")
_RULE = re.compile(r"^\s*#\s*--+\s*(.*?)\s*-*\s*#?\s*$")


def field_docs(cls) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    """Return ``({field: comment lines}, {field: section heading before it})``."""
    try:
        source = textwrap.dedent(inspect.getsource(cls))
    except (OSError, TypeError):
        return {}, {}
    lines = source.splitlines()
    # Skip decorators: the body starts after the ``class`` line itself.
    start = next((i for i, l in enumerate(lines) if l.lstrip().startswith("class ")), 0)
    lines = lines[start:]
    body_indent = None
    docs: Dict[str, List[str]] = {}
    headings: Dict[str, str] = {}
    pending: List[str] = []
    heading = ""
    in_docstring = False
    for line in lines[1:]:
        stripped = line.strip()
        if not stripped:
            pending = []
            continue
        indent = len(line) - len(line.lstrip())
        if body_indent is None:
            body_indent = indent
        if indent < body_indent:
            break
        if stripped.startswith(('"""', "'''")):
            quote = stripped[:3]
            if not (stripped.count(quote) >= 2 and len(stripped) > 3):
                in_docstring = not in_docstring
            continue
        if in_docstring or indent != body_indent:
            continue
        if stripped.startswith("def ") or stripped.startswith("@"):
            break
        rule = _RULE.match(line)
        if rule and stripped.startswith("# --"):
            heading = rule.group(1).strip()
            pending = []
            continue
        if stripped.startswith("#:"):
            pending.append(stripped[2:].strip())
            continue
        if stripped.startswith("#"):
            continue
        match = _FIELD.match(line)
        if match:
            name = match.group(2)
            docs[name] = list(pending)
            if heading:
                headings[name] = heading
                heading = ""
        pending = []
    return docs, headings
