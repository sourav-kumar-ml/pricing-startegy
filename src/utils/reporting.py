from pathlib import Path
from typing import Iterable


def write_markdown(path: Path, title: str, sections: Iterable[tuple[str, list[str]]]) -> None:
    lines = [f"# {title}", ""]
    for header, body_lines in sections:
        lines.append(f"## {header}")
        lines.extend(body_lines)
        lines.append("")
    path.write_text("\n".join(lines).strip() + "\n")
