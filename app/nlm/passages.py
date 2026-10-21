"""Retrieved passages and the [n] citations that point at them."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from html import escape

_CITATION = re.compile(r"\[(\d+(?:\s*[-–,;]\s*\d+)*)\]")


@dataclass(frozen=True)
class Passage:
    n: int  # 1-based number shown to the model and used in citations
    chunk_id: str
    doc_id: str
    file_name: str
    page_start: int | None
    page_end: int | None
    section: str | None
    content: str
    score: float | None = None

    @property
    def location(self) -> str:
        pages = page_label(self.page_start, self.page_end)
        return f"{self.file_name}, {pages}" if pages else self.file_name

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Passage":
        return cls(**{k: data.get(k) for k in cls.__dataclass_fields__})


def page_label(start: int | None, end: int | None) -> str:
    if start is None:
        return ""
    if end is None or end == start:
        return f"p. {start}"
    return f"pp. {start}-{end}"


def cited_numbers(text: str) -> list[int]:
    """Numbers cited as [2], [1, 3] or [4-6], in order of first appearance."""
    numbers: list[int] = []
    for match in _CITATION.finditer(text):
        for part in re.split(r"[,;]", match.group(1)):
            bounds = [int(x) for x in re.split(r"[-–]", part)]
            is_range = len(bounds) == 2 and 0 <= bounds[1] - bounds[0] <= 50
            for n in range(bounds[0], bounds[1] + 1) if is_range else bounds:
                if n not in numbers:
                    numbers.append(n)
    return numbers


def strip_citations(text: str) -> str:
    """Drop [n] markers, e.g. from past answers whose numbers refer to older passages."""
    return _CITATION.sub("", text).replace(" .", ".").replace(" ,", ",")


def cited_passages(text: str, passages: list[Passage]) -> list[Passage]:
    by_number = {p.n: p for p in passages}
    return [by_number[n] for n in cited_numbers(text) if n in by_number]


def format_passages(passages: list[Passage]) -> str:
    blocks = []
    for p in passages:
        attrs = [f'id="{p.n}"', f'documento="{escape(p.file_name)}"']
        if pages := page_label(p.page_start, p.page_end):
            attrs.append(f'pagine="{pages}"')
        if p.section:
            attrs.append(f'sezione="{escape(p.section)}"')
        content = p.content.replace("</passaggio>", "")
        blocks.append(f"<passaggio {' '.join(attrs)}>\n{content}\n</passaggio>")
    return "\n\n".join(blocks)
