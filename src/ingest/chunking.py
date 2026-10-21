"""Turn parsed documents into retrieval chunks.

Pure Python, no Spark: the ingest notebook calls it on the driver and the unit
tests run it locally.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from html.parser import HTMLParser

# Layout noise that only hurts retrieval.
SKIP_TYPES = {"page_header", "page_footer", "page_number", "signature"}
HEADING_TYPES = {"title", "section_header"}

TEXT_EXTENSIONS = {".txt", ".md", ".markdown"}

_SEPARATORS = ("\n\n", "\n", ". ", "; ", ", ", " ")
_MD_HEADING = re.compile(r"^#{1,6}\s+(.*)$")


@dataclass(frozen=True)
class Element:
    type: str
    text: str
    page: int | None  # 1-based, None for formats without pages


@dataclass(frozen=True)
class Chunk:
    index: int
    content: str
    page_start: int | None
    page_end: int | None
    section: str | None


def doc_id_for(path: str) -> str:
    return hashlib.sha1(path.encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------


def elements_from_parsed(parsed: dict) -> list[Element]:
    """Normalise the `ai_parse_document` output (VARIANT serialised as JSON)."""
    document = parsed.get("document") or {}
    elements = []
    for raw in document.get("elements") or []:
        etype = (raw.get("type") or "text").lower()
        if etype in SKIP_TYPES:
            continue
        text = _element_text(etype, raw)
        if text:
            elements.append(Element(etype, text, _element_page(raw)))
    return elements


def elements_from_text(text: str) -> list[Element]:
    """Plain text / Markdown: blank lines separate blocks, `#` lines are headings."""
    elements = []
    for block in re.split(r"\n\s*\n", text.replace("\r\n", "\n")):
        lines = block.strip().split("\n")
        while lines and (match := _MD_HEADING.match(lines[0].strip())):
            elements.append(Element("section_header", match.group(1).strip(), None))
            lines = lines[1:]
        body = "\n".join(lines).strip()
        if body:
            elements.append(Element("text", body, None))
    return elements


def _element_text(etype: str, raw: dict) -> str:
    content = (raw.get("content") or "").strip()
    if etype == "table" and "<t" in content:
        return html_table_to_text(content)
    if etype == "figure":
        # description: AI-generated caption; content: NULL, or the data of a readable chart.
        description = (raw.get("description") or "").strip()
        body = "\n".join(part for part in (description, content) if part)
        return f"[Figura] {body}" if body else ""
    return content


def _element_page(raw: dict) -> int | None:
    for box in raw.get("bbox") or []:
        if isinstance(box, dict) and box.get("page_id") is not None:
            return int(box["page_id"]) + 1
    if raw.get("page_id") is not None:
        return int(raw["page_id"]) + 1
    return None


class _TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self._row = []
        elif tag in ("td", "th"):
            self._cell = []
        elif tag == "br" and self._cell is not None:
            self._cell.append(" ")

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self._cell is not None:
            if self._row is not None:
                self._row.append(" ".join("".join(self._cell).split()))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            if any(self._row):
                self.rows.append(self._row)
            self._row = None

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)


def html_table_to_text(html: str) -> str:
    """HTML table -> one `a | b | c` line per row (readable for both LLM and embeddings)."""
    parser = _TableParser()
    parser.feed(html)
    parser.close()
    if not parser.rows:
        return " ".join(re.sub(r"<[^>]+>", " ", html).split())
    return "\n".join(" | ".join(row) for row in parser.rows)


# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------


def chunk_elements(
    elements: list[Element],
    max_chars: int = 2000,
    overlap_chars: int = 200,
    min_chars: int = 400,
) -> list[Chunk]:
    """Pack elements into chunks of at most ~max_chars, tracking pages and section.

    A new heading closes the current chunk once it holds at least `min_chars`, so
    chunks tend to follow the document structure. Chunks closed because they are
    full carry the last `overlap_chars` into the next one.
    """
    chunks: list[Chunk] = []
    parts: list[tuple[str, int | None]] = []
    size = 0
    fresh = False  # whether `parts` holds anything besides the carried overlap
    section: str | None = None
    chunk_section: str | None = None

    def flush(carry: bool) -> None:
        nonlocal parts, size, fresh, chunk_section
        content = "\n\n".join(text for text, _ in parts).strip()
        pages = [page for _, page in parts if page is not None]
        chunks.append(
            Chunk(
                index=len(chunks),
                content=content,
                page_start=min(pages) if pages else None,
                page_end=max(pages) if pages else None,
                section=chunk_section,
            )
        )
        tail = _tail(content, overlap_chars) if carry and overlap_chars > 0 else ""
        parts = [(tail, parts[-1][1])] if tail else []
        size = len(tail)
        fresh = False
        chunk_section = section

    for element in elements:
        if element.type in HEADING_TYPES:
            if fresh and size >= min_chars:
                flush(carry=False)
            section = element.text[:200]
            if not fresh or chunk_section is None:
                chunk_section = section
        for piece in _split_text(element.text, max(max_chars - overlap_chars, 1)):
            if fresh and size + len(piece) > max_chars:
                flush(carry=True)
            parts.append((piece, element.page))
            size += len(piece) + 2
            fresh = True

    if fresh:
        flush(carry=False)
    return chunks


def _split_text(text: str, limit: int) -> list[str]:
    """Split `text` into pieces <= limit, preferring paragraph, line, sentence, word breaks."""
    text = text.strip()
    if len(text) <= limit:
        return [text] if text else []
    sep = next((s for s in _SEPARATORS if s in text), None)
    if sep is None:
        return [text[i : i + limit] for i in range(0, len(text), limit)]

    raw = text.split(sep)
    segments = [s + sep for s in raw[:-1]] + [raw[-1]]
    pieces: list[str] = []
    current = ""
    for segment in segments:
        if len(current) + len(segment) <= limit:
            current += segment
            continue
        if current.strip():
            pieces.append(current.strip())
        if len(segment) <= limit:
            current = segment
        else:
            # The stripped segment no longer contains `sep`, so recursion moves
            # on to a finer separator and terminates.
            sub = _split_text(segment, limit)
            pieces.extend(sub[:-1])
            current = sub[-1] if sub else ""
    if current.strip():
        pieces.append(current.strip())
    return pieces


def _tail(text: str, n: int) -> str:
    """Last ~n chars of `text`, starting at a line, sentence or word boundary."""
    if len(text) <= n:
        return text
    cut = text[-n:]
    for boundary in ("\n", ". ", " "):
        pos = cut.find(boundary)
        # Never drop more than half of the overlap to reach a nicer boundary.
        if 0 <= pos < len(cut) // 2:
            return cut[pos + len(boundary) :].lstrip()
    return cut


# --------------------------------------------------------------------------
# Output rows
# --------------------------------------------------------------------------


def render_document_text(elements: list[Element]) -> str:
    """Whole-document text with page markers, used by the Studio features."""
    blocks = []
    page = None
    for element in elements:
        if element.page is not None and element.page != page:
            page = element.page
            blocks.append(f"[Pagina {page}]")
        blocks.append(f"## {element.text}" if element.type in HEADING_TYPES else element.text)
    return "\n\n".join(blocks)


def build_chunk_rows(
    doc_id: str,
    notebook: str,
    path: str,
    file_name: str,
    elements: list[Element],
    **chunk_options,
) -> list[dict]:
    rows = []
    for chunk in chunk_elements(elements, **chunk_options):
        header = f"Documento: {file_name}\n"
        if chunk.section:
            header += f"Sezione: {chunk.section}\n"
        rows.append(
            {
                "chunk_id": f"{doc_id}-{chunk.index:05d}",
                "doc_id": doc_id,
                "notebook": notebook,
                "path": path,
                "file_name": file_name,
                "chunk_index": chunk.index,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "section": chunk.section,
                "content": chunk.content,
                # Indexed text: the header gives the embedding (and the keyword
                # side of hybrid search) the document/section context.
                "embed_text": f"{header}\n{chunk.content}",
            }
        )
    return rows
