from chunking import (
    Element,
    build_chunk_rows,
    chunk_elements,
    doc_id_for,
    elements_from_parsed,
    elements_from_text,
    html_table_to_text,
    render_document_text,
    _split_text,
)

PARSED = {
    "document": {
        "pages": [{"id": 0}, {"id": 1}],
        "elements": [
            {"id": 0, "type": "page_header", "content": "ACME S.p.A. - riservato", "bbox": [{"page_id": 0}]},
            {"id": 1, "type": "title", "content": "Relazione annuale", "bbox": [{"page_id": 0}]},
            {"id": 2, "type": "text", "content": "Il fatturato è cresciuto del 12%.", "bbox": [{"page_id": 0}]},
            {
                "id": 3,
                "type": "table",
                "content": "<table><tr><th>Anno</th><th>Ricavi</th></tr><tr><td>2025</td><td>10</td></tr></table>",
                "bbox": [{"page_id": 1}],
            },
            {"id": 4, "type": "figure", "content": None, "description": "Grafico dei ricavi", "bbox": [{"page_id": 1}]},
            {"id": 5, "type": "page_number", "content": "2", "bbox": [{"page_id": 1}]},
            {"id": 6, "type": "signature", "content": "M. Rossi", "bbox": [{"page_id": 1}]},
        ],
    },
    "error_status": [],
}


def test_elements_from_parsed_skips_noise_and_converts_pages():
    elements = elements_from_parsed(PARSED)
    assert [e.type for e in elements] == ["title", "text", "table", "figure"]
    assert [e.page for e in elements] == [1, 1, 2, 2]
    assert elements[2].text == "Anno | Ricavi\n2025 | 10"
    assert elements[3].text == "[Figura] Grafico dei ricavi"


def test_elements_from_parsed_tolerates_missing_document():
    assert elements_from_parsed({"error_status": [{"error_message": "boom"}]}) == []


def test_html_table_to_text_falls_back_to_stripping_tags():
    assert html_table_to_text("<b>solo</b> testo") == "solo testo"


def test_elements_from_text_markdown_headings():
    elements = elements_from_text("# Titolo\nPrimo paragrafo.\n\n## Sezione\n\nSecondo paragrafo.")
    assert [(e.type, e.text) for e in elements] == [
        ("section_header", "Titolo"),
        ("text", "Primo paragrafo."),
        ("section_header", "Sezione"),
        ("text", "Secondo paragrafo."),
    ]
    assert all(e.page is None for e in elements)


def test_split_text_respects_limit_and_terminates():
    text = "parola " * 500 + "x" * 1000
    pieces = _split_text(text, 300)
    assert all(0 < len(p) <= 300 for p in pieces)
    assert "".join(pieces).replace(" ", "") == text.replace(" ", "")


def test_chunks_respect_size_and_track_pages():
    elements = [Element("text", f"Frase numero {i} del documento di prova.", page=i // 10 + 1) for i in range(200)]
    chunks = chunk_elements(elements, max_chars=500, overlap_chars=50)
    assert len(chunks) > 5
    assert all(len(c.content) <= 500 + 50 for c in chunks)
    assert chunks[0].page_start == 1
    assert chunks[-1].page_end == 20
    assert all(c.page_start <= c.page_end for c in chunks)
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_chunks_carry_overlap_between_full_chunks():
    elements = [Element("text", f"Frase {i}.", page=1) for i in range(300)]
    chunks = chunk_elements(elements, max_chars=300, overlap_chars=40)
    for previous, current in zip(chunks, chunks[1:]):
        # The next chunk starts with a suffix of the previous one, cut at a line break.
        assert current.content.startswith("Frase ")
        assert any(previous.content.endswith(current.content[:n]) for n in range(10, 41))


def test_headings_start_new_chunks_and_set_section():
    elements = [
        Element("title", "Capitolo 1", 1),
        Element("text", "a" * 450, 1),
        Element("section_header", "Capitolo 2", 2),
        Element("text", "b" * 100, 2),
    ]
    chunks = chunk_elements(elements, max_chars=2000, overlap_chars=200, min_chars=400)
    assert [c.section for c in chunks] == ["Capitolo 1", "Capitolo 2"]
    assert chunks[1].content.startswith("Capitolo 2")
    assert (chunks[1].page_start, chunks[1].page_end) == (2, 2)


def test_small_sections_are_merged():
    elements = [
        Element("section_header", "A", 1),
        Element("text", "breve", 1),
        Element("section_header", "B", 1),
        Element("text", "anche breve", 1),
    ]
    chunks = chunk_elements(elements, min_chars=400)
    assert len(chunks) == 1
    assert chunks[0].section == "A"


def test_no_chunk_is_only_overlap():
    elements = [Element("text", "x" * 1900, 1), Element("text", "y" * 1900, 2)]
    chunks = chunk_elements(elements, max_chars=2000, overlap_chars=200)
    assert chunks[-1].content.endswith("y")
    for previous, current in zip(chunks, chunks[1:]):
        assert not previous.content.endswith(current.content)


def test_build_chunk_rows_ids_and_embed_text():
    elements = elements_from_parsed(PARSED)
    doc_id = doc_id_for("/Volumes/c/s/v/nb/report.pdf")
    rows = build_chunk_rows(doc_id, "nb", "/Volumes/c/s/v/nb/report.pdf", "report.pdf", elements)
    assert rows[0]["chunk_id"] == f"{doc_id}-00000"
    assert rows[0]["embed_text"].startswith("Documento: report.pdf\nSezione: Relazione annuale\n")
    assert rows[0]["notebook"] == "nb"


def test_render_document_text_has_page_markers():
    text = render_document_text(elements_from_parsed(PARSED))
    assert text.startswith("[Pagina 1]\n\n## Relazione annuale")
    assert "[Pagina 2]" in text
