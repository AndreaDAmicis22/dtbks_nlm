from nlm.passages import Passage, cited_numbers, cited_passages, format_passages, page_label


def _passage(n, **kwargs):
    defaults = dict(
        chunk_id=f"c{n}",
        doc_id="d",
        file_name="report.pdf",
        page_start=n,
        page_end=n,
        section=None,
        content=f"testo {n}",
    )
    return Passage(n=n, **(defaults | kwargs))


def test_cited_numbers_formats_and_order():
    text = "A [3]. B [1, 2]. C [3][5]. D [6-8]. E [2; 9]."
    assert cited_numbers(text) == [3, 1, 2, 5, 6, 7, 8, 9]


def test_cited_numbers_ignores_markdown_links_text():
    assert cited_numbers("vedi [la guida](https://x.y) e [2]") == [2]


def test_cited_passages_drops_unknown_numbers():
    passages = [_passage(1), _passage(2)]
    # [2025] looks like a citation but is a year: it must not break anything.
    assert [p.n for p in cited_passages("nel [2025] ... [2] [1] [7]", passages)] == [2, 1]


def test_page_label():
    assert page_label(None, None) == ""
    assert page_label(3, 3) == "p. 3"
    assert page_label(3, None) == "p. 3"
    assert page_label(3, 5) == "pp. 3-5"


def test_location_and_roundtrip():
    passage = _passage(2, page_end=4)
    assert passage.location == "report.pdf, pp. 2-4"
    assert Passage.from_dict(passage.to_dict()) == passage
    assert _passage(1, page_start=None, page_end=None).location == "report.pdf"


def test_format_passages_escapes_attributes():
    block = format_passages([_passage(1, file_name='a"b.pdf', section="Intro & scopo")])
    assert block.startswith('<passaggio id="1" documento="a&quot;b.pdf" pagine="p. 1" sezione="Intro &amp; scopo">')
    assert block.endswith("</passaggio>")
