import sys
sys.path.insert(0, ".")

from src.data.chunking import chunk_filing, split_into_sections


def test_chunk_filing_indices_are_globally_unique_across_sections():
    body = "word " * 2000
    text = (
        f"Item 1A. Risk Factors\n{body}\n"
        f"Item 7. Management Discussion and Analysis\n{body}\n"
        f"Item 8. Financial Statements\n{body}\n"
    )
    chunks = chunk_filing(text, chunk_size_tokens=200, overlap_tokens=40)

    indices = [c.chunk_index for c in chunks]
    assert len(indices) == len(set(indices))
    assert indices == list(range(len(chunks)))

    sections_present = {c.section for c in chunks}
    assert len(sections_present) == 3
    
def test_split_into_sections_detects_item_headings():
    text = "Item 1A. Risk Factors\nSome risk text.\nItem 7. Management Discussion and Analysis\nSome MD&A text."
    sections = split_into_sections(text)
    assert len(sections) == 2
    assert sections[0][0] == "Item 1A. Risk Factors"
    assert sections[1][0] == "Item 7. Management Discussion and Analysis"


def test_split_into_sections_falls_back_without_headings():
    text = "Just some plain text with no item headings at all."
    sections = split_into_sections(text)
    assert len(sections) == 1
    assert sections[0][0] == "Full Document"


def test_chunk_filing_respects_overlap():
    body = "word " * 2000
    text = f"Item 1A. Risk Factors\n{body}"
    chunks = chunk_filing(text, chunk_size_tokens=200, overlap_tokens=40)
    assert len(chunks) > 1
    # every chunk should be tagged with its source section
    assert all(c.section == "Item 1A. Risk Factors" for c in chunks)
    # chunk indices should be sequential starting at 0
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_chunk_filing_empty_section_skipped():
    text = "Item 1A. Risk Factors\n\nItem 7. Management Discussion and Analysis\nActual content here."
    chunks = chunk_filing(text)
    sections = {c.section for c in chunks}
    assert "Item 7. Management Discussion and Analysis" in sections
