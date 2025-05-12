import os
import pytest

# Import functions from ingest.py
from ingest import extract_text_from_pdf, preprocess_text, chunk_text, clean_game_name

data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
sample_pdf = os.path.join(data_dir, 'Ticket_to_Ride_Manual.pdf')

@pytest.mark.skipif(not os.path.exists(sample_pdf), reason="Sample PDF not found.")
def test_extract_text_from_pdf():
    text = extract_text_from_pdf(sample_pdf)
    print("Extracted text (first 500 chars):\n", text[:500])
    assert isinstance(text, str)
    assert len(text) > 100  # Should extract some text


def test_preprocess_text():
    raw = "This   is   a   test.\n\nWith   multiple   spaces.\nPage 1 of 2\n"
    cleaned = preprocess_text(raw)
    print("Preprocessed text:", cleaned)
    assert '  ' not in cleaned
    assert cleaned.startswith("This is a test.")


def test_chunk_text():
    text = "A" * 2500
    chunks = chunk_text(text, chunk_size=1000, chunk_overlap=200)
    print("Chunk lengths:", [len(c) for c in chunks])
    assert len(chunks) > 1
    assert all(len(c) <= 1000 for c in chunks)


@pytest.mark.skipif(not os.path.exists(sample_pdf), reason="Sample PDF not found.")
def test_full_pdf_processing():
    print(f"\n---\nTesting full PDF processing for: {sample_pdf}\n---")
    raw = extract_text_from_pdf(sample_pdf)
    print("Raw text sample (first 300 chars):", raw[:300])
    assert len(raw) > 100
    processed = preprocess_text(raw)
    print("Processed text sample (first 300 chars):", processed[:300])
    assert len(processed) > 50
    chunks = chunk_text(processed)
    print(f"Number of chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks[:2]):
        print(f"Chunk {i+1} (first 200 chars):", chunk[:200])
    assert len(chunks) > 0

    # Output the fully read text for manual verification
    outputs_dir = os.path.join(os.path.dirname(__file__), 'pdf_text_outputs')
    os.makedirs(outputs_dir, exist_ok=True)
    game_name = clean_game_name(os.path.basename(sample_pdf))
    output_path = os.path.join(outputs_dir, f"{game_name}.txt")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(processed)
    print(f"Full PDF text written to: {output_path}")


@pytest.mark.skipif(not os.environ.get('RUN_ALL_PDF_TESTS'), reason="Set RUN_ALL_PDF_TESTS=1 to enable this test.")
def test_full_pdf_processing_all():
    """Process all PDFs in the data directory, extract and preprocess text, and write to pdf_text_outputs."""
    pdf_files = [f for f in os.listdir(data_dir) if f.lower().endswith('.pdf')]
    assert pdf_files, "No PDF files found in data directory."
    outputs_dir = os.path.join(os.path.dirname(__file__), 'pdf_text_outputs')
    os.makedirs(outputs_dir, exist_ok=True)
    for pdf_file in pdf_files:
        pdf_path = os.path.join(data_dir, pdf_file)
        print(f"\n---\nProcessing: {pdf_file}\n---")
        try:
            raw = extract_text_from_pdf(pdf_path)
            processed = preprocess_text(raw)
            game_name = clean_game_name(os.path.basename(pdf_file))
            output_path = os.path.join(outputs_dir, f"{game_name}.txt")
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(processed)
            print(f"Processed text written to: {output_path}")
            assert len(processed) > 50
        except Exception as e:
            print(f"Error processing {pdf_file}: {e}")
            assert False, f"Failed to process {pdf_file}: {e}"
