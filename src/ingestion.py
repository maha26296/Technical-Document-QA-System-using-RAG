
import fitz
import json
from pathlib import Path

# Input and output folders
input_dir = Path("data/raw/documents")
output_dir = Path("data/processed")
output_dir.mkdir(parents=True, exist_ok=True)

all_pages = []

# Read every PDF document
for pdf_file in input_dir.glob("*.pdf"):

    print(f"Extracting: {pdf_file.name}")

    pdf = fitz.open(pdf_file)

    for page_number, page in enumerate(pdf, start=1):

        text = page.get_text("text").strip()

        if text:
            all_pages.append({
                "document_name": pdf_file.name,
                "page_number": page_number,
                "text": text
            })

    pdf.close()

# Save extracted text as JSONL
output_file = output_dir / "extracted_documents.jsonl"

with open(output_file, "w", encoding="utf-8") as file:
    for item in all_pages:
        file.write(json.dumps(item, ensure_ascii=False) + "\n")

print("\nPDF text extraction completed!")
print("Total pages extracted:", len(all_pages))
print("Output file:", output_file)
