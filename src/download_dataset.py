
from datasets import load_dataset
from pathlib import Path
import json
import pandas as pd

output_dir = Path("data/raw/scifact")
output_dir.mkdir(parents=True, exist_ok=True)

print("Downloading SciFact corpus...")
corpus = load_dataset("BeIR/scifact", "corpus", split="corpus")

print("Downloading SciFact queries...")
queries = load_dataset("BeIR/scifact", "queries", split="queries")

print("Downloading SciFact test qrels...")
qrels = load_dataset("BeIR/scifact-qrels", split="test")

with open(output_dir / "corpus.jsonl", "w", encoding="utf-8") as f:
    for item in corpus:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

with open(output_dir / "queries.jsonl", "w", encoding="utf-8") as f:
    for item in queries:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

qrels_df = pd.DataFrame(qrels)
qrels_df.to_csv(output_dir / "qrels.tsv", sep="\t", index=False)

print("\nSciFact dataset ready!")
print("Corpus documents:", len(corpus))
print("Queries:", len(queries))
print("Qrels:", len(qrels))
print("Saved location:", output_dir)
