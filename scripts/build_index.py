import json, csv, pandas as pd
from tqdm import tqdm
from pathlib import Path
from pyserini.index.lucene import LuceneIndexer

CSV_PATH = Path("data/raw/papersum_cleaned.csv")
JSONL_DIR = Path("data/processed")
INDEX_DIR = Path("index/bm25")

JSONL_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)

def sniff_csv(input_path):
    with open(input_path, 'r', encoding='utf-8', errors='replace') as infile:
        sample = infile.read(1024)  # Read a chunk to analyze
        dialect = csv.Sniffer().sniff(sample)
        print("Detected Dialect:")
        print(f"Delimiter: {repr(dialect.delimiter)}")
        print(f"Quotechar: {repr(dialect.quotechar)}")
        print(f"Escapechar: {repr(dialect.escapechar)}")


def csv_to_jsonl():
    print("Converting CSV to JSONL files...")

    total_rows = sum(1 for _ in open(CSV_PATH)) - 1  # subtract header
    try:
        for chunk in tqdm(pd.read_csv(CSV_PATH, chunksize=50_000, quotechar='"', escapechar='\\', engine='python'), total=(total_rows // 50_000) + 1, desc="Processing CSV Chunks"):
            try:
                for _, row in chunk.iterrows():
                    doc = {
                        "id": row.paperID,
                        "title": row.title,
                        "abstract": row.abstract,
                        "venue": row.venue,
                        "year":  row.year,
                        "text":  f"{row.title}. {row.abstract} {row.conclusion or ''}"
                    }
                    out = JSONL_DIR / f"{doc['id']}.json"
                    with open(out, 'w', encoding='utf-8') as f:
                        f.write(json.dumps(doc))
            except Exception as e:
                print(f"error processing row: {e}")
    except Exception as e:
        print(f"error during csv to jsonl conversion: {e}")

    print(f"CSV to JSONL conversion completed. Files are stored in {JSONL_DIR}")

def build_bm25():
    print("Building BMN25 Index with Lucene Indexer...")
    writer = LuceneIndexer(str(INDEX_DIR))

    json_files = list(JSONL_DIR.glob("*.json"))

    if not json_files:
        print(f"no json files found in {JSONL_DIR}. exiting")
        return

    for path in tqdm(json_files, desc="Indexing Documents"):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            writer.add_doc_raw(content)
        except Exception as e:
            print(f"error indexing {path.name}: {e}")

    writer.commit()
    print(f"Indexing completed. Index is stored in {INDEX_DIR}")

if __name__ == "__main__":
    
    csv_to_jsonl()
    # build_bm25()
            