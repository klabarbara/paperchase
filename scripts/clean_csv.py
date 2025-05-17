import csv, pathlib, shutil, tempfile

src = pathlib.Path("data/raw/papersum.csv")
tmp = tempfile.NamedTemporaryFile(delete=False, mode="w", encoding="utf‑8", newline="")

with src.open(newline="", encoding="utf‑8") as fin, tmp as fout:
    rdr = csv.reader(fin, delimiter=",", quotechar='"',
                     escapechar="\\", strict=False)   
    wtr = csv.writer(fout, delimiter=",", quotechar='"',
                     escapechar="\\", quoting=csv.QUOTE_MINIMAL)
    for row in rdr:
        wtr.writerow(row)

shutil.move(tmp.name, "data/raw/papersum_clean.csv")