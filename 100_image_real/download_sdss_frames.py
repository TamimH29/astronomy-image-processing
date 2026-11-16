#!/usr/bin/env python3
"""Download N random SDSS DR17 imaging frames (FITS) into a subfolder next to this script.

Uses only the Python standard library (Python 3.8+).
Run:  python download_sdss_frames.py
"""
import csv
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

# ---- Settings you may want to change ------------------------------------
N_IMAGES = 100          # how many frames to end up with
BAND = "r"              # one of u, g, r, i, z
SEED = None             # set to an int (e.g. 42) for a reproducible sample
SUBFOLDER = "sdss_frames"
# --------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent          # folder the script lives in
OUT_DIR = HERE / SUBFOLDER
FIELDS_CSV = HERE / "fields.csv"                # cached candidate list

SQLS = [
    # 1) ~1% pseudo-random sample of fields
    "SELECT run, rerun, camcol, field FROM Field "
    "WHERE rerun = 301 AND (run * field) % 97 = 0",
    # 2) simpler fallback: every field in a spread of runs (no modulo arithmetic)
    "SELECT TOP 5000 run, rerun, camcol, field FROM Field "
    "WHERE rerun = 301 AND field % 10 = 0",
    # 3) last resort: first rows (less random, but shuffled locally by the script)
    "SELECT TOP 5000 run, rerun, camcol, field FROM Field WHERE rerun = 301",
]
SQL_URLS = [
    "https://skyserver.sdss.org/dr17/SkyServerWS/SearchTools/SqlSearch",
    "https://skyserver.sdss.org/dr16/SkyServerWS/SearchTools/SqlSearch",
]
FRAME_BASE = "https://data.sdss.org/sas/dr17/eboss/photoObj/frames"
HEADERS = {"User-Agent": "Mozilla/5.0 (sdss-frame-downloader; python-urllib)"}


def urlopen(url, timeout=120):
    return urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS),
                                  timeout=timeout)


def fetch_fields_csv():
    """Try each endpoint/query until SkyServer returns a CSV table."""
    for base in SQL_URLS:
        for sql in SQLS:
            url = base + "?" + urllib.parse.urlencode({"cmd": sql, "format": "csv"})
            try:
                print(f"Trying: {base.split('/')[3]}  |  {sql[:60]}...")
                with urlopen(url) as resp:
                    data = resp.read()
                text = data.decode("utf-8", "replace")
                if "run" in text.lower().splitlines()[1 if text.startswith("#") else 0]:
                    return data
                print("  Response was not a CSV table:", text[:200].strip())
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", "replace")[:300].strip()
                print(f"  HTTP {e.code}: {body}")
            except Exception as e:
                print(f"  Failed: {e}")
    return None


def load_fields():
    """Return a list of (run, rerun, camcol, field) tuples, querying SkyServer if needed."""
    if not FIELDS_CSV.exists():
        print("Querying SkyServer for candidate fields...")
        data = fetch_fields_csv()
        if data is None:
            raise SystemExit(
                "\nCould not get fields from SkyServer automatically.\n"
                "Manual fix: open https://skyserver.sdss.org/public/SearchTools/SQL/ , "
                "set the format to CSV, run:\n"
                f"  {SQLS[0]}\n"
                "then save the result as 'fields.csv' next to this script and run again."
            )
        FIELDS_CSV.write_bytes(data)

    lines = [ln for ln in FIELDS_CSV.read_text().splitlines()
             if ln.strip() and not ln.startswith("#")]
    reader = csv.DictReader(lines)
    if not reader.fieldnames or "run" not in [f.strip().lower() for f in reader.fieldnames]:
        FIELDS_CSV.unlink()
        raise SystemExit(
            "SkyServer did not return a CSV table. Run the SQL query on the SkyServer "
            "SQL Search page (CSV format), save it as fields.csv next to this script, "
            "and run the script again."
        )
    rows = []
    for r in reader:
        r = {k.strip().lower(): v for k, v in r.items()}
        rows.append((int(r["run"]), int(r["rerun"]), int(r["camcol"]), int(r["field"])))
    return rows


def frame_name(run, camcol, field):
    return f"frame-{BAND}-{run:06d}-{camcol}-{field:04d}.fits.bz2"


def download(url, dest):
    """Download url to dest. Returns True on success, False on 404 or repeated failure."""
    tmp = dest.with_name(dest.name + ".part")
    for _ in range(3):
        try:
            with urlopen(url) as resp, open(tmp, "wb") as f:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            tmp.replace(dest)
            return True
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return False
            time.sleep(2)
        except Exception:
            time.sleep(2)
    if tmp.exists():
        tmp.unlink()
    return False


def main():
    OUT_DIR.mkdir(exist_ok=True)
    pool = load_fields()
    random.Random(SEED).shuffle(pool)

    have = len(list(OUT_DIR.glob("*.fits.bz2")))
    print(f"{len(pool)} candidate fields; {have} frames already in {OUT_DIR}")

    for run, rerun, camcol, field in pool:
        if have >= N_IMAGES:
            break
        name = frame_name(run, camcol, field)
        dest = OUT_DIR / name
        if dest.exists():
            continue
        url = f"{FRAME_BASE}/{rerun}/{run}/{camcol}/{name}"
        if download(url, dest):
            have += 1
            print(f"[{have}/{N_IMAGES}] {name}")
        else:
            print(f"  skipped {name} (not available), trying another field")
        time.sleep(1)  # be polite to the server

    print(f"Done: {have} frames in {OUT_DIR}")


if __name__ == "__main__":
    main()
