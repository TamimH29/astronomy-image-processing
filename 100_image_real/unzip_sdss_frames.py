#!/usr/bin/env python3
"""Decompress every .fits.bz2 in the sdss_frames subfolder (next to this script)
into a plain .fits file.

Uses only the Python standard library.
Run:  python unzip_sdss_frames.py
"""
import bz2
import shutil
from pathlib import Path

# ---- Settings you may want to change ------------------------------------
SUBFOLDER = "sdss_frames"
KEEP_BZ2 = False        # False = delete each .bz2 after it unzips successfully
# --------------------------------------------------------------------------

FOLDER = Path(__file__).resolve().parent / SUBFOLDER


def main():
    files = sorted(FOLDER.glob("*.fits.bz2"))
    if not files:
        raise SystemExit(f"No .fits.bz2 files found in {FOLDER}")

    ok = failed = 0
    for i, bz2_path in enumerate(files, 1):
        fits_path = bz2_path.with_suffix("")          # drops the trailing .bz2
        tmp = fits_path.with_name(fits_path.name + ".part")
        try:
            with bz2.open(bz2_path, "rb") as src, open(tmp, "wb") as dst:
                shutil.copyfileobj(src, dst)
            tmp.replace(fits_path)
            if not KEEP_BZ2:
                bz2_path.unlink()
            ok += 1
            print(f"[{i}/{len(files)}] {fits_path.name}")
        except Exception as e:
            tmp.unlink(missing_ok=True)
            failed += 1
            print(f"[{i}/{len(files)}] FAILED {bz2_path.name}: {e} (kept the .bz2)")

    print(f"Done: {ok} unzipped, {failed} failed. Output in {FOLDER}")


if __name__ == "__main__":
    main()
