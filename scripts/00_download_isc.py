import html
import re
import sys
import time

import requests

BASE_URL = "https://www.isc.ac.uk/cgi-bin/web-db-run"

BASE_PARAMS = {
    "request": "COMPREHENSIVE",
    "out_format": "ISF2",
    "searchshape": "RECT",
    "bot_lat": "26.5",
    "top_lat": "30",
    "left_lon": "50",
    "right_lon": "53",
    "ctr_lat": "",
    "ctr_lon": "",
    "radius": "",
    "max_dist_units": "deg",
    "srn": "",
    "grn": "",
    "start_month": "1",
    "start_day": "01",
    "start_time": "00:00:00",
    "end_month": "1",
    "end_day": "01",
    "end_time": "00:00:00",
    "min_dep": "",
    "max_dep": "",
    "min_mag": "",
    "max_mag": "",
    "req_mag_type": "",
    "req_mag_agcy": "",
    "min_def": "",
    "max_def": "",
    "prime_only": "on",
    "include_phases": "on",
    "include_magnitudes": "on",
    "include_headers": "on",
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) ISC-data-downloader",
}

PRE_RE = re.compile(r"<pre[^>]*>(.*?)</pre>", flags=re.DOTALL | re.IGNORECASE)

REQUEST_DELAY = 3


def fetch_year(start_year: int, end_year: int) -> str:
    """Download one year range. Returns 'ok', 'exists', 'empty' or 'error'."""
    filename = f"{start_year}-{end_year}.txt"

    if is_downloaded(start_year, end_year):
        print(f"[SKIP ] {filename} already downloaded", flush=True)
        return "exists"

    params = dict(BASE_PARAMS)
    params["start_year"] = str(start_year)
    params["end_year"] = str(end_year)

    print(f"[ GET ] {filename} downloading ...", flush=True)

    try:
        resp = requests.get(BASE_URL, params=params, headers=HEADERS, timeout=600)
        resp.raise_for_status()
    except Exception as exc:
        print(f"[ERROR] {filename} request failed: {exc}", flush=True)
        return "error"

    blocks = PRE_RE.findall(resp.text)
    if not blocks:
        print(f"[ERROR] {filename} no <pre> tag found in response", flush=True)
        return "empty"

    content = "\n".join(html.unescape(b) for b in blocks)
    content = content.replace("\r\n", "\n")

    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"[ OK ] {filename} saved ({len(content):,} chars)", flush=True)
    return "ok"


def is_downloaded(start_year: int, end_year: int) -> bool:
    """A file counts as downloaded only if it exists, is non-empty and has data lines."""
    import os

    filename = f"{start_year}-{end_year}.txt"
    if not os.path.isfile(filename):
        return False
    if os.path.getsize(filename) == 0:
        return False
    with open(filename, "r", encoding="utf-8", errors="ignore") as f:
        return any(line.startswith(("Event", "DATA_TYPE")) for line in f)


def main() -> None:
    args = [a for a in sys.argv[1:]]
    start = int(args[0]) if len(args) > 0 else 2010
    end = int(args[1]) if len(args) > 1 else 2026

    print(f"Range: {start} to {end} ({end - start} file(s))\n", flush=True)

    downloaded, skipped, failed = [], [], []

    for year in range(start, end):
        status = fetch_year(year, year + 1)
        if status == "ok":
            downloaded.append(year)
        elif status in ("exists",):
            skipped.append(year)
        else:
            failed.append(year)
        if year + 1 < end:
            time.sleep(REQUEST_DELAY)

    print("\n===== SUMMARY =====")
    print(f"Downloaded now : {len(downloaded)}" + (f"  -> {[f'{y}-{y+1}' for y in downloaded]}" if downloaded else ""))
    print(f"Already had    : {len(skipped)}" + (f"  -> {[f'{y}-{y+1}' for y in skipped]}" if skipped else ""))
    print(f"Failed         : {len(failed)}" + (f"  -> {[f'{y}-{y+1}' for y in failed]}" if failed else ""))

    if failed:
        print("\nRe-run the same command to retry only the failed ones.")


main()
