import re
import sys

import requests

from camera import grab_frame

HEADERS = {"User-Agent": "Mozilla/5.0 ParkingAI/1.0"}

CANDIDATES = [
    "https://www.nps.gov/webcams-yose/aawest.jpg",
    "https://www.nps.gov/webcams-grca/hopi_point.jpg",
    "https://www.nps.gov/webcams-zion/zt_southentrance.jpg",
    "https://www.nps.gov/webcams-grte/goats_full.jpg",
    "https://bouldercounty.gov/open-space/parks-and-trails/live-trailhead-cameras/",
    "https://www.trafikoa.eus/",
    "https://www.camarasentorno.es/gipuzkoa.html",
    "https://www.gipuzkoasansebastian.eus/eu/webcam",
]


def inspect_page(url: str) -> None:
    try:
        response = requests.get(url, timeout=15, headers=HEADERS)
    except Exception as exc:
        print(f"PAGE ERR {url} -> {exc}")
        return
    ctype = response.headers.get("content-type", "")
    print(f"PAGE {response.status_code} {ctype} {url}")
    if "html" not in ctype:
        return
    found = re.findall(r"https?://[^\"']+\.(?:jpg|jpeg|png|mjpg)(?:\?[^\"']*)?", response.text, flags=re.I)
    unique = list(dict.fromkeys(found))[:12]
    for img in unique:
        print("  IMG", img)


def try_frame(url: str) -> None:
    frame = grab_frame(url)
    if frame is None:
        print(f"FRAME FAIL {url}")
        return
    print(f"FRAME OK {frame.shape} {url}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    for url in CANDIDATES:
        inspect_page(url)
        try_frame(url)
        print("---")
