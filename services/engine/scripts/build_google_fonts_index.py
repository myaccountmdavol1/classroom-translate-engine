"""Write ct_engine/fonts/google_fonts.json: every Google Fonts family, its
files and its weights, at one commit of github.com/google/fonts.

Usage: python scripts/build_google_fonts_index.py [COMMIT]

The engine looks a worksheet's own font up in it (fonts/google.py) and
fetches only the one file it needs, from jsDelivr's copy of that commit, so
the files never change under a document. Run again to take newer fonts.
"""
from __future__ import annotations

import json
import re
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "ct_engine" / "fonts" / "google_fonts.json"
CDN = "https://cdn.jsdelivr.net/gh/google/fonts@{commit}/{path}"
# The family's kind of face, as one letter: s(ans), (se)r(if), d(isplay),
# h(andwriting), m(ono).
CATEGORY = {"SANS_SERIF": "s", "SERIF": "r", "DISPLAY": "d", "HANDWRITING": "h", "MONOSPACE": "m"}


def _get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read()


def _metadata(text: str):
    name = re.search(r'^name: "([^"]+)"', text, re.M)
    category = re.search(r'^category: "([^"]+)"', text, re.M)
    fonts = [
        (int(w), s == "italic", f)
        for s, w, f in re.findall(
            r'fonts \{[^}]*?style: "(\w+)"[^}]*?weight: (\d+)[^}]*?filename: "([^"]+)"', text, re.S)
    ]
    axes = [
        (tag, float(lo), float(hi))
        for tag, lo, hi in re.findall(
            r'axes \{\s*tag: "(\w+)"\s*min_value: ([\d.]+)\s*max_value: ([\d.]+)', text, re.S)
    ]
    return (name.group(1) if name else None), (category.group(1) if category else ""), fonts, axes


def main() -> None:
    commit = sys.argv[1] if len(sys.argv) > 1 else json.loads(
        _get("https://api.github.com/repos/google/fonts/commits/main"))["sha"]
    tree = json.loads(_get(f"https://api.github.com/repos/google/fonts/git/trees/{commit}?recursive=1"))
    assert not tree.get("truncated"), "tree listing truncated"
    sizes = {e["path"]: e.get("size", 0) for e in tree["tree"]}
    paths = [p for p in sizes if p.endswith("/METADATA.pb") and p.split("/")[0] in ("ofl", "apache", "ufl")]

    def one(path: str):
        try:
            return path, _metadata(_get(CDN.format(commit=commit, path=path)).decode("utf-8"))
        except Exception as exc:  # noqa: BLE001 - reported, the rest go on
            print("skipped", path, exc, file=sys.stderr)
            return path, None

    families = {}
    with ThreadPoolExecutor(24) as pool:
        for path, meta in pool.map(one, paths):
            if not meta or not meta[0] or not meta[2]:
                continue
            name, category, fonts, axes = meta
            folder = path.rsplit("/", 1)[0]
            key = re.sub(r"[^a-z0-9]", "", name.lower())
            files = [[w, int(i), f, sizes.get(f"{folder}/{f}", 0)] for w, i, f in sorted(fonts)]
            families[key] = [folder, name, CATEGORY.get(category, ""), files, [list(a) for a in axes]]
    OUT.write_text(json.dumps({"commit": commit, "families": dict(sorted(families.items()))},
                              separators=(",", ":"), ensure_ascii=False) + "\n")
    print(f"{len(families)} families at {commit[:12]} -> {OUT} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
