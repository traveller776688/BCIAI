"""Rebuild DAME paper PDFs (EN/CN): markdown -> pandoc HTML (MathML) -> Edge headless PDF.

Usage: python paper/build_pdf.py            # rebuild both EN and CN
       python paper/build_pdf.py EN         # rebuild one language
"""
from __future__ import annotations

import pathlib
import re
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
PAPER = ROOT / "paper"


def find_pandoc() -> str:
    base = pathlib.Path.home() / "AppData/Local/Microsoft/WinGet/Packages"
    hits = sorted(base.glob("JohnMacFarlane.Pandoc_*/pandoc-*/pandoc.exe"))
    if hits:
        return str(hits[-1])
    raise FileNotFoundError("pandoc.exe not found under WinGet Packages")


def find_edge() -> str:
    for p in (r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
              r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"):
        if pathlib.Path(p).exists():
            return p
    raise FileNotFoundError("msedge.exe not found")


def build(lang: str, pandoc: str, edge: str) -> None:
    md = PAPER / f"DAME_Paper_{lang}.md"
    html = PAPER / f"DAME_Paper_{lang}.html"
    pdf = PAPER / f"DAME_EEG_Paper_{lang}.pdf"
    css = PAPER / "style.css"

    # 1. markdown -> standalone HTML with native MathML math
    subprocess.run([pandoc, str(md), "-o", str(html), "--standalone",
                    "--mathml", f"--css={css}"], check=True)

    text = html.read_text(encoding="utf-8")
    # 2a. drop pandoc's duplicated title-block header (the H1 in the body is the real title)
    text = re.sub(r'<header id="title-block-header">.*?</header>\s*', "", text,
                  flags=re.S)
    # 2b. absolute file:/// URLs for stylesheet and images so Edge can load them
    text = text.replace(f'href="{css.as_posix()}"', f'href="file:///{css.as_posix()}"')
    for rel, absdir in (("../figures/", ROOT / "figures"),
                        ("../results/", ROOT / "results")):
        text = text.replace(f'src="{rel}', f'src="file:///{absdir.as_posix()}/')
    html.write_text(text, encoding="utf-8")

    # 3. Edge headless print-to-PDF (isolated profile so a running Edge never clashes)
    with tempfile.TemporaryDirectory() as profile:
        subprocess.run([edge, "--headless=new", "--disable-gpu",
                        "--disable-extensions", f"--user-data-dir={profile}",
                        "--no-pdf-header-footer", f"--print-to-pdf={pdf}",
                        html.as_uri()], check=True, timeout=600)
    print(f"built {pdf.name}: {pdf.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    pandoc = find_pandoc()
    edge = find_edge()
    for lang in sys.argv[1:] or ("EN", "CN"):
        build(lang, pandoc, edge)
