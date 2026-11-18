"""Desenha as Figuras Ricas (spec JSON -> rough.js SVG -> PNG) com o Chrome via Playwright.

Uso: python desenhar.py spec1.json [spec2.json ...]   (grava <spec>.png ao lado)
"""
import json
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).parent
TEMPLATE = (HERE / "render.html").read_text(encoding="utf-8")

with sync_playwright() as p:
    browser = p.chromium.launch(channel="chrome")
    for spec_path in (Path(a).resolve() for a in sys.argv[1:]):
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        html = TEMPLATE.replace("/*SPEC*/null", json.dumps(spec, ensure_ascii=False))
        page_file = spec_path.with_suffix(".html")
        page_file.write_text(html, encoding="utf-8")
        page = browser.new_page(viewport={"width": spec["w"], "height": spec["h"]}, device_scale_factor=1.5)
        page.goto(page_file.as_uri())
        page.wait_for_selector("body[data-done='1']", timeout=60000)
        out = spec_path.with_suffix(".png")
        page.locator("#fig").screenshot(path=str(out))
        print(out.name)
        page.close()
    browser.close()
