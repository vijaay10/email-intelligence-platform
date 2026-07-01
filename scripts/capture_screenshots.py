"""
Real UI Screenshot Capture
==========================
Drives the running Streamlit app with headless Chromium (Playwright) and saves
real screenshots of the dashboard pages to docs/post/. Unlike the generated
reel, these are authentic captures of the live app + local llama3 model.

Prereqs (already handled by the setup here):
    pip install playwright && playwright install chromium

Run (the app must be running, e.g. via this script's launcher):
    python scripts/capture_screenshots.py
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from playwright.sync_api import sync_playwright

BASE = os.environ.get("APP_URL", "http://localhost:8502")
OUT = os.path.join(ROOT, "docs", "post")
os.makedirs(OUT, exist_ok=True)


def _settle(page, ms=4500):
    try:
        page.wait_for_load_state("networkidle", timeout=15000)
    except Exception:
        pass
    page.wait_for_timeout(ms)


def _shot(page, name):
    path = os.path.join(OUT, name)
    page.screenshot(path=path, full_page=True)
    print(f"  saved {path}")


def _nav(page, link_name):
    """Click a sidebar page link by its visible name."""
    try:
        page.get_by_role("link", name=link_name, exact=False).first.click(timeout=8000)
        return True
    except Exception as exc:
        print(f"  [!] could not navigate to '{link_name}': {exc}")
        return False


def main() -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1000},
                                device_scale_factor=2)  # retina-crisp

        print(f"Opening {BASE} …")
        page.goto(BASE, timeout=30000)
        _settle(page, 6000)
        _shot(page, "app_1_home.png")

        # LLM Settings — the "model" screenshot.
        if _nav(page, "LLM Settings"):
            _settle(page)
            _shot(page, "app_2_llm_settings.png")

        # Semantic Search — type a real query and capture the ranked results.
        if _nav(page, "Semantic Search"):
            _settle(page)
            try:
                box = page.get_by_role("textbox").first
                box.click()
                box.fill("which invoices are overdue?")
                box.press("Enter")
                _settle(page, 5000)
            except Exception as exc:
                print(f"  [!] search interaction failed: {exc}")
            _shot(page, "app_3_semantic_search.png")

        # AI Summary — click Analyze and wait for the LLM to fill the page.
        if _nav(page, "AI Summary"):
            _settle(page)
            try:
                page.get_by_role("button", name="Analyze with AI").click(timeout=8000)
                # LLM runs several features; give it time, then settle.
                page.wait_for_timeout(35000)
                _settle(page, 3000)
            except Exception as exc:
                print(f"  [!] AI Summary interaction failed: {exc}")
            _shot(page, "app_4_ai_summary.png")

        browser.close()
    print("Done.")


if __name__ == "__main__":
    main()
