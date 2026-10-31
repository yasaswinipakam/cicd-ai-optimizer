"""Capture the dashboard screenshots used in the paper from a running demo.

Start the demo first (see reports/demo_script.md), then:

    pip install playwright   # once; it drives the installed Google Chrome
    python paper/capture_screens.py [http://localhost:8010]

Uses the installed Google Chrome through Playwright and waits for every
animation to finish before each capture.
"""
from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8010"
OUT = Path(__file__).resolve().parent / "screens"
SETTLE_MS = 4500  # longest chart animation is ~1.8 s; text scrambles and count-ups finish sooner


def settle(page, locator=None) -> None:
    if locator is not None:
        locator.scroll_into_view_if_needed()
    page.wait_for_timeout(SETTLE_MS)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome")
        page = browser.new_page(viewport={"width": 1280, "height": 900}, device_scale_factor=2)

        page.goto(f"{BASE}/"); settle(page)
        page.screenshot(path=OUT / "dashboard_overview.png", clip={"x": 0, "y": 0, "width": 1280, "height": 900})
        panels = page.locator("section, div").filter(has=page.get_by_text("Predicted risk distribution")).filter(
            has=page.get_by_text("How good is the model?")).last
        settle(page, panels); panels.screenshot(path=OUT / "dashboard_models.png")

        page.goto(f"{BASE}/pipelines"); settle(page)
        page.locator("main").screenshot(path=OUT / "pipelines.png")

        page.goto(f"{BASE}/pipelines/twbs%2Fbootstrap"); settle(page)
        timeline = page.get_by_text("Risk over time").locator("xpath=ancestor::*[contains(@class,'card')][1]")
        timeline.screenshot(path=OUT / "pipeline_timeline.png")

        page.goto(f"{BASE}/submit-build"); settle(page)
        for label, value in (("Repository", "yt-dlp/yt-dlp"), ("Workflow ID", "5235551"),
                             ("PR author", "dev-13"), ("Head branch", "branch-19")):
            page.get_by_label(label).fill(value)
        page.get_by_role("button", name="Analyze build").click()
        page.get_by_text("History used").wait_for(); settle(page)
        page.locator("main").screenshot(path=OUT / "submit_result.png")

        page.goto(f"{BASE}/what-if"); settle(page)
        page.get_by_label("Previous run on this branch failed").check(); settle(page)
        page.get_by_label("Pull request run").uncheck(); settle(page)
        page.locator("main > div > div").first.screenshot(path=OUT / "what_if.png")
        browser.close()
    for path in sorted(OUT.glob("*.png")):
        print(path.name)


if __name__ == "__main__":
    main()
