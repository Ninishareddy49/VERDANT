"""Capture application screenshots for the paper (needs backend + frontend running).

    python paper/screenshots.py http://localhost:5199
"""
from __future__ import annotations

import os
import sys

from playwright.sync_api import sync_playwright

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:5173"
HERE = os.path.dirname(os.path.abspath(__file__))
SHOTS = os.path.join(HERE, "figures", "screens")
LEAF = os.path.join(os.path.dirname(HERE), "frontend", "public", "assets", "feature", "diseased-leaf.png")
os.makedirs(SHOTS, exist_ok=True)


def nav(page, label):
    page.get_by_role("button", name=label).first.click()
    page.wait_for_timeout(1200)


with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=2)
    page.goto(BASE, wait_until="networkidle")
    page.wait_for_timeout(1500)
    page.get_by_text("Launch Platform").first.click()
    page.wait_for_timeout(2000)
    page.screenshot(path=os.path.join(SHOTS, "overview.png"))

    nav(page, "Plant Doctor")
    page.locator("input[type=file]").first.set_input_files(LEAF)
    page.wait_for_timeout(1000)
    for label in ("Diagnose", "Analyze", "Run diagnosis", "Analyse"):
        btn = page.get_by_role("button", name=label)
        if btn.count():
            btn.first.click()
            break
    page.wait_for_timeout(15000)
    page.screenshot(path=os.path.join(SHOTS, "doctor.png"))

    nav(page, "AI Advisor")
    box = page.get_by_placeholder("Ask about crops, nutrients, climate, pests…")
    box.fill("My lettuce roots look brown and slimy in the DWC tank. What should I do?")
    box.press("Enter")
    page.wait_for_timeout(25000)
    page.screenshot(path=os.path.join(SHOTS, "advisor.png"))

    nav(page, "Grow Simulator")
    page.wait_for_timeout(1500)
    page.screenshot(path=os.path.join(SHOTS, "simulator.png"))

    nav(page, "VPD Matrix")
    page.wait_for_timeout(1500)
    page.screenshot(path=os.path.join(SHOTS, "vpd.png"))
    browser.close()
print("saved to", SHOTS)
