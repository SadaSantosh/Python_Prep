"""Smoke tests: boot each Streamlit app headlessly and assert it renders cleanly.

These guard against import errors, missing artifacts, and UI exceptions being
shipped to the deployed Streamlit apps.
"""

import os

import pytest
from streamlit.testing.v1 import AppTest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

APP_PATHS = {
    "telco_churn": os.path.join(PROJECT_ROOT, "Project_03_Telco_Customer_Churn", "app.py"),
    "real_estate": os.path.join(PROJECT_ROOT, "project_04_real_estate_ai", "app.py"),
    "phishshield": os.path.join(PROJECT_ROOT, "project_05_spam_phishing_detector", "app.py"),
}


@pytest.mark.parametrize("name,app_path", APP_PATHS.items())
def test_app_boots_without_errors(name, app_path):
    at = AppTest.from_file(app_path, default_timeout=90)
    at.run()

    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    assert at.title, f"{name} app should render a page title"

    # The neumorphic theme must be injected as a <style> block
    css_markdown = [md.value or "" for md in at.markdown]
    assert any("--neu-bg" in block for block in css_markdown), (
        f"{name} app is missing the neumorphic CSS theme"
    )
