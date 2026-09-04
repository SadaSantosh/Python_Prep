# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-09-04

### Added
- **Neumorphism design system** for all three Streamlit apps (Telco Churn, ValuaAI, PhishShield): soft gray `#e0e0e0` canvas, 15px border radius, extruded dual box-shadows, inset inputs, and charcoal text for high readability.
- **Headless Streamlit smoke tests** (`tests/test_app_smoke.py`) that boot each app with `streamlit.testing.v1.AppTest` and assert zero exceptions/errors and a rendered theme.
- **`streamlit-smoke` CI job** that runs the app smoke tests on every push.
- Per-project `README.md` files for the Student Performance Classifier and Corporate Employee Analytics projects.
- Real deployment links for ValuaAI and PhishShield in the root portfolio README.

### Changed
- **Bug fixes & stabilization:**
  - ValuaAI market map now uses `px.scatter_map` (MapLibre) with a legacy fallback — `px.scatter_mapbox` was removed in plotly 7 and crashed the app on fresh installs.
  - Telco What-If Explorer now honors its tenure, monthly charge, internet, and payment inputs (previously ignored in favor of sidebar defaults).
  - Retention Simulator contract dropdown no longer offers duplicate options; simulated-score delta renders correctly with `delta_color="inverse"`.
  - All Streamlit apps migrated from the deprecated `use_container_width` kwarg to `width="stretch"`.
  - PhishShield training artifacts are now written relative to the script (`BASE_DIR`), safe to run from any working directory.
- **Codebase linting is now clean under strict flake8** (`--max-line-length=120`, no ignores); the CI lint job was tightened to enforce it.
- **Projects 01 & 02 polished to repository standard:** module docstrings, `BASE_DIR`-anchored paths, `main()` guards, clean comments, and consistent output. Fixed a latent crash in `ml_viz.py` where the 2D decision-boundary grid was predicted against a 3-feature model.
- Root `README.md` overhauled into a FAANG-caliber portfolio profile (positioning header, project grid, engineering focus).
- `CONTRIBUTING.md` updated to document the Neumorphism design system and the expanded test suite.

### Removed
- Orphaned `dashboard.png` asset (unreferenced and predating the current UI).
- Bytecode and cache artifacts (`__pycache__`, `.pyc`, `.pytest_cache`) from the repository.

---

## Previous development (untagged)

- **2026-08-31** — Minimalist UI redesign of the Streamlit apps, Project 04 runtime fix, and removal of AI-generated traces from source files.
- **2026-08** — Repository restructure into five standardized project folders, initial CI/CD pipeline, per-project requirements, and the root portfolio README.
- **Earlier** — Unit test suite (model artifact loading, prediction pipelines), contribution guide, and the first versions of all five projects.

[1.0.0]: https://github.com/SadaSantosh/Python_Prep/releases/tag/v1.0.0
