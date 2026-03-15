# Plan: Publish `mlxmolkit` to PyPI

## Current State

- Modern `pyproject.toml` with project metadata, dependencies, and URLs
- Flat layout: `mlxmolkit/` package at repo root
- MIT license present
- Clean `__init__.py` with `__all__` and `__version__`
- `.gitignore` covers build artifacts, dist/, *.egg-info
- **Missing:** build backend, classifiers, README content, CI/CD, wheel exclusions

---

## Step 1: Add Build Backend to `pyproject.toml`

Add `[build-system]` using **hatchling** (lightweight, modern, PEP 517 compliant):

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

---

## Step 2: Add Trove Classifiers

Add classifiers to `pyproject.toml` for PyPI discoverability:

```toml
classifiers = [
    "Development Status :: 3 - Alpha",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Chemistry",
    "Operating System :: MacOS :: MacOS X",
    "Intended Audience :: Science/Research",
]
```

---

## Step 3: Single-Source Version (Option B)

Currently version is defined in **two places** (`pyproject.toml` and `mlxmolkit/__init__.py`).
Use hatchling's dynamic versioning to read from `__init__.py` as single source of truth.

**Changes:**

1. Remove `version = "0.1.0"` from `pyproject.toml`
2. Add `dynamic = ["version"]` to `[project]`
3. Add version source config:

```toml
[tool.hatch.version]
path = "mlxmolkit/__init__.py"
```

4. Keep `__version__ = "0.1.0"` in `mlxmolkit/__init__.py` — this becomes the single source.

---

## Step 4: Write README.md

Populate `README.md` (currently empty). This renders as the PyPI landing page.

Include:
- One-liner description
- What it does (GPU-accelerated molecular 3D coordinate generation on Apple Silicon)
- Installation: `pip install mlxmolkit`
- Quick usage example (using `EmbedMolecules`)
- Requirements (macOS, Apple Silicon, Python >=3.12)
- Dependencies note (mlx, rdkit, numpy)
- License
- Link to repo

---

## Step 5: Exclude Non-Package Files from Wheel

Ensure tests, benchmarks, and dev files don't ship in the distributed package:

```toml
[tool.hatch.build.targets.wheel]
packages = ["mlxmolkit"]
```

---

## Step 6: Build & Validate Locally

```bash
uv pip install build twine
python -m build            # produces dist/*.whl and dist/*.tar.gz
twine check dist/*         # validates metadata and README rendering
pip install dist/*.whl     # test install in a clean venv
```

---

## Step 7: PyPI Account Setup

1. Register at https://pypi.org/account/register/
2. Enable 2FA (required for new projects)
3. Create an API token (Account Settings → API tokens)
4. Scope token to the project after first upload

---

## Step 8: Test Upload to TestPyPI

```bash
twine upload --repository testpypi dist/*
```

Verify at https://test.pypi.org/project/mlxmolkit/

---

## Step 9: Production Upload

```bash
twine upload dist/*
```

Verify at https://pypi.org/project/mlxmolkit/

---

## Step 10: GitHub Actions CI/CD

GitHub Actions is **free for public repos** (unlimited minutes).
Private repos get 2,000 min/month free on the Free tier.

### Workflow A: CI (`ci.yml`)
- Trigger: push to main, PRs
- Steps: checkout → setup Python 3.12 → install deps → run `pytest`

### Workflow B: Publish (`publish.yml`)
- Trigger: GitHub Release creation (tag `v*`)
- Steps: checkout → build → upload to PyPI
- Use **Trusted Publishers** (OIDC) — no API token secrets needed
  1. Configure PyPI project to trust the GitHub repo
  2. GitHub Actions uses OIDC authentication
  3. No secrets to manage or rotate

---

## Step 11: Versioning Strategy

Follow **SemVer** (MAJOR.MINOR.PATCH). Release flow:

1. Bump `__version__` in `mlxmolkit/__init__.py`
2. Commit: `git commit -m "Bump version to 0.2.0"`
3. Tag: `git tag v0.2.0`
4. Push: `git push && git push --tags`
5. Create GitHub Release → triggers publish workflow

---

## Checklist

- [ ] Add `[build-system]` to `pyproject.toml`
- [ ] Add classifiers to `pyproject.toml`
- [ ] Single-source version via hatchling dynamic versioning
- [ ] Write README.md content
- [ ] Add wheel exclusion config
- [ ] Local build + `twine check`
- [ ] Create PyPI account + enable 2FA
- [ ] Test upload to test.pypi.org
- [ ] Production upload to pypi.org
- [ ] Add GitHub Actions CI workflow
- [ ] Add GitHub Actions publish workflow
