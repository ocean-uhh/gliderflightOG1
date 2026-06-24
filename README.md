# gliderflightOG1

A package for running multiple glider flight models on different glider platforms. Using OG1 as the common input format makes this possible: once your data is in OG1, any supported flight model can be applied regardless of the original glider type.

📘 Documentation is available at:
👉 https://ocean-uhh.github.io/gliderflightOG1/

Originating from conversations associated with the "vertical velocities" group at [OceanGlidersCommunity](https://github.com/OceanGlidersCommunity/Vertical_Velocities_SOP). This package is now hosted and maintained under the University of Hamburg Oceanography GitHub organization: https://github.com/ocean-uhh/gliderflightOG1.

---

## 🚀 What's Included

- ✅ Python package layout: `gliderflightOG1/*.py`
- 📓 Jupyter notebook demo: `notebooks/demo.ipynb`
- 📄 Markdown and Sphinx-based documentation in `docs/`
- 🔍 Tests with `pytest` in `tests/`, CI with GitHub Actions
- 🎨 Code style via `black`, `ruff`, `pre-commit`
- 📦 Package config via `pyproject.toml` + optional PyPI release workflow
- 🧾 Machine-readable citation: `CITATION.cff`

---

## 🔧 Quickstart

Install in development mode:

```bash
git clone https://github.com/ocean-uhh/gliderflightOG1.git
cd gliderflightOG1
python -m venv venv       # if you manage environments with venv
source venv/bin/activate  # if you manage environments with venv
pip install -r requirements-dev.txt
pip install -e .
```

To run tests:

```bash
pytest
```

To build the documentation locally:

```bash
cd docs
make html
```

---

## 🤝 Contributing

Contributions are welcome!  Please also consider adding an [issue](https://github.com/ocean-uhh/gliderflightOG1/issues) when something isn't clear.

---

## Acknowledgements

Development of this package is supported by [Voice of the Ocean (VOTO)](https://voiceoftheocean.org/) through the SeaExplorer–Seaglider Cross-platform Open Diagnostics & Evaluation (SEA-CODE) project (2026).

---

## Current state and future plans

A first version of the Seaglider flight model has been implemented (translated from MATLAB), including steady-state and unsteady flight solvers and hydrodynamic parameter optimization.

Planned additions include flight models based on FW2011, the [Seaglider basestation](https://github.com/iop-apl-uw/basestation3/blob/master/FlightModel.py), and Lucas Merckelbach's [gliderflight](https://gliderflight.readthedocs.io/en/latest/using_gliderflight.html) for Slocum data.

