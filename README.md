# Image2Biomass Kaggle Competition

This repository contains code and resources for the [Image2Biomass Kaggle competition](https://www.kaggle.com/competitions/csiro-biomass) hosted by CSIRO. The goal of this competition is to develop models that can accurately predict biomass of grass from images.

The work is done as part of the [Deep Learning Course in KSE](https://github.com/AI301-Deep-Learning/kse_deep_learning).

## Prerequisites for Local Setup

- **Python 3.13** installed on your system. You can download it from the [official Python website](https://www.python.org/downloads/). See `.python-version` file for the exact version.
- **Git** installed on your system. You can download it from the [official Git website](https://git-scm.com/downloads).
- **uv CLI** installed and on your `PATH` (see [Astral uv docs](https://docs.astral.sh/uv/)).

### How to install uv?

For all systems should work the following command:

```bash
pip install uv
```

See the [Astral uv docs](https://docs.astral.sh/uv/) for more installation options.

## How setup the project?

1. Clone the repository:

   ```bash
   git clone https://github.com/KharinTymofii/CSIRO-Image2Biomass-Prediction
   cd CSIRO-Image2Biomass-Prediction
   ```

2. Sync the dependencies:

   ```bash
   uv sync
   ```

3. Run what you want using `uv run <script>` command. For example, to update the README file with the countdown to the deadline, run:

   ```bash
   uv run src/scripts/script.py
   ```

   or directly with Python:

   ```bash
   # Activate your virtual environment first
   # (Windows)
   .venv\Scripts\Activate

   # (macOS/Linux)
   source .venv/bin/activate

   # Run the script
   python src/scripts/script.py
   ```

4. To run a Jupyter notebook, ensure you have selected the correct Python kernel in your Jupyter environment. You can do this by opening the notebook and selecting the kernel that corresponds to your virtual environment.
