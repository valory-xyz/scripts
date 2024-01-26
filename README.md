Contains independent scripts/demos.


## fpmms Project
This following README provide guidance on understanding tools used in fpmms.

### Getting Started
Follow these steps to set up the project:

### Initialize Submodules

fpmms uses submodules, which need to be initialized:

```bash
git submodule update --init --recursive
```

### Install Dependencies
The project uses Poetry for dependency management:

1. If you don't have Poetry installed, install it first.
2. Update and install the project dependencies:
```bash
poetry update
poetry install
```

### Data Preparation
Run the `0. download_data.ipynb` notebook located in the `nbs` folder to download and prepare the data required for the project.

### Documentation
To understand the workings of the FPMMS project, refer to the following notebooks:

1. Tools Evaluation:
Location: fpmms/nbs/Tools_Evaluation.ipynb
Description: This notebook outlines how the dataset is built and evaluates the tools used in the project.

2. Costing Analysis:
Location: fpmms/nbs/costing_analysis.ipynb
Description: This notebook provides an analysis of the costs associated with running different tools within the project.

3. Traders Profitability Analysis:
Location: fpmms/nbs/profitability_analysis.ipynb
Description: This notebook provided analysis on profitability of the traders. Before running the notebbok, you could either run `profitability.py` script or get the required data by running `0. download_data.ipynb`

