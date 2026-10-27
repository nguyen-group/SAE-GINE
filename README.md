# SAE-GINE
Attribute-Enriched Crystal Graph Learning for Optical-Spectrum Prediction and Photovoltaic Materials Discovery.

<img src="https://github.com/user-attachments/assets/3cbd606a-bd7d-4634-817c-32d604739e76" width="600">

# Requirement

## Database
SAE-GINE requires the database ``database_300.zip'' from M. Grunert et al. Phys. Rev. Materials 8, L122201 (2024): https://doi.org/10.1103/PhysRevMaterials.8.L122201

This database includes 9915 dielectric tensors of semiconductors and insulators calculated in the independent-particle approximation (IPA).

## Packages
SAE-GINE also requires the packages as follows: 
- `torch`: an open-source machine learning library with strong GPU acceleration.
- `jupyterlab`: a web-based interactive development environment for notebooks, code, and data.
- `pandas`: a Python library used to work with data sets.
- `scipy`: an open-source software for mathematics, science, and engineering.
- `pymatgen`: an open-source Python library used for materials analysis, crystal structure manipulation, and high-throughput computational materials science.

# Directory Description

```md
GNNOpt
├── DFT-data
│   └── collect_data_epsilon.zip: the calculated dataset of 65 of Zintl-like and AB2C2 candidates for solar-cell application by using the DFT with Quantum ESPRESSO.
├── utils
├── sae-gine.ipynb: Main SAE-GINE file.
├── direct_alpha_SLME_500nm_heldout_FINAL.ipynb: running the SLME value
└── build_zintl_like_pool_no_element_filter_DISCOVERY_READY_DB2_FIXED.ipynb: High-throughput screen the Zintl-like and AB2C2 materials with SAE-GINE
```
# How to run
Step 1: Download the SAE-GINE package:

    git clone https://github.com/nguyen-group/SAE-GINE.git

Step 2: Go to the source code in the SAE-GINE directory to run the program:

    cd SAE-GINE
    jupyter-lab sae-gine.ipynb

Note: This program is time-consuming on an RTX 4090 (takes several days). Therefore, it should be run on a GPU workstation.
