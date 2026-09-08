# Small-scale mixing regulates the reconfiguration of the Arabian Sea oxygen minimum zone

This repository contains the analysis and figure-generation code associated with the manuscript:

**“Small-scale mixing regulates the reconfiguration of the Arabian Sea oxygen minimum zone.”**

The study uses high-resolution Seaexplorer glider observations from the northwestern Arabian Sea and Sea of Oman to investigate how distinct oxygen sources and small-scale mixing processes redistribute oxygen within the oxygen minimum zone.

## Data

The processed dataset used to reproduce the analyses is archived separately on Zenodo:

**Processed dataset:** (https://doi.org/110.5281/zenodo.22650492)

The original glider observations are also available separately:

**Raw data:** (https://doi.org/10.5281/zenodo.16976500)

Download the processed dataset and update the corresponding data paths in the notebooks before running the analysis.

## Repository structure

The analysis is organized into four Jupyter notebooks:

- `00_Necessary_variables.ipynb` — calculates and prepares the variables required for the analysis.
- `01_Basic_plots.ipynb` — preliminary analysis and visualization of the observations.
- `02_Flux_calculation.ipynb` — calculates diapycnal oxygen fluxes and associated quantities.
- `03_Paper_figures.ipynb` — performs the final analyses and generates the figures used in the manuscript.

`MAPJtools.py` contains additional functions used by the notebooks.

## Reproducing the analysis

Run the notebooks sequentially:

`00_Necessary_variables.ipynb` → `01_Basic_plots.ipynb` → `02_Flux_calculation.ipynb` → `03_Paper_figures.ipynb`

The notebooks use the processed dataset available through Zenodo and contain the calculations required to reproduce the analyses and figures presented in the manuscript.

## References of data and other software:

GEBCO Compilation Group (2023) GEBCO 2023 Grid [Dataset]. doi:10.5285/f98b053b-0cbc-6c23-e053-6c86abc0af7b

Queste, B. Y., Rollo, C. & Font, E. (2023). Glider AD2CP [Software]. (https://pypi.org/project/gliderad2cp/).

Egbert, G. D. & Erofeeva, S. Y. Efficient Inverse Modeling of Barotropic Ocean Tides. J. Atmos. Ocean. Technol. 19, 183–204 (2002).

Global Ocean Gridded L 4 Sea Surface Heights And Derived Variables Reprocessed 1993 Ongoing. (https://doi.org/https://doi.org/10.48670/moi-00148.)

McDougall, T. & Krzysik, O. Spiciness. J. Mar. Res. 73, (2015).

McDougall, T. J. & Barker, P. M. Getting started with TEOS-10 and the Gibbs Seawater (GSW) Oceanographic Toolbox. in 28 pp. (2011).

GSW Oceanographic Toolbox (TEOS-10) – For thermodynamic seawater calculations
http://www.teos-10.org
Please cite: IOC, SCOR, and IAPSO, 2010. The international thermodynamic equation of seawater – 2010: Calculation and use of thermodynamic properties. Intergovernmental Oceanographic Commission, Manuals and Guides No. 56.

cmocean colormaps – Perceptually uniform colormaps for oceanography
https://matplotlib.org/cmocean/
Please cite: Thyng, K. M., Greene, C. A., Hetland, R. D., Zimmerle, H. M., & DiMarco, S. F. (2016). True colors of oceanography: Guidelines for effective and accurate colormap selection. Oceanography, 29(3), 9–13.
