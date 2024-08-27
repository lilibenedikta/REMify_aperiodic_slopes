Here's a revised and more concise README file for your GitHub repository:

---

# Automated REM Detection using YASA and MNE-Python

## Authors
Lili Benedikta Nádasy - [lilibenedikta@gmail.com](mailto:lilibenedikta@gmail.com)

## Overview

This repository contains a Python script designed to support the analysis described in the paper:

**Aperiodic Neural Activity Distinguishes Between Phasic and Tonic REM Sleep**  
*Yevgenia Rosenblum, Tamás Bogdány, Lili Benedikta Nádasy, Ilona Kovács, Ferenc Gombos, Péter Ujma, Róbert Bódizs, Nico Adelhöfer, Péter Simor, Martin Dresler*

### Description

The provided code focuses on the automated detection of REM sleep using the YASA library and MNE-Python. It processes EEG data, applying specific filters and detection algorithms to identify REM phases. The output is designed to facilitate further analysis as detailed in the associated research.

### Dependencies

To run the script, ensure you have the following Python packages installed:

```sh
pip install numpy pandas mne yasa
```

### How to Use

- **Prepare Input Data:** Ensure your EDF, hypnogram, and artefact files are correctly formatted and placed in the designated directories.
- **Execute the Script:** Adjust the file paths as necessary and run the script to process the EEG data.
- **Output:** The results are stored in specified directories as .txt files.

### Contact

For inquiries, please reach out to Lili Benedikta Nádasy at [lilibenedikta@gmail.com](mailto:lilibenedikta@gmail.com).

---

This README succinctly ties the code to the broader research while maintaining a level of abstraction suitable for its supporting role.
