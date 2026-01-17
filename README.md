# Beyond Time-Optimality: A Harm-Minimizing Medical Drone Delivery Planning Framework (IEEE ICUAS '26)

**Authors:** Arjan Khadka & Rithik Satarla (Co-First Authors)  
**Affiliation:** Unaffiliated (Paris, TX / Suwanee, GA)

## Overview
This repository contains the source code and LaTeX manuscript for our paper submitted to the **2026 International Conference on Unmanned Aircraft Systems (ICUAS)**.

We propose **HARE (Harm-Aware Routing Engine)**, a novel framework that shifts medical drone delivery objectives from minimizing time to **minimizing irreversible physiological harm**. By modeling patient-specific "Points of No Return" (PNR), HARE reduces critical mortality events in simulation compared to traditional FIFO and Distance-Optimal approaches.

## Repository Structure
- `ICUAS_MedDrone_Paper.tex`: LaTeX source code for the conference paper.
- `simulation.py`: Complete Python simulation environment (discrete-event) used to generate the results.
- `references.bib`: BibTeX references.
- `*.png`: Generated graphs used in the paper (Fleet Sensitivity, MCI Surge, etc.).

## Running the Simulation
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the script:
   ```bash
   python simulation.py
   ```
   This will reproduce the experiments and regenerate the figures found in the paper (e.g., `mci_surge_response.png`).

## Key Features
- **Harm Functions**: Non-linear deterioration models (Sigmoid for Cardiac, Linear for Trauma).
- **MCI Surge Mode**: Simulates a 500% load spike to test system resilience.
- **Hardware Stats**: Modeled on DJI Matrice 300 RTK + NVIDIA Jetson Orin Nano.

## License
MIT License.
