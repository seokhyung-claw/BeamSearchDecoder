# Beam Search Decoder for Quantum LDPC Codes

This repository contains the source code required to reproduce the simulation results presented in the paper:
**[Beam Search Decoder for Quantum LDPC Codes](https://arxiv.org/abs/2512.07057)**.

## Reproduction Instructions

To reproduce the results, follow these two steps:

1. **Build the Decoder Extension:**
   Navigate to the `decoder` directory and compile the C++ extension:
   ```bash
   cd decoder
   python3 setup.py build_ext --inplace
   ```
   Note: You might be asked to install some Python packages such as `ldpc` and `stimbposd`.

2. **Run the Simulations:** Open `Beam_Search.ipynb` and execute the cells following the instructions provided within the notebook.

## File Overview

* `StimCircuit/`: Stim circuits used for simulation.

* `simulation_results/`: Directory for storing simulation outputs.

* `decoder/src_cpp/`: C++ source files (main file: `beam_search.hpp`).

* `decoder/beam_search_decoder/`: Python wrappers.

* `Beam_Search.ipynb`: Notebook to reproduce paper results.

* `simulation_functions.py`: Python scripts that are used in `Beam_Search.ipynb`