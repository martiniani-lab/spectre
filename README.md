# spectre


This repositry contains scripts used to generate figures in the paper titled: "Explicit rational function solutions for the power spectral density of stochastic linear time-invariant systems."

The folder 'spectrum_general' contains classes to calculate the spectrum of an LTI SDE with a given Jacobian (J) and noise matrices (L and S), using the matrix, recursive, simulation and element-wise solution.

The folder 'model_classes' contains classes for the individual models described by nonlinear SDEs exhibiting a fixed point solution. 

The folder 'plotting_scripts' contains jupyter notebooks used to generate the plots for the models.

## Features

- **Feature 1**: A brief description of the first main feature.
- **Feature 2**: A brief description of the second main feature.
- **Feature 3**: A brief description of the third main feature.

## Installation

Follow these steps to install Spectre:

### Prerequisites
- List any required software or dependencies here.

### Installation Steps
1. Clone the repository:
    ```bash
    git clone https://github.com/martiniani-lab/spectre.git
    cd spectre
    ```
2. Add the current directory to your Python path:
    ```bash
    export PYTHONPATH=$(pwd):$PYTHONPATH
    ```

## Usage

### Example 1: Basic Usage
```python
import spectre
```

## Citation
```bibtex
@article{rawat2023element,
  title={Element-wise and Recursive Solutions for the Power Spectral Density of Biological Stochastic Dynamical Systems at Fixed Points},
  author={Rawat, Shivang and Martiniani, Stefano},
  journal={ArXiv},
  year={2023},
  publisher={arXiv}
}
```