
# DropGNN: Graph Neural Networks with DropNode for Drug Discovery

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2111.06283-b31b1b.svg)](https://arxiv.org/abs/2111.06283)

This repository contains an implementation of DropGNN, a graph neural network approach with DropNode regularization for drug discovery tasks, based on the research paper ["DropGNN: Random Dropouts Increase the Expressiveness of Graph Neural Networks"](https://arxiv.org/pdf/2111.06283).

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Repository Structure](#repository-structure)
- [Results](#results)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

## Overview

DropGNN introduces random dropouts of entire nodes during training to improve the expressiveness and performance of Graph Neural Networks (GNNs) on molecular property prediction tasks. Key features:

- Implements DropNode regularization for GNNs
- Improves model generalization
- Enhances expressiveness of message passing GNNs
- Particularly effective for drug discovery applications

The original paper demonstrates that DropGNN achieves state-of-the-art performance on several molecular property prediction benchmarks.

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/dropgnn-implementation.git
cd dropgnn-implementation
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Jupyter Notebook
The main implementation is provided in `DropGNN_Implementation.ipynb`. Open it with Jupyter:
```bash
jupyter notebook DropGNN_Implementation.ipynb
```

### Python Script
Alternatively, you can use the standalone Python implementation:
```bash
python dropgnn.py
```

### Configuration
Modify the hyperparameters in the notebook/script to experiment with different settings:
- Dropout rate
- Number of GNN layers
- Hidden layer dimensions
- Learning rate
- Training epochs

## Repository Structure

```
dropgnn-implementation/
├── DropGNN_Implementation.ipynb    # Main implementation notebook
├── dropgnn.py                     # Python implementation
├── documentation/                  # Additional documentation
│   ├── methodology.md
│   └── results_analysis.md
├── requirements.txt                # Dependencies
└── LICENSE
```

## Results

Include here any results you've obtained from your implementation, or refer to the original paper's results if you haven't run experiments yet.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements.

## Citation

If you use this implementation in your research, please cite the original paper:

```bibtex
@article{papp2021dropgnn,
  title={DropGNN: Random Dropouts Increase the Expressiveness of Graph Neural Networks},
  author={P{\'a}l Andras Papp and Roger Wattenhofer},
  journal={arXiv preprint arXiv:2111.06283},
  year={2021}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```

### Notes:
1. Replace `yourusername` with your actual GitHub username in the clone URL
2. Add your actual `requirements.txt` file with the necessary dependencies
3. Include any additional documentation you have in the documentation folder
4. Add your implementation results if available
5. Customize the repository structure if your files are organized differently

Would you like me to modify any specific part of this README or add more details about your particular implementation?
