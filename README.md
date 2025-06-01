
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
git clone https://github.com/JaanuNan/DropGNN.git
cd dropgnn.ipynb
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```


## Usage

### Jupyter Notebook
The main implementation is provided in `DropGNN.ipynb`. Open it with Jupyter:
```bash
jupyter notebook dropgnn.ipynb
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
├── dropgnn.ipynb    # Main implementation notebook
├── dropgnn.py                     # Python implementation
├── documentation              # Implementation documentation
├── Research paper                # Paper Pdf
└── LICENSE
```

## Results

![image](https://github.com/user-attachments/assets/6fa60768-363a-4f32-83d0-a8438a07bd64)


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
