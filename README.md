# qNetTI: Quantum Network Topology Inferrer

*Python tools and demos for inferring quantum network topology.*

[![Latest](https://img.shields.io/badge/docs-latest-blue.svg)](https://chitambarlab.github.io/qNetTI/index.html)[![PyPI version](https://badge.fury.io/py/qNetTI.svg)](https://badge.fury.io/py/qNetTI)[![Tests](https://github.com/ChitambarLab/qNetTI/actions/workflows/run_tests.yml/badge.svg?branch=main)](https://github.com/ChitambarLab/qNetTI/actions/workflows/run_tests.yml)[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)[![DOI](https://zenodo.org/badge/581250970.svg)](https://zenodo.org/badge/latestdoi/581250970)

## Features

QNetTI extends [PennyLane](https://pennylane.ai) and the [Quantum Network Variational Optimizer (QNetVO)](https://chitambarlab.github.io/qNetVO/index.html) with variational quantum network inference functionality. The goal of which is to determine the entanglement/correlation structure of source nodes in a quantum network using variational quantum optimization of local measurements. Our methods are compatible with both quantum hardware and simulations thereof.

See our preprint titled "Inferring Quantum Network Topology using Local Measurements" for details [https://arxiv.org/abs/2212.07987](https://arxiv.org/abs/2212.07987).

Please review the [documentation](https://chitambarlab.github.io/qNetTI/index.html) for details regarding this project.  

## Quick Start

Install qNetTI:

```
$ pip install qnetti
```

Install PennyLane:

```
$ pip install pennylane==0.29.1
```

Install QNetVO:

```
$ pip install qnetvo==0.4.2
```

Import packages:

```
import pennylane as qml
import qnetvo
import qnetti
```

<div class="admonition note">
<p class="admonition-title">
Note
</p>
<p>
For optimal use, QNetTI should be used with the compatible versions of PennyLane and QNetVO. Version compatiblity may change in a future release of QNetTI
</p>
</div>

## Project Structure

* `./src/qnetti` - Application code.
* `./test` - Unit tests for application code.
* `./script` - Scripts for numerical experiments, data collection, and plotting.
* `./data` - Stored data from numerical experiments.
* `./demos` - User oriented notebooks demoing the application of our code. 
* `./docs` - Source code for generating the static documentation pages.

## Contributing

We welcome outside contributions to qNetTI.
Please see the [Contributing](https://chitambarlab.github.io/qNetTI/development.html)
page for details and a development guide. 

## How to Cite

[![DOI](https://zenodo.org/badge/581250970.svg)](https://zenodo.org/badge/latestdoi/581250970)

See [CITATION.bib](https://github.com/ChitambarLab/qNetTI/blob/main/CITATION.bib) for a BibTex reference to qNetVO.

## License

QNetTI is free and open-source.
The software is released under the Apache License, Version 2.0.
See [LICENSE](https://github.com/ChitambarLab/qNetTI/blob/main/LICENSE) for details.

## Acknowledgements

This material is based upon work supported by the U.S. Department of Energy, Office of Science, National
Quantum Information Science Research Centers, and the Office of Advanced Scientific Computing Research,
Accelerated Research for Quantum Computing program under contract number DE-AC02-06CH11357.



