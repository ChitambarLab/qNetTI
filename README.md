# qNetTI: Quantum Network Topology Inferrer

*Python tools and demos for inferring quantum network topology.*

[![Tests](https://github.com/ChitambarLab/qNetTI/actions/workflows/run_tests.yml/badge.svg?branch=main)](https://github.com/ChitambarLab/qNetTI/actions/workflows/run_tests.yml)[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

See our preprint titled "Inferring Quantum Network Topology using Local Qubit Measurements" for details [https://arxiv.org/abs/2212.07987](https://arxiv.org/abs/2212.07987).

## Development

### Git Flow

### Environment Setup

1. Create the `qnetti-dev` Anaconda environment described in `environment.yml`:

```
(base) $ conda env create -f environment.yml
```

2. Activate the `qnetti-dev` environment:

```
(base) $ conda activate qnetti-dev
```

3. Install the `qnetti` package locally in editable mode:

```
(qnetti-dev) $ pip install -e .
```

### Running Tests

Run a particular test:

```
(qnetti-dev) $ pytest ./test/path/to/test_file.py
```

Run all tests in the `./test` directory:

```
(qnetti-dev) $ pytest
```

### Formatting Code


```
(qnetti-dev) $ black -l 100 test src
```


### Project Structure

* `./src/qnetti` - Application code.
* `./test` - Unit tests for application code.
* `./script` - *(Not Currently Implemented)* Numerical experiments and data collection.
* `./data` - *(Not Currently Implemented)* Stored data from numerical experiments.
* `./demos` - *(Not Currently Implemented)* User oriented notebooks demoing the application of our code. 


### Building Documentation


### API

qubit_characteristic_matrix(qnetvo.PrepareNode)

qubit_covariance_matrix(qnetvo.PrepareNode)

network_topology(qubit_characteristic_matrix or qubit_covariance_matrix) 





