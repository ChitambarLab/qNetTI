# qNetTI: Quantum Network Topology Inferrer
Python tools and demos for inferring quantum network topology using local qubit measurements (see [https://arxiv.org/abs/2212.07987](https://arxiv.org/abs/2212.07987)).

## Development


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


### Project Structure

* `./src/qnetti` - Application code.
* `./test` - Unit tests for application code.
* `./script` - Numerical experiments and data collection.
* `./data` - Stored data from numerical experiments.
* `./demos` - User oriented demonstrations of application code


### Building Documentation


### API

qubit_characteristic_matrix(qnetvo.PrepareNode)

qubit_covariance_matrix(qnetvo.PrepareNode)

network_topology(qubit_characteristic_matrix or qubit_covariance_matrix) 





