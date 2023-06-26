# qNetTI: Quantum Network Topology Inferrer

*Python tools and demos for inferring quantum network topology.*

[![Tests](https://github.com/ChitambarLab/qNetTI/actions/workflows/run_tests.yml/badge.svg?branch=main)](https://github.com/ChitambarLab/qNetTI/actions/workflows/run_tests.yml)[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)[![DOI](https://zenodo.org/badge/581250970.svg)](https://zenodo.org/badge/latestdoi/581250970)

See our preprint titled "Inferring Quantum Network Topology using Local Qubit Measurements" for details [https://arxiv.org/abs/2212.07987](https://arxiv.org/abs/2212.07987).

## Development


### Git Flow

It is best practice to develop new code in a `feature-branch`, and to merge that code into `main` via a *pull request*.
Merging into `main` will require a code review and automated tests to pass. 
For instructions, please review the [Git Flow Development Guide](https://github.com/ChitambarLab/Development-Guide#git-flow).

### Environment Setup

To ensure a consistent development environment, use the [Anaconda](https://docs.conda.io/projects/conda/en/latest/glossary.html#anaconda-glossary).
Follow the Anaconda [installation instructions](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html#installation) to set up the `conda` command line tool.
For more details on how to use `conda` see the [managing environments web page](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).

To create the `qnetti-dev` environment, navigate to the root directory of the repository and follow these steps.

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

### Running Demos

Demos are written using [Jupyter Notebooks](https://jupyter.org/).
When the conda development environment is set up, a Jupyter notebook server can be set up locally:

```
(qnetti-dev) $ jupyter notebook
```

This command should launch a web page in your default browser.
From there navigate to the `./demos` directory to view the available notebooks.

### Running Tests

All developed code should be tested for correctness.
We use the [`pytest`](https://docs.pytest.org/en/7.2.x/) framework for this task.
To run tests, use the following commands.

Run a particular test:

```
(qnetti-dev) $ pytest ./test/path/to/test_file.py
```

Run all tests in the `./test` directory:

```
(qnetti-dev) $ pytest
```

### Formatting Code

Before committing changes, please autoformat your code.

```
(qnetti-dev) $ black -l 100 test src demos
```


### Project Structure

* `./src/qnetti` - Application code.
* `./test` - Unit tests for application code.
* `./script` - *(Not Currently Implemented)* Numerical experiments and data collection.
* `./data` - *(Not Currently Implemented)* Stored data from numerical experiments.
* `./demos` - User oriented notebooks demoing the application of our code. 

