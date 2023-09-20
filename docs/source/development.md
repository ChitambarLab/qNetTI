# Development


## Git Flow

It is best practice to develop new code in a `feature-branch`, and to merge that code into `main` via a *pull request*.
Merging into `main` will require a code review and automated tests to pass. 
For instructions, please review the [Git Flow Development Guide](https://github.com/ChitambarLab/Development-Guide#git-flow).

## Environment Setup

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

## Running Demos

Demos are written using [Jupyter Notebooks](https://jupyter.org/).
When the conda development environment is set up, a Jupyter notebook server can be set up locally:

```
(qnetti-dev) $ jupyter notebook
```

This command should launch a web page in your default browser.
From there navigate to the `./demos` directory to view the available notebooks.

## Running Tests

All developed code should be tested for correctness.
We use the [pytest](https://docs.pytest.org/en/7.2.x/) framework for this task.
To run tests, use the following commands.

Run a particular test:

```
(qnetti-dev) $ pytest ./test/path/to/test_file.py
```

Run all tests in the `./test` directory:

```
(qnetti-dev) $ pytest
```

## Formatting Code

Before committing changes, please autoformat your code.

```
(qnetti-dev) $ black -l 100 test src demos
```

## Building and Viewing Documenation

We use the Sphinx framework for autogenerating code documentation. To build the docs from scratch run:

```
(qnetti-dev) $ sphinx-build -b html docs/source/ docs/build/html
```

This command will build a static HTML site in the `docs/build/html` directory. To view the site navigate to the `docs/build/html` directory and run

```
(qnetti-dev) $ python -m http.server --bind localhost
```

to initialize an HTTP server hosting the site locally.

Semantic Versioning
-------------------

This project uses `semantic versioning <https://semver.org/>`_ to manage
releases and maintain consistent software.

Packaging and Releases
----------------------

This project is packaged using PyPI, the `Python Package Index <https://pypi.org/>`_.
Please refer to `this tutorial <https://packaging.python.org/en/latest/tutorials/packaging-projects/>`_ for details on releasing a new version.