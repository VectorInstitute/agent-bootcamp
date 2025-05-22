# AI Engineering template (with uv)

----------------------------------------------------------------------------------------

[![code checks](https://github.com/VectorInstitute/aieng-template-uv/actions/workflows/code_checks.yml/badge.svg)](https://github.com/VectorInstitute/aieng-template-uv/actions/workflows/code_checks.yml)
[![integration tests](https://github.com/VectorInstitute/aieng-template-uv/actions/workflows/integration_tests.yml/badge.svg)](https://github.com/VectorInstitute/aieng-template-uv/actions/workflows/integration_tests.yml)
[![docs](https://github.com/VectorInstitute/aieng-template-uv/actions/workflows/docs.yml/badge.svg)](https://github.com/VectorInstitute/aieng-template-uv/actions/workflows/docs.yml)
[![codecov](https://codecov.io/github/VectorInstitute/aieng-template-uv/graph/badge.svg?token=83MYFZ3UPA)](https://codecov.io/github/VectorInstitute/aieng-template-uv)
![GitHub License](https://img.shields.io/github/license/VectorInstitute/aieng-template-uv)

A template repo for AI Engineering projects (using ``python``) and ``uv``. This
template is like our original AI Engineering [template](https://github.com/VectorInstitute/aieng-template),
however, unlike how that template uses poetry, this one uses uv for dependency
management (as well as packaging and publishing).

## 🧑🏿‍💻 Developing

### Installing dependencies

The development environment can be set up using
[uv](https://github.com/astral-sh/uv?tab=readme-ov-file#installation). Hence, make sure it is
installed and then run:

```bash
uv sync
source .venv/bin/activate
```

In order to install dependencies for testing (codestyle, unit tests, integration tests),
run:

```bash
uv sync --dev
source .venv/bin/activate
```

In order to exclude installation of packages from a specific group (e.g. docs),
run:

```bash
uv sync --no-group docs
```

If you're coming from `poetry` then you'll notice that the virtual environment
is actually stored in the project root folder and is by default named as `.venv`.
The other important note is that while `poetry` uses a "flat" layout of the project,
`uv` opts for the the "src" layout. (For more info, see [here](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/))

### Poetry to UV

The table below provides the `uv` equivalent counterparts for some of the more
common `poetry` commands.

| Poetry                                               | UV                                          |
|------------------------------------------------------|---------------------------------------------|
| `poetry new <project-name>`  # creates new project   | `uv init <project-name>`                    |
| `poetry install`  # installs existing project        | `uv sync`                                   |
| `poetry install --with docs,test`                    | `uv sync --group docs --group test`         |
| `poetry add numpy`                                   | `uv add numpy`                              |
| `poetry add pytest pytest-asyncio --groups dev`      | `uv add pytest pytest-asyncio --groups dev` |
| `poetry remove numpy`                                | `uv remove numpy`                           |
| `poetry lock`                                        | `uv lock`                                   |
| `poetry run <cmd>`  # runs cmd with the project venv | `uv run <cmd>`                              |
| `poetry build`                                       | `uv build`                                  |
| `poetry publish`                                     | `uv publish`                                |
| `poetry cache clear pypi --all`                      | `uv cache clean`                            |

For the full list of `uv` commands, you can visit the official [docs](https://docs.astral.sh/uv/reference/cli/#uv).

### Tidbit

If you're curious about what "uv" stands for, it appears to have been more or
less chosen [randomly](https://github.com/astral-sh/uv/issues/1349#issuecomment-1986451785).
