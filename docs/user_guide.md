# User Guide

## pyproject.toml file and dependency management

If your project doesn't have a pyproject.toml file, simply copy the one from the
template and update file according to your project.

For managing dependencies, this template makes use of [uv](https://docs.astral.sh/uv/),
which according to some [benchmarks](https://github.com/astral-sh/uv/blob/main/BENCHMARKS.md)
is faster than alternative like Poetry (which our original AI Engineering Template
makes use of).

Hence, be sure to install uv in order to to setup the development virtual environment.
Instructions for installing uv can be found [here](https://docs.astral.sh/uv/getting-started/installation/).
Note that uv supports [optional dependency groups](https://docs.astral.sh/uv/concepts/projects/dependencies/#dependency-groups)
which helps to manage dependencies for different parts of development such as `documentation`,
`testing`, etc. The core dependencies are installed using the command:

```bash
uv sync
```

Additional dependency groups can be installed using the `--group` flag followed
by the group name. For example:

```bash
uv sync --all-extras --group docs --group test
```

!!! important "mypy configuration options"
    By default, the `mypy` configuration in the `pyproject.toml` disallows subclassing
    the `Any` type - `allow_subclassing_any = false`. In cases where the type checker
    is not able to determine the types of objects in some external library (e.g. `PyTorch`),
    it will treat them as `Any` and raise errors. If your codebase has many of such
    cases, you can set `allow_subclassing_any = true` in the `mypy` configuration or
    remove it entirely to use the default value (which is `true`). For example, in
    a `PyTorch` project, subclassing `nn.Module` will raise errors if `allow_subclassing_any`
    is set to `false`.


## pre-commit

You can use [pre-commit](https://pre-commit.com/) to run pre-commit hooks (code checks,
liniting, etc.) when you run `git commit` and commit your code. Simply copy the
`.pre-commit-config.yaml` file to the root of the repository and install the test
dependencies which installs pre-commit. Then run:

```bash
pre-commit install
```

If you prefer to not enforce using pre-commit every time you run `git commit`,
you will have to run `pre-commit run --all-files` from the command line before you
commit your code.

!!! important "hook configuration"
    Some of the pre-commit hooks use [supported hooks](https://pre-commit.com/hooks.html)
    from the web.

    For some others, they are locally installed and hence use the python virtual environment
    locally. If `language` is set to `python`, each time the hook is installed, a separate
    python virtual environment is created and you can specify dependencies needed using
    `additional_dependencies`.

    If `language` is set to `system`, the activated python virtual environment is used and
    and hence you have to ensure that the required dependencies and their versions are
    correctly installed.

    ```yaml
      - repo: local
        hooks:
        - id: pytest
          name: pytest
          entry: python3 -m pytest -m "not integration_test"
          language: python/system # set according to your project needs
    ```

!!! warning "typos"
    The [typos](https://github.com/crate-ci/typos) pre-commit hook is used to check for
    common spelling mistakes in the codebase. While useful, it may require some
    configuration to ignore certain words or phrases that are not typos. You can
    [configure the typos hook](https://github.com/crate-ci/typos/blob/master/docs/reference.md)
    in the `pyproject.toml` file. In a large codebase, it may be useful to disable
    the typos hook and only run it occasionally on the entire codebase.


### pre-commit ci

Instead of fixing pre-commit errors manually, a CI to fix them as well as update
pre-commit hooks periodically can be enabled for your repository. Please check
[pre-commit.ci](https://pre-commit.ci/) and add your repository. The configuration for
``pre-commit.ci`` can be added to the ``.pre-commit-config.yaml`` file.


## documentation

If your project doesn't have documentation, copy the directory named `docs` to the root
directory of your repository. This template uses [MkDocs](https://www.mkdocs.org/) with
the [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) theme.

In order to build the documentation, install the documentation dependencies as mentioned
in the previous section, then run the command:

```bash
mkdocs build
```

If you're making changes to the docs, and want to serve them locally on your machine,
then you can use this command instead:

```bash
mkdocs serve
```

The above will launch the docs locally on `http://127.0.0.1:8000`, which you can
enter into your browser of choice. Conveniently, this process also watches for any
changes you make to the docs and will update them as they occur.

You can configure the documentation by updating the `mkdocs.yml` file at the root of
your repository. The markdown files in the `docs` directory can be updated to reflect
the project's documentation.


## github actions

The template consists of some github action continuous integration workflows that you
can add to your repository.

The available workflows are:

- [code checks](https://github.com/VectorInstitute/aieng-template/blob/main/.github/workflows/code_checks.yml): Static code analysis, code formatting and unit tests
- [documentation](https://github.com/VectorInstitute/aieng-template/blob/main/.github/workflows/docs.yml): Project documentation including example API reference
- [integration tests](https://github.com/VectorInstitute/aieng-template/blob/main/.github/workflows/integration_tests.yml): Integration tests
- [publish](https://github.com/VectorInstitute/aieng-template/blob/main/.github/workflows/publish.yml):
Publishing python package to PyPI. Create a `PYPI_API_TOKEN` and add it to the
repository's actions [secret variables](https://docs.github.com/en/actions/security-guides/using-secrets-in-github-actions)
in order to publish PyPI packages when new software releases are created on Github.

The test workflows also compute coverage and upload code coverage metrics to
[codecov.io](https://app.codecov.io/gh/VectorInstitute/aieng-template). Create a
`CODECOV_TOKEN` and add it to the repository's actions [secret variables](https://docs.github.com/en/actions/security-guides/using-secrets-in-github-actions).

!!! warning "codecov"
    The [codecov](https://app.codecov.io/github/VectorInstitute) tool is subscribed under the free tier
    which makes it usable only for public open-source repos. Hence, if you would like to develop in a
    private repo, it is recommended to remove the codecov actions from the github workflow files.
