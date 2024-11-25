# MPS to circuit

Tool to convert a matrix product state to a Qiskit quantum circuit.

## Installation

This package and all dependencies can be installed using [uv](https://github.com/astral-sh/uv).

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.12
uv venv --python 3.12
uv python pin 3.12
uv pip install -e .
```

## Examples

One can run the examples as follows.

```sh
uv run examples/<example_name>.py
```

## Testing

One can lint the code using [pytest](https://github.com/pytest-dev/pytest) .

```sh
uv run pytest
```

## Linting

One can lint the code using the [Ruff Linter](https://docs.astral.sh/ruff/linter/).

```sh
uv run ruff check
```

## Formatting

One can format the code using the [Ruff Formatter](https://github.com/astral-sh/ruff/formatter/).

```sh
uv run ruff format
```

## Sorting imports

In order to both sort imports and format, call the Ruff linter and then the formatter:

```sh
uv run ruff check --select I --fix
uv run ruff format
```

## License

[Apache License 2.0](LICENSE.txt)

## Code of conduct

All members of this project agree to adhere to the Qiskit Code of Conduct listed
[here](https://github.com/Qiskit/qiskit/blob/master/CODE_OF_CONDUCT.md).

## Acknowledgements

This work was supported by the Hartree National Centre for Digital Innovation, a collaboration
between the Science and Technology Facilities Council and IBM.
