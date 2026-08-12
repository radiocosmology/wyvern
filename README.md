<h1 align="center">wyvern</h1>

**wyvern** provides fast implementations of a variety of algorithms, motivated by
the requirements of [draco](https://github.com/radiocosmology/draco).

## Bindings
- [Python](https://github.com/ljgray/wyvern/tree/main/bindings/python)

## Installation

To install from source:
```bash
pip install git+https://github.com/ljgray/wyvern.git#subdirectory=bindings/python
```

## Development
### Tools

This project uses [just](https://github.com/casey/just) as a command runner, which
doubles as a library of recipes for building, linting, etc...

List all available commands:
```bash
just
```

Python bindings are built using [maturin](https://github.com/PyO3/maturin). For
development work, clone the project and install into an active virtual environment:
```bash
pip install maturin

just develop
```

Installing with pip will always use a `release` build.
