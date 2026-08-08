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

### Development

This project uses [maturin](https://github.com/PyO3/maturin). For development work,
clone the project and install into an active virtual environment:
```bash
pip install maturin

cd bindings/python
maturin develop [--release]
```

Installing with pip will always use a `release` build.
