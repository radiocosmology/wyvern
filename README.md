<h1 align="center">wyvern</h1>

**wyvern** provides fast implementations of a variety of algorithms, motivated by
the requirements of [draco](https://github.com/radiocosmology/draco).

- [Installation](https://ljgray.github.io/wyvern/#installation)
- [User guide](https://ljgray.github.io/wyvern/user_guide/)
- [API reference](https://ljgray.github.io/wyvern/api/)


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
