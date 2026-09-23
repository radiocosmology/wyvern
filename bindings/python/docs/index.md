# wyvern

## Installation {: #installation }
To install from source:
```bash
pip install git+https://github.com/ljgray/wyvern.git#subdirectory=bindings/python
```
or using [uv](https://docs.astral.sh/uv/):
```bash
uv run --active --directory "bindings/python" maturin develop --release
```

- [Python User Guide](user_guide/index.md)
- [Python API Overview](api/index.md)
- [Rust API](rust/index.html)
