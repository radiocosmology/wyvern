set shell := ["bash", "-eo", "pipefail", "-c"]

base_dir:= justfile_directory()
bindings_dir := base_dir / "bindings/python"
py_src := bindings_dir / "py_src"

# list all available commands (default behaviour)
default:
    just --list

# --- Cargo ---

# lint the entire workspace with clippy
clippy:
    cargo clippy --workspace --all-targets --all-features -- -D warnings

# run all workspace tests
test:
    cargo nextest run --workspace --profile ci --no-tests="warn"
    cargo test --doc --workspace --all-features

# run all benchmarks
bench:
    cargo bench --benches --package "wyvern" --no-fail-fast --features "mimalloc"

# --- Stub generation ---

# generate python stubs
gen-stubs:
    cargo run --bin stub_gen --manifest-path {{bindings_dir}}/Cargo.toml --features stub-gen
    uvx --directory {{bindings_dir}} ruff check --fix
    uvx --directory {{bindings_dir}} ruff format

# generate and check that python stubs are up to date
check-stubs: gen-stubs
    git diff --exit-code -- {{py_src}} || \
        (echo "Generated stubs are out of date. Run 'just gen-stubs' and commit the changes." && exit 1)

# --- Python ---

# install python bindings with dev build
test-setup:
    uv sync --directory {{bindings_dir}} --group test

dev-setup:
    uv sync --directory {{bindings_dir}} --group dev

develop: test-setup
    uv run --directory {{bindings_dir}} maturin develop --uv

# install python bindings with release build
release:
    uv run --active --directory {{bindings_dir}} maturin develop --release

# install python test dependencies and run the pytest suite
test-python: develop
    uv run --directory {{bindings_dir}} pytest tests

# --- Docs ---
    
# build all workspace docs
doc: gen-stubs
    cargo doc --workspace --lib --release --no-deps

# build all workspace docs and open
doc-open: doc
    open "target/doc/wyvern/index.html"

# build all workspace docs and modify build for GitHub pages
doc-hosted: doc
    echo "<meta http-equiv=\"refresh\" content=\"0; url=wyvern/index.html\">" > "target/doc/index.html"
    touch "target/doc/.nojekyll"

# build python docs
doc-python: dev-setup
    uv run --directory {{bindings_dir}} zensical build --config-file zensical.toml

# combine python and rust docs
doc-site: doc-hosted doc-python
    mkdir -p {{bindings_dir}}/site/rust
    cp -r {{base_dir}}/target/doc/* {{bindings_dir}}/site/rust/

doc-serve: doc-site
    uv run --no-project python -m http.server 8000 --directory {{bindings_dir}}/site

# --- Aggregate ---

# run all CI checks
ci: clippy test test-python doc doc-python check-stubs

# format all rust and python files
fmt:
    cargo fmt --all
    uvx --directory {{bindings_dir}} ruff format
