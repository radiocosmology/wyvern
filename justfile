set shell := ["bash", "-eo", "pipefail", "-c"]

bindings_dir := "bindings/python"
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

# build all workspace docs
doc:
    cargo doc --workspace --lib --release --no-deps --document-private-items

# build all workspace docs and open
doc-open: doc
    open "target/doc/wyvern/index.html"

# build all workspace docs and modify build for GitHub pages
doc-hosted: doc
    echo "<meta http-equiv=\"refresh\" content=\"0; url=wyvern/index.html\">" > "target/doc/index.html"
    touch "target/doc/.nojekyll"

# --- Stub generation ---

# generate python stubs
gen-stubs:
    cargo run --bin stub_gen --manifest-path {{bindings_dir}}/Cargo.toml --features stub-gen
    uvx ruff check --fix
    uvx ruff format

# generate and check that python stubs are up to date
check-stubs: gen-stubs
    git diff --exit-code -- {{py_src}} || \
        (echo "Generated stubs are out of date. Run 'just gen-stubs' and commit the changes." && exit 1)

# --- Python ---

# install python bindings with dev build
develop:
    cd {{bindings_dir}} && maturin develop

# install python bindings with release build
release:
    cd {{bindings_dir}} && maturin develop --release

# --- Aggregate ---

# run all CI checks
ci: clippy test doc check-stubs

# format all rust and python files
fmt:
    cargo fmt --all
    uvx ruff format
