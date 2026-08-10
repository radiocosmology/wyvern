set shell := ["bash", "-eo", "pipefail", "-c"]

bindings_dir := "bindings/python"
py_src := bindings_dir / "py_src"

default:
    just --list

# --- Cargo ---

clippy:
    cargo clippy --workspace --all-targets --all-features -- -D warnings

test:
    cargo nextest run --workspace --profile ci --no-tests="warn"
    cargo test --doc --workspace --all-features

doc:
    cargo doc --workspace --no-deps --document-private-items

doc-hosted: doc
    echo "<meta http-equiv=\"refresh\" content=\"0; url=wyvern/index.html\">" > "target/doc/index.html"
    touch "target/doc/.nojekyll"

# --- Stub generation ---

gen-stubs:
    cargo run --bin stub_gen --manifest-path {{bindings_dir}}/Cargo.toml --features stub-gen
    uvx ruff check --fix {{py_src}}
    uvx ruff format {{py_src}}

check-stubs: gen-stubs
    git diff --exit-code -- {{py_src}} || \
        (echo "Generated stubs are out of date. Run 'just gen-stubs' and commit the changes." && exit 1)

# --- Python ---

develop:
    cd {{bindings_dir}} && maturin develop

release:
    cd {{bindings_dir}} && maturin develop --release

# --- Aggregate ---
ci: clippy test doc check-stubs

fmt:
    cargo fmt --all
    uvx ruff format {{py_src}}
