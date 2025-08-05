.PHONY: javascript
javascript:
	# Browser ESM build
	wasm-pack build --target web --out-dir javascript/dist/web --features wasm-binding --no-default-features

	# Node.js ESM/CJS-compatible build
	wasm-pack build --target nodejs --out-dir javascript/dist/node --features wasm-binding --no-default-features


.PHONY: python-local
python-local:
	maturin develop -F python-binding --release
