# Asteroid — Rust + WebAssembly

Working asteroids clone, all gameplay logic in Rust, rendered to HTML5 canvas via wasm-bindgen.

## Build

You need `wasm-pack` installed: `cargo install wasm-pack`

```
wasm-pack build --target web --release
```

Output: `pkg/asteroid_wasm.js` + `pkg/asteroid_wasm_bg.wasm`

## Run

Serve over HTTP (browsers won't load WASM from `file://`):

```
python -m http.server 8080
```

Open http://localhost:8080

## Controls

- ← → rotate
- ↑ thrust
- SPACE fire

## Code shape

- `src/lib.rs` — game state, physics, collision, render via `CanvasRenderingContext2d`
- `Cargo.toml` — wasm-bindgen + web-sys feature flags
- `index.html` — host page that imports the wasm module

## Notes

- Screen-wrap physics (no edges)
- Asteroids split into 2 smaller pieces when shot (until size < 20)
- Game over on collision with asteroid; refresh to restart
- ~150 lines of Rust, no external game engine, all browser APIs via web-sys
