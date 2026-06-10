---
title: New `Tonemapping::GranTurismo7` variant
pull_requests: []
---

The `Tonemapping` enum has a new variant: `GranTurismo7`, a native port of Polyphony
Digital's Gran Turismo 7 tone-mapping operator.

If you exhaustively `match` on `Tonemapping` (it is not `#[non_exhaustive]`), add an arm for
`Tonemapping::GranTurismo7` or a wildcard arm:

```rust
match tonemapping {
    // ...existing arms...
    Tonemapping::GranTurismo7 => { /* ... */ }
}
```

No behavior changes for existing tonemappers; the default remains `Tonemapping::TonyMcMapface`.
