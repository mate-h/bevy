---
title: New `Tonemapping::GranTurismo7` variant
pull_requests: []
---

The `Tonemapping` enum has a new variant, `GranTurismo7`, a native port of Polyphony
Digital's Gran Turismo 7 tone-mapping operator.

`Tonemapping` is not `#[non_exhaustive]`, so any exhaustive `match` on it needs a new
`Tonemapping::GranTurismo7` arm or a wildcard:

```rust
match tonemapping {
    // existing arms
    Tonemapping::GranTurismo7 => { /* ... */ }
}
```

No behavior change for existing tonemappers; the default remains `Tonemapping::TonyMcMapface`.
