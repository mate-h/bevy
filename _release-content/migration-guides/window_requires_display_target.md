---
title: "`Window` now requires the `DisplayTarget` component"
pull_requests: []
---

The `Window` component now requires the new `DisplayTarget` component, which
describes the display the window is presented on (paper white luminance, peak
luminance, black level, color gamut, and transfer function). It is the
foundation for upcoming HDR display output support.

Every entity with a `Window` automatically receives a `DisplayTarget` through
the required-component machinery, defaulting to `DisplayTarget::SDR_SRGB`
(paper white and peak of 100 nits, Rec.709 gamut, sRGB transfer). This default
matches Bevy's existing behavior exactly, and in this release the component is
inert data, so no action is required for rendering: output is unchanged.

Things you may notice:

- Queries like `Query<&DisplayTarget, With<Window>>` now match all window
  entities, and archetype-based assumptions about window entities (e.g. exact
  component sets in tests or editors) must account for the extra component.
- Window entities serialized with reflection-based scene formats now include
  the `DisplayTarget` component alongside `Window`.
- Spawning a `Window` with a custom `DisplayTarget` works like any other
  required component: insert your own value to override the default, e.g.
  `commands.spawn((Window::default(), DisplayTarget { peak_luminance_nits: 1000.0, ..Default::default() }))`.

Additionally, a new `WindowMonitorChanged` message is registered by
`WindowPlugin` and emitted by `bevy_winit` when the monitor a window is on
changes. Bevy never mutates `DisplayTarget` automatically in response — the
component is user-authoritative — so existing code is unaffected unless you
choose to listen for the new message.
