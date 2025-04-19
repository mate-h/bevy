# Bevy Atmosphere and Clouds System

This module provides a physically-based atmospheric scattering renderer for Bevy Engine, based on [Hillaire's 2020 paper](https://sebh.github.io/publications/egsr2020.pdf).

## Features

- Real-time atmospheric scattering with accurate light transport
- Dynamic time-of-day lighting that affects the atmosphere appearance
- Multiple directional lights with proper physical representation
- Volumetric clouds with realistic lighting
- Seamless integration with existing skyboxes

## Usage

### Basic Atmosphere Setup

Add the `Atmosphere` and `AtmosphereSettings` components to your main camera:

```rust
use bevy::prelude::*;
use bevy_pbr::atmosphere::{Atmosphere, AtmosphereSettings};

fn setup(mut commands: Commands) {
    // Spawn camera with atmosphere
    commands.spawn((
        Camera3dBundle::default(),
        Atmosphere::default(), // Uses Earth-like atmosphere parameters
        AtmosphereSettings::default(),
    ));

    // Add a sun light
    commands.spawn(DirectionalLightBundle {
        directional_light: DirectionalLight {
            illuminance: 100000.0,
            ..default()
        },
        transform: Transform::from_rotation(Quat::from_rotation_x(-std::f32::consts::FRAC_PI_4)),
        ..default()
    });
}
```

### Adding Clouds

To add clouds to your atmosphere, add the `Clouds` component to your camera:

```rust
use bevy::prelude::*;
use bevy_pbr::atmosphere::{Atmosphere, AtmosphereSettings, Clouds};

fn setup(mut commands: Commands) {
    // Spawn camera with atmosphere and clouds
    commands.spawn((
        Camera3dBundle::default(),
        Atmosphere::default(),
        AtmosphereSettings::default(),
        Clouds {
            coverage: 0.5,         // 0.0 = no clouds, 1.0 = fully overcast
            altitude: 2000.0,      // base altitude in meters
            thickness: 4000.0,     // cloud layer thickness in meters
            density: 0.05,         // controls overall cloud density
            
            // Control cloud appearance
            shape_scale: Vec3::new(0.0001, 0.0002, 0.0001),
            detail_scale: Vec3::new(0.001, 0.001, 0.001),
            detail_strength: 0.3,
            
            // Cloud movement
            wind_speed: Vec2::new(4.0, 2.0),
            
            // Controls
            brightness: 1.0,
            ray_march_steps: 32,   // increase for better quality
            light_samples: 4,      // increase for better lighting quality
            ..default()
        },
    ));

    // Add a sun light
    commands.spawn(DirectionalLightBundle {
        directional_light: DirectionalLight {
            illuminance: 100000.0,
            ..default()
        },
        transform: Transform::from_rotation(Quat::from_rotation_x(-std::f32::consts::FRAC_PI_4)),
        ..default()
    });
}
```

### Time of Day Animation

You can animate the sun direction to simulate time of day changes:

```rust
use bevy::prelude::*;

fn update_sun(time: Res<Time>, mut query: Query<&mut Transform, With<DirectionalLight>>) {
    for mut transform in &mut query {
        // Rotate around the Y axis for day-night cycle
        let angle = time.elapsed_seconds() * 0.1;
        let y = angle.sin() * 0.7;
        transform.rotation = Quat::from_rotation_x(-1.0)
            * Quat::from_rotation_y(angle);
    }
}
```

## Performance Tuning

The atmosphere system uses several Look-Up Tables (LUTs) to efficiently compute atmospheric scattering. You can adjust their size and sample counts for better performance or quality:

```rust
AtmosphereSettings {
    // Higher values = better quality, lower performance
    sky_view_lut_size: UVec2::new(400, 200),
    sky_view_lut_samples: 16,
    
    // Cloud specific
    clouds.ray_march_steps: 16,  // Lower for better performance
    clouds.light_samples: 2,     // Lower for better performance
    ..default()
}
```

## Technical Details

The atmosphere system simulates several physical phenomena:

1. Rayleigh scattering (from air molecules)
2. Mie scattering (from larger particles like dust)
3. Ozone absorption
4. Multiple scattering effects
5. Volumetric cloud rendering

For volumetric clouds, a ray-marching approach is used with procedural noise functions to create realistic cloud formations that respond to lighting conditions. 