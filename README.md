# Vectors, quaternions, and matrices for general purposes, and computer graphics. 

[![Crate](https://img.shields.io/crates/v/lin_alg.svg)](https://crates.io/crates/lin_alg)
[![Docs](https://docs.rs/lin_alg/badge.svg)](https://docs.rs/lin_alg)

Vector, matrix, and quaternion data structures and operations. Uses f32 or f64 based types.

Example use cases:

- Computer graphics
- Biomechanics
- Robotics and unmanned aerial vehicles.
- Structural chemistry and biochemistry
- Cosmology modeling
- Various scientific and engineering applications
- Aircraft attitude systems and autopilots

Vector and Quaternion types are *copy*.

For Compatibility with no_std tgts, e.g. embedded, Use the `no_std` feature. This feature omits `std::fmt::Display` implementations. For computer-graphics
functionality (e.g. specialty matrix constructors, and [de]serialization to byte arrays for passing to and from GPUs), use the `computer_graphics` 
feature. For[bincode](https://docs.rs/bincode/latest/bincode/) binary encoding and decoding, use the `encode` feature.

Do not run `cargo fmt` on this code base; the macro used to prevent duplication of code between `f32` and `f64` modules causes 
undesirable behavior.

For information on practical quaternion operations: [Quaternions: A practical guide](https://www.anyleaf.org/blog/quaternions:-a-practical-guide).

The `From` trait is implemented for most types, for converting between `f32` and `f64` variants using the `into()` syntax.

See the official documentation (Linked above) for details. Below is a brief, impractical syntax overview:

```rust
use core::f32::consts::TAU;

use lin_alg::f32::{Vec3, Quaternion};

fn main() {
    let _ = Vec3::new_zero();
    
    let a = Vec3::new(1., 1., 1.);
    let b = Vec3::new(0., -1., 10.);
    
    let c = a + b;
    
    let mut d = a.dot(b);
    
    d.normalize();
    let e = c.to_normalized();
    
    a.magnitude();
    
    let f = a.cross(b);
    
    let g = Quaternion::from_unit_vecs(d, e);
    
    let h = g.inverse();
    
    let k = Quaternion::new_identity();
    
    let l = k.rotate_vec(c);
    
    l.magnitude();
    
    let m = Quaternion::from_axis_angle(Vec3::new(1., 0., 0.), TAU / 16.);
}

```