# Vectors and quaternions, and matrices for general purposes, and computer graphics. 

[![Crate](https://img.shields.io/crates/v/lin-alg2.svg)](https://crates.io/crates/lin-alg2)
[![Docs](https://docs.rs/lin-alg2/badge.svg)](https://docs.rs/lin-alg2)

Vector, matrix, and quaternion data structures and operations. Uses f32 or f64 based types.

Example use cases:

- Computer graphics
- Biomechanics
- Structural chemistry and biochemistry
- Various scientific and engineering applications
- Aircraft attitude systems and autopilots

Vector and Quaternion types are *copy*.

For Compatibility with no_std tgts, eg embedded. Use the `no_std` feature. For computer-graphics
functionality (e.g. specialty matrix constructors, and [de]serialization to byte arrays), use the `computer_graphics` feature.

Do not run `cargo fmt` on this code base; the macro used to prevent duplication of code between `f32` and f64` mules causes undesirable behavior.

For information on practical quaternion operations: [Quaternions: A practical guide](https://www.anyleaf.org/blog/quaternions:-a-practical-guide)

See the official documention (Linked above) for details. Below is a brief, impractical syntax overview:

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
    
    let quaternion = Quaternion::from_axis_angle(Vec3::new(1., 0., 0.), TAU / 16.);
}

```