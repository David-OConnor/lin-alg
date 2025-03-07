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
feature. For [bincode](https://docs.rs/bincode/latest/bincode/) binary encoding and decoding, use the `encode` feature.

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

If using for computer graphics, this functionality may be helpful:

```rust
    let a = Vec3::new(1., 1., 1.);
    let bytes = a.to_bytes(); // Send this to the GPU. `Quaternion` and `Matrix` have similar methods.

    let model_mat = Mat4::new_translation(self.position)
        * self.orientation.to_matrix()
        * Mat4::new_scaler_partial(self.scale);

    let proj_mat = Mat4::new_perspective_lh(self.fov_y, self.aspect, self.near, self.far);

    let view_mat = self.orientation.inverse().to_matrix() * Mat4::new_translation(-self.position);

    // Example of rolling a camera around the forward axis:
    let fwd = orientation.rotate_vec(FWD_VEC);
    let rotation = Quaternion::from_axis_angle(fwd, -rotate_key_amt);
    orientation = rotation * orientation;
```

A practical geometry example:

```rust
/// Calculate the dihedral angle between 4 positions (3 bonds).
/// The `bonds` are one atom's position, substracted from the next. Order matters.
pub fn calc_dihedral_angle(bond_middle: Vec3, bond_adjacent1: Vec3, bond_adjacent2: Vec3) -> f64 {
    // Project the next and previous bonds onto the plane that has this bond as its normal.
    // Re-normalize after projecting.
    let bond1_on_plane = bond_adjacent1.project_to_plane(bond_middle).to_normalized();
    let bond2_on_plane = bond_adjacent2.project_to_plane(bond_middle).to_normalized();

    // Not sure why we need to offset by 𝜏/2 here, but it seems to be the case
    let result = bond1_on_plane.dot(bond2_on_plane).acos() + TAU / 2.;

    // The dot product approach to angles between vectors only covers half of possible
    // rotations; use a determinant of the 3 vectors as matrix columns to determine if what we
    // need to modify is on the second half.
    let det = det_from_cols(bond1_on_plane, bond2_on_plane, bond_middle);

    if det < 0. { result } else { TAU - result }
}
```

