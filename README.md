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

For Compatibility with no_std targets, e.g. embedded, disable default features, and enable the `no_std` feature. This  omits
`std::fmt::Display` implementations, and enables [num_traits](https://docs.rs/num-traits/latest/num_traits/)'s `libm` capabilities
for certain operations. `lin_alg = { version = "^1.1.0", default-features = false, features = ["no_std"] }`

For computer-graphics functionality (e.g. specialty matrix constructors, and [de]serialization to byte arrays for passing to and from GPUs), use the `computer_graphics` 
feature. For [bincode](https://docs.rs/bincode/latest/bincode/) binary encoding and decoding, use the `encode` feature.

For information on practical quaternion operations: [Quaternions: A practical guide](https://www.anyleaf.org/blog/quaternions:-a-practical-guide).

The `From` trait is implemented for most types, for converting between `f32` and `f64` variants using the `into()` syntax.


## SIMD

Includes WIP SIMD constructs (SoA layout): The `Vec3S`,`Vec4S`, and `QuaternionS` types. They are configured with 256-bit
wide (AVX) values, performing (for vectors) operations on 8 `f32` Vec3 or Vec4s, or 4 `f64` ones. See the example below for details.
not all functionality is implemented, and only `f32` variants are implemented at this time.

Various operator overloads are implemented. For example, you can (scalar) multiply a `Vec3S` by a `f32`, a `[f32; 8]`, or
a `__m256`. This applies to quaternion operations, like multiplication, as well.


## Examples

See the official documentation (Linked above) for details. Below is a brief, impractical syntax overview:

```rust
use core::f32::consts::TAU;

use lin_alg::f32::{Vec3, Quaternion};

fn main() {
    let _ = Vec3::new_zero();
    
    let a = Vec3::new(1., 1., 1.);
    let b = Vec3::new(0., -1., 10.);
    
    let mut c = a + b;
    
    let d = a.dot(b);
    
    c.normalize(); // or:
    let e = c.to_normalized();
    
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
    let bytes = a.to_bytes(); // Send this to the GPU. `Quaternion` and `MatN` have similar methods.

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

A SIMD example over vector operations:
```rust
use lin_alg::f32:{Vec3, Vec3S};

// Non-SIMD Vec3s we'll start with.
let vec_a = Vec3::new(1., 2., 3.);
let vec_b = Vec3::new(4., 5., 6.);

// An example where we copy the same Vec3 into all 8 slots. In most practical uses,
// each slot will contain a different value.
let a = Vec3S::new([vec_a; 8]);
let b = Vec3S::new([vec_b; 8]);

// Perform vector addition on 8 Vec3s at once.
let c = a + b;

// Create a [Vec3; 8], due to the `unpack` method.
let d = a.cross(b).unpack();

// Create a `__m256`, then convert to an array.
let dot_result: [f32; 8] = unsafe { transmute(a.dot(b)) };

// Create a [f32; 8].
let dot_result = a.dot_unpack(b);

let e = vec_a * 3.;
let f = vec_a * [3.; 8];
let g = vec_a * _mm256_set1_ps(3.);
```

A SIMD example of rotating vectors.
```rust
use core::f32::consts::TAU;
use lin_alg::f32::{Quaternion, Vec3, QuaternionS, Vec3S};

let rot_init = [
    Quaternion::from_unit_vecs(UP, FORWARD),
    Quaternion::from_unit_vecs(UP, -FORWARD),
    Quaternion::from_unit_vecs(UP, RIGHT),
    Quaternion::from_unit_vecs(UP, -RIGHT),
    Quaternion::from_unit_vecs(UP, UP),
    Quaternion::from_unit_vecs(UP, -UP),
    Quaternion::from_axis_angle(RIGHT, TAU/4.),
    Quaternion::from_axis_angle(RIGHT, TAU/8.),
];

let rotation = QuaternionS::new(rot_init);

// This could be 8 separate values.
let vec = Vec3S::new([UP; 8]);

let result = rotation.rotate_vec(vec).unpack();

let sqrt_2_div_2 = 2_f32.sqrt()/2.;
let angled = Vec3::new(0., -sqrt_2_div_2, sqrt_2_div_2);

assert!((result[0] - FORWARD).magnitude() < f32::EPSILON);
assert!((result[1] - -FORWARD).magnitude() < f32::EPSILON);
assert!((result[2] - RIGHT).magnitude() < f32::EPSILON);
assert!((result[3] - -RIGHT).magnitude() < f32::EPSILON);
assert!((result[4] - UP).magnitude() < f32::EPSILON);
assert!((result[5] - -UP).magnitude() < f32::EPSILON);
assert!((result[6] - -FORWARD).magnitude() < f32::EPSILON);
assert!((result[7] - angled).magnitude() < f32::EPSILON);

```

