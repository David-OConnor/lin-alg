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

Includes SIMD constructs (SoA layout) for Vec and Quaternion types. For example: `Vec3x8`, `Vec3x4``Vec4x8`, and `Quaternionx8` etc,
for `f32` and `f64` types. They are configured with 256-bit wide (AVX) values, performing (for vectors) operations on 8 `f32` `Vec3`,
4 `f64` `Vec3`, etc. See the examples below for details.

This library exposes an `f32x8` SIMD type that wraps `__m256` with appropriate constructors, operator overloads etc, and similar. This,
and the `Vec3x8` etc APIs, mimic the nightly [core::simd](https://doc.rust-lang.org/beta/core/simd/index.html) library. They're used internally by our SIMD vector and quaternion types.
It also includes `f64x4`. We are waiting to add 512-bit wide types until their operations are in stable rust. Hopefully soon!

We take this approach so this library will work on stable rust. We'll remove these when `core::simd` is stable.


## CUDA (GPU)
This library includes two helper functions for use with the `cudarc` library; these are to allocated `Vec3` and `Quaternion`
types. (f32 and f64). They perform host-to-device copies.
```rust
pub fn alloc_vec3s(dev: &Arc<CudaDevice>, data: &[Vec3]) -> CudaSlice<f32> {}

pub fn alloc_quaternions(dev: &Arc<CudaDevice>, data: &[Quaternion]) -> CudaSlice<f32> {}
```


## A note on performance
For performance-sensitive operations, depending on the details of your computation and hardware, you may wish to 
use a mix of GPU (CUDA, or graphics shaders), parallelization via threads (e.g. Rayon), and SIMD operations. This 
library aims to assist in these operations, and leaves details to the application.


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

A SIMD example of vector operations:
```rust
use lin_alg::f32:{Vec3, Vec3x8};

// Non-SIMD Vec3s we'll start with.
let vec_a = Vec3::new(1., 2., 3.);
let vec_b = Vec3::new(4., 5., 6.);

// An example where we copy the same Vec3 into all 8 slots. In most practical uses,
// each slot will contain a different value.
let a = Vec3x8::from_array([vec_a; 8]);
let b = Vec3x8::from_array([vec_b; 8]);

// Perform vector addition on 8 Vec3s at once.
let c = a + b;

// Create a [Vec3; 8].
let d = a.cross(b).to_array();

// Create a `f32x8`, then convert to an array.
let dot_result = a.dot(b).to_array();

let e = vec_a * 3.;
let f = vec_a * f32x8::from_array([3.; 8]);
let g = vec_a * f32x8::splat(3.)
```

A SIMD example of rotating vectors.
```rust
use core::f32::consts::TAU;
use lin_alg::f32::{Quaternion, Vec3, Quaternionx8, Vec3x8};

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

let rotation = Quaternionx8::from_array(rot_init);

// This could be 8 separate values.
let vec = Vec3x8::from_array([UP; 8]);

let result = rotation.rotate_vec(vec).to_array();

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

An example function using SIMD for a practical use, integrating `Vec3x8s` with SIMD types directly.

```rust
use lin_alg::f32::{Vec3, Vec3x8, f32x8};

// ...

fn run_lj(atom_0_posits: &[Vec3], atom_1_posits: &[Vec3]) {
    // Convert all Vec3s to their SIMD variants, and loop through them. This converts then to 
    // `Vec<Vec3x8>`
    let atom_0_posits_simd = pack_vec3(&atom_0_posits);
    let atom_1_posits_simd = pack_vec3(&atom_1_posits);

    // We also provide a `pack_f32`, `unpack_f32`, `unpack_vec3` functions to convert between SIMD and native types.
    
    // todo: Or, parellilize with Rayon.
    for i in 0..atom_0_posits_simd {
        let atom_1 = atom_0_posits_simd[i];       
        let atom_0 = atom_1_posits_simd[i];       
        
        lj_potential(atom_0_posit, atom_1_posit, // ...);)
    }
// ...


fn lj_potential(
    atom_0_posit: Vec3x8,
    atom_1_posit: Vec3x8,
    atom_0_els: [Element; 8],
    atom_1_els: [Element; 8],
) -> f32x8 {
    // This line demonstrates use of this library; the rest of the code below
    // is for context. We have already partitioned a set of `Vec3` into 
    // `Vec3x8`, grouped in blocks of 8, prior to this function.
    let r = (atom_0_posit - atom_1_posit).magnitude(); // This is a Vec3x8.

    let mut sig = [0.0; 8];
    let mut eps = [0.0; 8];
    for i in 0..8 {
        (sig[i], eps[i]) = get_lj_params(atom_0_els[i], atom_1_els[i], lj_lut)
    }

    let sig_ = f32x8::from_slice(&sig);
    let eps_ = f32x8::from_array(eps);

    let sr = sig_ / r;
    let sr6 = sr.powi(6);
    let sr12 = sr6.powi(2);
    
    f32x8::splat(4.) * eps_ * (sr12 - sr6)
}
```
