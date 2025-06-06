#![macro_use]
// todo:  You can combine these with your non-SIMD ones, for the most common operations.

//! Notes.
//! Array of structs (AoS): Each vector (x,y,z,w) is stored contiguously in memory.
//! For a single 4D vector of $f, you can hold all 4 floats in one __m128.
//!
//! Structure of Arrays (SoA): Each register holds the same component (x or y or z or w) but across
//! multiple “instances.” This is often used in data-parallel algorithms (e.g. operating on 4
//! separate vectors at once).
//!
//! Note on Vec3: We just don't use the final 32 or 64 bits. The 4th lane is effectively unused,
//! and set to 0. or 1., depending on application.
//!
//! I think, using the SoA approach is more efficient, from what I read. I.e. allow each vec to
//! effectively act as 4 vecs... as opposed to placing all 4 values in the same vec field, as Glam
//! does (?)
//!
//! It appears that 128-bit wide SIMD is the most common. ("SSE"). 256-bit is called AVX, and is only
//! available on (relatively) newer CPUs. AVX-512 is less common.

use std::array::from_fn;

macro_rules! create_simd {
    ($f:ident, $fx:ident, $vec3_ty:ident, $vec4_ty:ident, $quat_ty:ident, $lanes:expr) => {
        use std::convert::TryInto;

        /// SoA. Performs operations on x Vecs.
        #[derive(Clone, Copy, Debug)]
        pub struct $vec3_ty {
            pub x: $fx,
            pub y: $fx,
            pub z: $fx,
        }

        /// SoA. Performs operations on x Vecs.
        #[derive(Clone, Copy, Debug)]
        pub struct $vec4_ty {
            pub x: $fx,
            pub y: $fx,
            pub z: $fx,
            pub w: $fx,
        }

        impl Neg for $vec3_ty {
            type Output = Self;

            fn neg(self) -> Self::Output {
                let zero = $fx::splat(0.);

                Self {
                    x: zero - self.x,
                    y: zero - self.y,
                    z: zero - self.z,
                }
            }
        }

        impl Add<$f> for $vec3_ty {
            type Output = Self;

            fn add(self, rhs: $f) -> Self::Output {
                let s = $fx::splat(rhs);
                self + s
            }
        }

        impl Sub<$f> for $vec3_ty {
            type Output = Self;

            fn sub(self, rhs: $f) -> Self::Output {
                let s = $fx::splat(rhs);
                self - s
            }
        }

        impl Mul<$f> for $vec3_ty {
            type Output = Self;

            fn mul(self, rhs: $f) -> Self::Output {
                let s = $fx::splat(rhs);
                self * s
            }
        }

        impl Div<$f> for $vec3_ty {
            type Output = Self;

            fn div(self, rhs: $f) -> Self::Output {
                let s = $fx::splat(rhs);
                self / s
            }
        }

        impl Neg for $vec4_ty {
            type Output = Self;

            fn neg(self) -> Self::Output {
                let zero = $fx::splat(0.);

                Self {
                    x: zero - self.x,
                    y: zero - self.y,
                    z: zero - self.z,
                    w: zero - self.w,
                }
            }
        }

        impl Mul<$f> for $vec4_ty {
            type Output = Self;

            fn mul(self, rhs: $f) -> Self::Output {
                self * $fx::splat(rhs)
            }
        }

        impl $vec3_ty {
            /// Create a new x-lane, SoA Vec3 from an array.
            pub fn from_array(arr: [Vec3; $lanes]) -> Self {
                let x_vals = arr.iter().map(|v| v.x).collect::<Vec<_>>();
                let y_vals = arr.iter().map(|v| v.y).collect::<Vec<_>>();
                let z_vals = arr.iter().map(|v| v.z).collect::<Vec<_>>();

                Self {
                    x: $fx::from_slice(&x_vals),
                    y: $fx::from_slice(&y_vals),
                    z: $fx::from_slice(&z_vals),
                }
            }

            pub fn from_slice(slice: &[Vec3]) -> Self {
                let x_vals = slice.iter().map(|v| v.x).collect::<Vec<_>>();
                let y_vals = slice.iter().map(|v| v.y).collect::<Vec<_>>();
                let z_vals = slice.iter().map(|v| v.z).collect::<Vec<_>>();

                Self {
                    x: $fx::from_slice(&x_vals),
                    y: $fx::from_slice(&y_vals),
                    z: $fx::from_slice(&z_vals),
                }
            }

            /// Convert the SoA data back into an array of eight Vec3s.
            pub fn to_array(self) -> [Vec3; $lanes] {
                let x_arr = self.x.to_array();
                let y_arr = self.y.to_array();
                let z_arr = self.z.to_array();

                let mut out = [Vec3 {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                }; $lanes];
                for i in 0..$lanes {
                    out[i] = Vec3 {
                        x: x_arr[i],
                        y: y_arr[i],
                        z: z_arr[i],
                    };
                }
                out
            }

            pub fn new_zero() -> Self {
                let zero = $fx::splat(0.);
                Self {
                    x: zero,
                    y: zero,
                    z: zero,
                }
            }

            pub fn splat(val: Vec3) -> Self {
                Self {
                    x: $fx::splat(val.x),
                    y: $fx::splat(val.y),
                    z: $fx::splat(val.z),
                }
            }
        }

        impl $vec4_ty {
            pub fn from_array(arr: [Vec4; $lanes]) -> Self {
                let x_vals = arr.iter().map(|v| v.x).collect::<Vec<_>>();
                let y_vals = arr.iter().map(|v| v.y).collect::<Vec<_>>();
                let z_vals = arr.iter().map(|v| v.z).collect::<Vec<_>>();
                let w_vals = arr.iter().map(|v| v.w).collect::<Vec<_>>();

                Self {
                    x: $fx::from_slice(&x_vals),
                    y: $fx::from_slice(&y_vals),
                    z: $fx::from_slice(&z_vals),
                    w: $fx::from_slice(&w_vals),
                }
            }

            pub fn from_slice(slice: &[Vec4]) -> Self {
                let x_vals = slice.iter().map(|v| v.x).collect::<Vec<_>>();
                let y_vals = slice.iter().map(|v| v.y).collect::<Vec<_>>();
                let z_vals = slice.iter().map(|v| v.z).collect::<Vec<_>>();
                let w_vals = slice.iter().map(|v| v.z).collect::<Vec<_>>();

                Self {
                    x: $fx::from_slice(&x_vals),
                    y: $fx::from_slice(&y_vals),
                    z: $fx::from_slice(&z_vals),
                    w: $fx::from_slice(&w_vals),
                }
            }

            /// Convert the SoA data back into an array of eight Vec3s.
            pub fn to_array(self) -> [Vec4; $lanes] {
                let x_arr = self.x.to_array();
                let y_arr = self.y.to_array();
                let z_arr = self.z.to_array();
                let w_arr = self.w.to_array();

                let mut out = [Vec4 {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                    w: 0.0,
                }; $lanes];
                for i in 0..$lanes {
                    out[i] = Vec4 {
                        x: x_arr[i],
                        y: y_arr[i],
                        z: z_arr[i],
                        w: w_arr[i],
                    };
                }
                out
            }

            pub fn new_zero() -> Self {
                let zero = $fx::splat(0.);
                Self {
                    x: zero,
                    y: zero,
                    z: zero,
                    w: zero,
                }
            }

            pub fn splat(val: Vec4) -> Self {
                Self {
                    x: $fx::splat(val.x),
                    y: $fx::splat(val.y),
                    z: $fx::splat(val.z),
                    w: $fx::splat(val.w),
                }
            }
        }

        /// SoA. Performs operations on x quaternions
        #[derive(Clone, Copy, Debug)]
        pub struct $quat_ty {
            pub w: $fx,
            pub x: $fx,
            pub y: $fx,
            pub z: $fx,
        }

        impl Mul<$f> for $quat_ty {
            type Output = Self;

            fn mul(self, rhs: $f) -> Self::Output {
                let s = $fx::splat(rhs);
                Self {
                    w: self.w * s,
                    x: self.x * s,
                    y: self.y * s,
                    z: self.z * s,
                }
            }
        }

        impl $quat_ty {
            /// Create a new x-lane, SoA, $f Vec3
            pub fn from_array(slots: [Quaternion; $lanes]) -> Self {
                let mut w_arr = [0.; $lanes];
                let mut x_arr = [0.; $lanes];
                let mut y_arr = [0.; $lanes];
                let mut z_arr = [0.; $lanes];

                for i in 0..$lanes {
                    w_arr[i] = slots[i].w;
                    x_arr[i] = slots[i].x;
                    y_arr[i] = slots[i].y;
                    z_arr[i] = slots[i].z;
                }

                Self {
                    w: $fx::from_slice(&w_arr),
                    x: $fx::from_slice(&x_arr),
                    y: $fx::from_slice(&y_arr),
                    z: $fx::from_slice(&z_arr),
                }
            }

            pub fn from_slice(slice: &[Quaternion]) -> Self {
                let mut w_arr = [0.; $lanes];
                let mut x_arr = [0.; $lanes];
                let mut y_arr = [0.; $lanes];
                let mut z_arr = [0.; $lanes];

                for i in 0..$lanes {
                    w_arr[i] = slice[i].w;
                    x_arr[i] = slice[i].x;
                    y_arr[i] = slice[i].y;
                    z_arr[i] = slice[i].z;
                }

                Self {
                    w: $fx::from_slice(&w_arr),
                    x: $fx::from_slice(&x_arr),
                    y: $fx::from_slice(&y_arr),
                    z: $fx::from_slice(&z_arr),
                }
            }

            /// Convert the SoA data back into an array of eight Quaternions.
            pub fn to_array(self) -> [Quaternion; $lanes] {
                let w_arr = self.w.to_array();
                let x_arr = self.x.to_array();
                let y_arr = self.y.to_array();
                let z_arr = self.z.to_array();

                // Reconstruct each Vec3 from corresponding lanes
                let mut out = [Quaternion {
                    w: 0.0,
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                }; $lanes];
                for i in 0..$lanes {
                    out[i] = Quaternion {
                        w: w_arr[i],
                        x: x_arr[i],
                        y: y_arr[i],
                        z: z_arr[i],
                    };
                }
                out
            }

            pub fn new_identity() -> Self {
                Self {
                    w: $fx::splat(1.),
                    x: $fx::splat(0.),
                    y: $fx::splat(0.),
                    z: $fx::splat(0.),
                }
            }

            pub fn splat(val: Quaternion) -> Self {
                Self {
                    w: $fx::splat(val.w),
                    x: $fx::splat(val.x),
                    y: $fx::splat(val.y),
                    z: $fx::splat(val.z),
                }
            }

            /// Converts the SIMD quaternion to a SIMD 3D vector, discarding `w`.
            pub fn to_vec(self) -> $vec3_ty {
                $vec3_ty {
                    x: self.x,
                    y: self.y,
                    z: self.z,
                }
            }
        }

        /// Used for creating a set of `Vec3x` from one of `Vec3`. Padded as required. The result
        /// will have approximately x fewer elements than the input.
        ///
        /// Important: When performing operations, make sure to discard data from *garbage* lanes
        /// in the remainder of your last packed value.
        ///
        /// Returns (packed_values, lanes valid in last chunk)
        pub fn pack_vec3(vecs: &[Vec3]) -> (Vec<$vec3_ty>, usize) {
            let remainder = vecs.len() % $lanes;
            let padding_needed = if remainder == 0 {
                0
            } else {
                $lanes - remainder
            };

            let mut padded = Vec::with_capacity(vecs.len() + padding_needed);
            padded.extend_from_slice(vecs);
            padded.extend((0..padding_needed).map(|_| Vec3::new_zero()));

            // Now `padded.len()` is a multiple of x, so chunks_exact(x) will consume it fully.
            let data = padded
                .chunks_exact($lanes)
                .map(|chunk| {
                    // Convert the slice chunk into an array of x Vec3 elements.
                    let arr: [Vec3; $lanes] = chunk.try_into().unwrap();
                    $vec3_ty::from_array(arr)
                })
                .collect();

            let valid_lanes_last_chunk = if remainder == 0 { $lanes } else { remainder };

            (data, valid_lanes_last_chunk)
        }

        /// Convert a slice of SIMD Vec3x values, x-wide to Vec3. The result
        /// will have approximately x as many elements as the input. Its parameters include
        /// the number of original values, so it knows to only use valid lanes on the last
        /// chunk.
        pub fn unpack_vec3(vals: &[$vec3_ty], len_orig: usize) -> Vec<Vec3> {
            let mut result = Vec::with_capacity(len_orig);

            for (i, val) in vals.iter().enumerate() {
                let lanes = if i == vals.len() - 1 {
                    let rem = len_orig % $lanes;
                    if rem == 0 { $lanes } else { rem }
                } else {
                    $lanes
                };

                result.extend(&val.to_array()[..lanes]);
            }
            result
        }

        /// Used for creating a set of `Quaternionx` from one of `Quatenrion`. Padded as required. The result
        /// will have approximately x fewer elements than the input.
        ///
        /// Important: When performing operations, make sure to discard data from *garbage* lanes
        /// in the remainder of your last packed value.
        ///
        /// Returns (packed_values, lanes valid in last chunk)
        pub fn pack_quaternion(vals: &[Quaternion]) -> (Vec<$quat_ty>, usize) {
            let remainder = vals.len() % $lanes;
            let padding_needed = if remainder == 0 {
                0
            } else {
                $lanes - remainder
            };

            let mut padded = Vec::with_capacity(vals.len() + padding_needed);
            padded.extend_from_slice(vals);
            padded.extend((0..padding_needed).map(|_| Quaternion::new_identity()));

            // Now `padded.len()` is a multiple of x, so chunks_exact(x) will consume it fully.
            let data = padded
                .chunks_exact($lanes)
                .map(|chunk| {
                    // Convert the slice chunk into an array of x Vec3 elements.
                    let arr: [Quaternion; $lanes] = chunk.try_into().unwrap();
                    $quat_ty::from_array(arr)
                })
                .collect();

            let valid_lanes_last_chunk = if remainder == 0 { $lanes } else { remainder };

            (data, valid_lanes_last_chunk)
        }

        /// Convert a slice of SIMD Quaternionx values, x-wide to Quaternion. The result
        /// will have approximately x as many elements as the input. Its parameters include
        /// the number of original values, so it knows to only use valid lanes on the last
        /// chunk.
        pub fn unpack_quaternion(vals: &[$quat_ty], len_orig: usize) -> Vec<Quaternion> {
            let mut result = Vec::with_capacity(len_orig);

            for (i, val) in vals.iter().enumerate() {
                let lanes = if i == vals.len() - 1 {
                    let rem = len_orig % $lanes;
                    if rem == 0 { $lanes } else { rem }
                } else {
                    $lanes
                };

                result.extend(&val.to_array()[..lanes]);
            }
            result
        }

        // todo: Use the paste lib etc to put `$f` in the fn names here.
        /// Convert a slice of `$f` to an array of SIMD `$f` values, x-wide. Padded as required. The result
        /// will have approximately 8x fewer elements than the input.
        ///
        /// Important: When performing operations, make sure to discard data from *garbage* lanes
        /// in the remainder of your last packed value.
        ///
        /// Returns (packed_values, lanes valid in last chunk)
        pub fn pack_float(vals: &[$f]) -> (Vec<$fx>, usize) {
            let remainder = vals.len() % $lanes;
            let padding_needed = if remainder == 0 {
                0
            } else {
                $lanes - remainder
            };

            let mut padded = Vec::with_capacity(vals.len() + padding_needed);
            padded.extend_from_slice(vals);
            padded.extend((0..padding_needed).map(|_| 0.));

            // Now `padded.len()` is a multiple of x, so chunks_exact(x) will consume it fully.
            let data = padded
                .chunks_exact($lanes)
                .map(|chunk| $fx::load(chunk.as_ptr()))
                .collect();

            let valid_lanes_last_chunk = if remainder == 0 { $lanes } else { remainder };

            (data, valid_lanes_last_chunk)
        }

        /// Convert a slice of SIMD floating point values to plain floating point ones. The result
        /// will have approximately x as many elements as the input. Its parameters include
        /// the number of original values, so it knows to only use valid lanes on the last
        /// chunk.
        pub fn unpack_float(vals: &[$fx], len_orig: usize) -> Vec<$f> {
            let mut result = Vec::with_capacity(len_orig);

            for (i, val) in vals.iter().enumerate() {
                let lanes = if i == vals.len() - 1 {
                    let rem = len_orig % $lanes;
                    if rem == 0 { $lanes } else { rem }
                } else {
                    $lanes
                };

                result.extend(&val.to_array()[..lanes]);
            }
            result
        }
    };
}

/// Convert a slice of any type to an array values, for use with SIMD. Padded as required. The result
/// will have approximately 8x fewer elements than the input.
///
/// Important: When performing operations, make sure to discard data from *garbage* lanes
/// in the remainder of your last packed value.
///
/// Returns (packed_values, lanes valid in last chunk)
pub fn pack_slice<T, const LANES: usize>(vals: &[T]) -> (Vec<[T; LANES]>, usize)
where
    T: Copy + Clone + Default,
{
    let remainder = vals.len() % LANES;
    let padding_needed = if remainder == 0 { 0 } else { LANES - remainder };

    let mut padded = Vec::with_capacity(vals.len() + padding_needed);
    padded.extend_from_slice(vals);
    padded.extend((0..padding_needed).map(|_| T::default()));

    let data = padded
        .chunks_exact(LANES)
        .map(|chunk| {
            let mut arr = [T::default(); LANES];
            arr.clone_from_slice(chunk);
            arr
        })
        .collect();

    let valid_lanes_last_chunk = if remainder == 0 { LANES } else { remainder };

    (data, valid_lanes_last_chunk)
}

// todo: DRY
pub fn pack_slice_noncopy<T, const LANES: usize>(vals: &[T]) -> (Vec<[T; LANES]>, usize)
where
    T: Clone + Default,
{
    let remainder = vals.len() % LANES;
    let valid_lanes_last = if remainder == 0 { LANES } else { remainder };

    let padding_needed = if remainder == 0 { 0 } else { LANES - remainder };

    let mut padded: Vec<T> = vals.to_vec(); // clone all T
    padded.reserve(padding_needed);
    padded.extend((0..padding_needed).map(|_| T::default()));

    let data: Vec<[T; LANES]> = padded
        .chunks_exact(LANES)
        .map(|chunk| {
            // for each index 0..LANES, clone chunk[i] into the array
            from_fn(|i| chunk[i].clone())
        })
        .collect();

    (data, valid_lanes_last)
}

pub fn unpack_slice<T: Copy, const LANES: usize>(vals: &[[T; LANES]], len_orig: usize) -> Vec<T> {
    let mut result = Vec::with_capacity(len_orig);

    for (i, chunk) in vals.iter().enumerate() {
        let lanes = if i == vals.len() - 1 {
            let rem = len_orig % LANES;
            if rem == 0 { LANES } else { rem }
        } else {
            LANES
        };

        result.extend(chunk[..lanes].iter().copied());
    }
    result
}

// todo: Unpack_slice_noncopy too
