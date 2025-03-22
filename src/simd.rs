#![macro_use]

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

macro_rules! create_simd {
    ($f:ident, $fx:ident, $vec3_ty:ident, $vec4_ty:ident, $quat_ty:ident, $lanes:expr) => {
        use std::{convert::TryInto, mem::transmute};

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

        impl Default for $vec3_ty {
            fn default() -> Self {
                Self::new_zero()
            }
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

        impl Add for $vec3_ty {
            type Output = Self;

            fn add(self, rhs: Self) -> Self::Output {
                Self {
                    x: self.x + rhs.x,
                    y: self.y + rhs.y,
                    z: self.z + rhs.z,
                }
            }
        }

        impl AddAssign for $vec3_ty {
            fn add_assign(&mut self, rhs: Self) {
                self.x += rhs.x;
                self.y += rhs.y;
                self.z += rhs.z;
            }
        }

        impl Add<$f> for $vec3_ty {
            type Output = Self;

            fn add(self, rhs: $f) -> Self::Output {
                let s = $fx::splat(rhs);
                self + s
            }
        }

        impl Add<$fx> for $vec3_ty {
            type Output = Self;

            fn add(self, rhs: $fx) -> Self::Output {
                Self {
                    x: self.x + rhs,
                    y: self.y + rhs,
                    z: self.z + rhs,
                }
            }
        }

        impl AddAssign<$fx> for $vec3_ty {
            fn add_assign(&mut self, rhs: $fx) {
                self.x += rhs;
                self.y += rhs;
                self.z += rhs;
            }
        }

        impl Sub for $vec3_ty {
            type Output = Self;

            fn sub(self, rhs: Self) -> Self::Output {
                Self {
                    x: self.x - rhs.x,
                    y: self.y - rhs.y,
                    z: self.z - rhs.z,
                }
            }
        }

        impl SubAssign for $vec3_ty {
            fn sub_assign(&mut self, rhs: Self) {
                self.x -= rhs.x;
                self.y -= rhs.y;
                self.z -= rhs.z;
            }
        }

        impl Sub<$f> for $vec3_ty {
            type Output = Self;

            fn sub(self, rhs: $f) -> Self::Output {
                let s = $fx::splat(rhs);
                self - s
            }
        }

        impl Sub<$fx> for $vec3_ty {
            type Output = Self;

            fn sub(self, rhs: $fx) -> Self::Output {
                Self {
                    x: self.x - rhs,
                    y: self.y - rhs,
                    z: self.z - rhs,
                }
            }
        }

        impl SubAssign<$fx> for $vec3_ty {
            fn sub_assign(&mut self, rhs: $fx) {
                self.x -= rhs;
                self.y -= rhs;
                self.z -= rhs;
            }
        }

        impl Mul<$f> for $vec3_ty {
            type Output = Self;

            fn mul(self, rhs: $f) -> Self::Output {
                let s = $fx::splat(rhs);
                self * s
            }
        }

        impl Mul<$fx> for $vec3_ty {
            type Output = Self;

            fn mul(self, rhs: $fx) -> Self::Output {
                Self {
                    x: self.x * rhs,
                    y: self.y * rhs,
                    z: self.z * rhs,
                }
            }
        }

        impl MulAssign<$fx> for $vec3_ty {
            fn mul_assign(&mut self, rhs: $fx) {
                self.x = self.x * rhs;
                self.y = self.y * rhs;
                self.z = self.z * rhs;
            }
        }

        impl Div<$f> for $vec3_ty {
            type Output = Self;

            fn div(self, rhs: $f) -> Self::Output {
                let s = $fx::splat(rhs);
                self / s
            }
        }

        impl Div<$fx> for $vec3_ty {
            type Output = Self;

            fn div(self, rhs: $fx) -> Self::Output {
                Self {
                    x: self.x / rhs,
                    y: self.y / rhs,
                    z: self.z / rhs,
                }
            }
        }

        impl DivAssign<$fx> for $vec3_ty {
            fn div_assign(&mut self, rhs: $fx) {
                self.x = self.x / rhs;
                self.y = self.y / rhs;
                self.z = self.z / rhs;
            }
        }

        impl Add for $vec4_ty {
            type Output = Self;

            fn add(self, rhs: Self) -> Self::Output {
                Self {
                    x: self.x + rhs.x,
                    y: self.y + rhs.y,
                    z: self.z + rhs.z,
                    w: self.w + rhs.w,
                }
            }
        }

        impl Sub for $vec4_ty {
            type Output = Self;

            fn sub(self, rhs: Self) -> Self::Output {
                Self {
                    x: self.x - rhs.x,
                    y: self.y - rhs.y,
                    z: self.z - rhs.z,
                    w: self.w - rhs.w,
                }
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
                let s = $fx::splat(rhs);

                Self {
                    x: self.x * s,
                    y: self.y * s,
                    z: self.z * s,
                    w: self.w * s,
                }
            }
        }

        impl $vec3_ty {
            /// Create a new x-lane, SoA, $f Vec3
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

            /// Dot product across x, y, z lanes. Each lane result is x_i*x_j + y_i*y_j + z_i*z_j.
            /// Returns an $fx of x dot products (one for each lane).
            pub fn dot(self, rhs: Self) -> $fx {
                self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
            }

            /// Hadamard product (lane-wise multiplication of x, y, z).
            pub fn hadamard_product(self, rhs: Self) -> Self {
                Self {
                    x: self.x * rhs.x,
                    y: self.y * rhs.y,
                    z: self.z * rhs.z,
                }
            }

            /// Returns the vector magnitude squared
            pub fn magnitude_squared(self) -> $fx {
                self.x.powi(2) + self.y.powi(2) + self.z.powi(2)
            }

            /// Returns the vector magnitude
            pub fn magnitude(&self) -> $fx {
                (self.x.powi(2) + self.y.powi(2) + self.z.powi(2)).sqrt()
            }

            pub fn normalize(&mut self) {
                let len = (self.x.powi(2) + self.y.powi(2) + self.z.powi(2)).sqrt();

                self.x /= len;
                self.y /= len;
                self.z /= len;
            }

            /// Returns the normalised version of the vector
            pub fn to_normalized(self) -> Self {
                let mag_recip = $fx::splat(1.) / self.magnitude();
                self * mag_recip
            }

            /// Lane-wise cross product
            /// cross.x = self.y * rhs.z - self.z * rhs.y
            /// cross.y = self.z * rhs.x - self.x * rhs.z
            /// cross.z = self.x * rhs.y - self.y * rhs.x
            pub fn cross(&self, rhs: Self) -> Self {
                Self {
                    x: self.y * rhs.z - self.z * rhs.y,
                    y: self.z * rhs.x - self.x * rhs.z,
                    z: self.x * rhs.y - self.y * rhs.x,
                }
            }

            /// Project a vector onto a plane defined by its normal vector. Assumes self and `plane_norm`
            /// are unit vectors.
            pub fn project_to_plane(self, plane_norm: Self) -> Self {
                self - plane_norm * self.dot(plane_norm)
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

            pub fn splat(val: Vec4) -> Self {
                Self {
                    x: $fx::splat(val.x),
                    y: $fx::splat(val.y),
                    z: $fx::splat(val.z),
                    w: $fx::splat(val.w),
                }
            }

            /// Dot product across x, y, z lanes. Each lane result is x_i*x_j + y_i*y_j + z_i*z_j.
            /// Returns an fx of x dot products (one for each lane).
            pub fn dot(self, rhs: Self) -> $fx {
                self.x * rhs.x + self.y * rhs.y + self.z * rhs.z + self.w * rhs.w
            }
        }

        /// SoA.
        #[derive(Clone, Copy, Debug)]
        pub struct $quat_ty {
            pub w: $fx,
            pub x: $fx,
            pub y: $fx,
            pub z: $fx,
        }

        impl Default for $quat_ty {
            fn default() -> Self {
                Self::new_identity()
            }
        }

        impl Add<Self> for $quat_ty {
            type Output = Self;

            fn add(self, rhs: Self) -> Self::Output {
                Self {
                    w: self.w + rhs.w,
                    x: self.x + rhs.x,
                    y: self.y + rhs.y,
                    z: self.z + rhs.z,
                }
            }
        }

        impl Sub<Self> for $quat_ty {
            type Output = Self;

            fn sub(self, rhs: Self) -> Self::Output {
                Self {
                    w: self.w - rhs.w,
                    x: self.x - rhs.x,
                    y: self.y - rhs.y,
                    z: self.z - rhs.z,
                }
            }
        }

        impl Mul for $quat_ty {
            type Output = Self;

            fn mul(self, rhs: Self) -> Self::Output {
                Self {
                    w: self.w * rhs.w - self.x * rhs.x - self.y * rhs.y - self.z * rhs.z,
                    x: self.w * rhs.x + self.x * rhs.w + self.y * rhs.z - self.z * rhs.y,
                    y: self.w * rhs.y - self.x * rhs.z + self.y * rhs.w + self.z * rhs.x,
                    z: self.w * rhs.z + self.x * rhs.y - self.y * rhs.x + self.z * rhs.w,
                }
            }
        }

        impl Mul<$vec3_ty> for $quat_ty {
            type Output = Self;

            /// Returns the multiplication of a Quaternion with a vector.  This is a
            /// normal Quaternion multiplication where the vector is treated a
            /// Quaternion with a W element value of zero.  The Quaternion is post-
            /// multiplied by the vector.
            fn mul(self, rhs: $vec3_ty) -> Self::Output {
                Self {
                    w: -self.x * rhs.x - self.y * rhs.y - self.z * rhs.z,
                    x: self.w * rhs.x + self.y * rhs.z - self.z * rhs.y,
                    y: self.w * rhs.y - self.x * rhs.z + self.z * rhs.x,
                    z: self.w * rhs.z + self.x * rhs.y - self.y * rhs.x,
                }
            }
        }

        impl Mul<$fx> for $quat_ty {
            type Output = Self;

            fn mul(self, rhs: $fx) -> Self::Output {
                Self {
                    w: self.w * rhs,
                    x: self.x * rhs,
                    y: self.y * rhs,
                    z: self.z * rhs,
                }
            }
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

        impl Div<Self> for $quat_ty {
            type Output = Self;

            fn div(self, rhs: Self) -> Self::Output {
                self * rhs.inverse()
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

            pub fn inverse(self) -> Self {
                Self {
                    w: self.w,
                    x: -self.x,
                    y: -self.y,
                    z: -self.z,
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
            /// Returns the magnitude.
            pub fn magnitude(&self) -> $fx {
                (self.w.powi(2) + self.x.powi(2) + self.y.powi(2) + self.z.powi(2)).sqrt()
            }

            /// Returns the normalised version of the quaternion
            pub fn to_normalized(self) -> Self {
                let mag_recip = $fx::splat(1.) / self.magnitude();
                self * mag_recip
            }

            /// Used by `slerp`.
            pub fn dot(&self, rhs: Self) -> $fx {
                self.w * rhs.w + self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
            }

            /// Rotate a vector using this quaternion. Note that our multiplication Q * v
            /// operation is effectively quaternion multiplication, with a quaternion
            /// created by a vec with w=0.
            /// Uses the right hand rule.
            pub fn rotate_vec(self, vec: $vec3_ty) -> $vec3_ty {
                (self * vec * self.inverse()).to_vec()
            }
        }

        /// Used for creating a set of `Vec3x` from one of `Vec3`. Padded as required. The result
        /// will have approximately x fewer elements than the input.
        pub fn pack_vec3(vecs: &[Vec3]) -> Vec<$vec3_ty> {
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
            padded
                .chunks_exact($lanes)
                .map(|chunk| {
                    // Convert the slice chunk into an array of x Vec3 elements.
                    let arr: [Vec3; $lanes] = chunk.try_into().unwrap();
                    $vec3_ty::from_array(arr)
                })
                .collect()
        }

        /// Unpack an array of SIMD `Vec3x` to normal `Vec3`. The result
        /// will have approximately x as many elements as the input.
        pub fn unpack_vec3(vecs: &[$vec3_ty]) -> Vec<Vec3> {
            let mut result = Vec::new();
            for vec in vecs {
                result.extend(&vec.to_array());
            }
            result
        }

        // todo: Use the paste lib etc to put `$f` in the fn names here.
        /// Convert a slice of `$f` to an array of SIMD `$f` values, x-wide. Padded as required. The result
        /// will have approximately 8x fewer elements than the input.
        pub fn pack_f32(vals: &[$f]) -> Vec<$fx> {
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
            padded
                .chunks_exact($lanes)
                // .map(|chunk| unsafe { _mm256_loadu_ps(chunk.as_ptr()) })
                .map(|chunk| $fx::load(chunk.as_ptr()))
                .collect()
        }

        /// Convert a slice of SIMD `$f` values, x-wide to an Vec of `$f`. The result
        /// will have approximately 8x as many elements as the input.
        pub fn unpack_f32(vals: &[$fx]) -> Vec<$f> {
            let mut result = Vec::new();
            for val in vals {
                result.extend(&val.to_array());
            }
            result
        }
    };
}
