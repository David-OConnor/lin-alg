//! Notes.
//! Array of structs (AoS): Each vector (x,y,z,w) is stored contiguously in memory.
//! For a single 4D vector of f32, you can hold all 4 floats in one __m128.
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

use std::{
    arch::x86_64::{
        __m256, __m256d, _CMP_EQ_OQ, _mm256_add_ps, _mm256_cmp_ps, _mm256_div_ps, _mm256_loadu_ps,
        _mm256_movemask_ps, _mm256_mul_ps, _mm256_set1_ps, _mm256_sqrt_ps, _mm256_sub_ps,
    },
    convert::TryInto,
    mem::transmute,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use crate::f32::{Quaternion, Vec3, Vec4};

/// SoA. Performs operations on 8 Vecs.
#[derive(Clone, Copy, Debug)]
pub struct Vec3x8 {
    pub x: f32x8,
    pub y: f32x8,
    pub z: f32x8,
}

/// SoA. Performs operations on 4 Vecs.
#[derive(Clone, Copy, Debug)]
pub struct Vec3sF64 {
    pub x: __m256d,
    pub y: __m256d,
    pub z: __m256d,
}

/// SoA. Performs operations on 8 Vecs.
#[derive(Clone, Copy, Debug)]
pub struct Vec4x8 {
    pub x: f32x8,
    pub y: f32x8,
    pub z: f32x8,
    pub w: f32x8,
}

/// SoA. Performs operations on 4 Vecs.
#[derive(Clone, Copy, Debug)]
pub struct Vec4x8F64 {
    pub x: __m256d,
    pub y: __m256d,
    pub z: __m256d,
    pub w: __m256d,
}

impl Default for Vec3x8 {
    fn default() -> Self {
        Self::new_zero()
    }
}

impl Neg for Vec3x8 {
    type Output = Self;

    fn neg(self) -> Self::Output {
        let zero = f32x8::splat(0.);

        Self {
            x: zero - self.x,
            y: zero - self.y,
            z: zero - self.z,
        }
    }
}

impl Add for Vec3x8 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

impl AddAssign for Vec3x8 {
    fn add_assign(&mut self, rhs: Self) {
        self.x += rhs.x;
        self.y += rhs.y;
        self.z += rhs.z;
    }
}

impl Add<f32> for Vec3x8 {
    type Output = Self;

    fn add(self, rhs: f32) -> Self::Output {
        let s = f32x8::splat(rhs);
        self + s
    }
}

impl Add<f32x8> for Vec3x8 {
    type Output = Self;

    fn add(self, rhs: f32x8) -> Self::Output {
        Self {
            x: self.x + rhs,
            y: self.y + rhs,
            z: self.z + rhs,
        }
    }
}

impl AddAssign<f32x8> for Vec3x8 {
    fn add_assign(&mut self, rhs: f32x8) {
        self.x += rhs;
        self.y += rhs;
        self.z += rhs;
    }
}

impl Sub for Vec3x8 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

impl SubAssign for Vec3x8 {
    fn sub_assign(&mut self, rhs: Self) {
        self.x -= rhs.x;
        self.y -= rhs.y;
        self.z -= rhs.z;
    }
}

impl Sub<f32> for Vec3x8 {
    type Output = Self;

    fn sub(self, rhs: f32) -> Self::Output {
        let s = f32x8::splat(rhs);
        self - s
    }
}

impl Sub<f32x8> for Vec3x8 {
    type Output = Self;

    fn sub(self, rhs: f32x8) -> Self::Output {
        Self {
            x: self.x - rhs,
            y: self.y - rhs,
            z: self.z - rhs,
        }
    }
}

impl SubAssign<f32x8> for Vec3x8 {
    fn sub_assign(&mut self, rhs: f32x8) {
        self.x -= rhs;
        self.y -= rhs;
        self.z -= rhs;
    }
}

impl Mul<f32> for Vec3x8 {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output {
        let s = f32x8::splat(rhs);
        self * s
    }
}

impl Mul<f32x8> for Vec3x8 {
    type Output = Self;

    fn mul(self, rhs: f32x8) -> Self::Output {
        Self {
            x: self.x * rhs,
            y: self.y * rhs,
            z: self.z * rhs,
        }
    }
}

impl MulAssign<f32x8> for Vec3x8 {
    fn mul_assign(&mut self, rhs: f32x8) {
        self.x = self.x * rhs;
        self.y = self.y * rhs;
        self.z = self.z * rhs;
    }
}

impl Div<f32> for Vec3x8 {
    type Output = Self;

    fn div(self, rhs: f32) -> Self::Output {
        let s = f32x8::splat(rhs);
        self / s
    }
}

impl Div<f32x8> for Vec3x8 {
    type Output = Self;

    fn div(self, rhs: f32x8) -> Self::Output {
        Self {
            x: self.x / rhs,
            y: self.y / rhs,
            z: self.z / rhs,
        }
    }
}

impl DivAssign<f32x8> for Vec3x8 {
    fn div_assign(&mut self, rhs: f32x8) {
        self.x = self.x / rhs;
        self.y = self.y / rhs;
        self.z = self.z / rhs;
    }
}

impl Add for Vec4x8 {
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

impl Sub for Vec4x8 {
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

impl Neg for Vec4x8 {
    type Output = Self;

    fn neg(self) -> Self::Output {
        let zero = f32x8::splat(0.);

        Self {
            x: zero - self.x,
            y: zero - self.y,
            z: zero - self.z,
            w: zero - self.w,
        }
    }
}

impl Mul<f32> for Vec4x8 {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output {
        let s = f32x8::splat(rhs);

        Self {
            x: self.x * s,
            y: self.y * s,
            z: self.z * s,
            w: self.w * s,
        }
    }
}

impl Vec3x8 {
    /// Create a new 8-lane, SoA, f32 Vec3
    pub fn from_array(arr: [Vec3; 8]) -> Self {
        let x_vals = arr.iter().map(|v| v.x).collect::<Vec<_>>();
        let y_vals = arr.iter().map(|v| v.y).collect::<Vec<_>>();
        let z_vals = arr.iter().map(|v| v.z).collect::<Vec<_>>();

        Self {
            x: f32x8::from_slice(&x_vals),
            y: f32x8::from_slice(&y_vals),
            z: f32x8::from_slice(&z_vals),
        }
    }

    pub fn from_slice(slice: &[Vec3]) -> Self {
        let x_vals = slice.iter().map(|v| v.x).collect::<Vec<_>>();
        let y_vals = slice.iter().map(|v| v.y).collect::<Vec<_>>();
        let z_vals = slice.iter().map(|v| v.z).collect::<Vec<_>>();

        Self {
            x: f32x8::from_slice(&x_vals),
            y: f32x8::from_slice(&y_vals),
            z: f32x8::from_slice(&z_vals),
        }
    }

    /// Convert the SoA data back into an array of eight Vec3s.
    pub fn to_array(self) -> [Vec3; 8] {
        let x_arr = self.x.to_array();
        let y_arr = self.y.to_array();
        let z_arr = self.z.to_array();

        let mut out = [Vec3 {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }; 8];
        for i in 0..8 {
            out[i] = Vec3 {
                x: x_arr[i],
                y: y_arr[i],
                z: z_arr[i],
            };
        }
        out
    }

    pub fn new_zero() -> Self {
        let zero = f32x8::splat(0.);
        Self {
            x: zero,
            y: zero,
            z: zero,
        }
    }

    pub fn splat(val: Vec3) -> Self {
        Self {
            x: f32x8::splat(val.x),
            y: f32x8::splat(val.y),
            z: f32x8::splat(val.z),
        }
    }

    /// Dot product across x, y, z lanes. Each lane result is x_i*x_j + y_i*y_j + z_i*z_j.
    /// Returns an f32x8 of 8 dot products (one for each lane).
    pub fn dot(self, rhs: Self) -> f32x8 {
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
    pub fn magnitude_squared(self) -> f32x8 {
        self.x.powi(2) + self.y.powi(2) + self.z.powi(2)
    }

    /// Returns the vector magnitude
    pub fn magnitude(&self) -> f32x8 {
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
        let mag_recip = f32x8::splat(1.) / self.magnitude();
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

impl Vec4x8 {
    pub fn from_array(arr: [Vec4; 8]) -> Self {
        let x_vals = arr.iter().map(|v| v.x).collect::<Vec<_>>();
        let y_vals = arr.iter().map(|v| v.y).collect::<Vec<_>>();
        let z_vals = arr.iter().map(|v| v.z).collect::<Vec<_>>();
        let w_vals = arr.iter().map(|v| v.w).collect::<Vec<_>>();

        Self {
            x: f32x8::from_slice(&x_vals),
            y: f32x8::from_slice(&y_vals),
            z: f32x8::from_slice(&z_vals),
            w: f32x8::from_slice(&w_vals),
        }
    }

    pub fn from_slice(slice: &[Vec4]) -> Self {
        let x_vals = slice.iter().map(|v| v.x).collect::<Vec<_>>();
        let y_vals = slice.iter().map(|v| v.y).collect::<Vec<_>>();
        let z_vals = slice.iter().map(|v| v.z).collect::<Vec<_>>();
        let w_vals = slice.iter().map(|v| v.z).collect::<Vec<_>>();

        Self {
            x: f32x8::from_slice(&x_vals),
            y: f32x8::from_slice(&y_vals),
            z: f32x8::from_slice(&z_vals),
            w: f32x8::from_slice(&w_vals),
        }
    }

    /// Convert the SoA data back into an array of eight Vec3s.
    pub fn to_array(self) -> [Vec4; 8] {
        let x_arr = self.x.to_array();
        let y_arr = self.y.to_array();
        let z_arr = self.z.to_array();
        let w_arr = self.w.to_array();

        let mut out = [Vec4 {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            w: 0.0,
        }; 8];
        for i in 0..8 {
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
            x: f32x8::splat(val.x),
            y: f32x8::splat(val.y),
            z: f32x8::splat(val.z),
            w: f32x8::splat(val.w),
        }
    }

    /// Dot product across x, y, z lanes. Each lane result is x_i*x_j + y_i*y_j + z_i*z_j.
    /// Returns an __m256 of 8 dot products (one for each lane).
    pub fn dot(self, rhs: Self) -> f32x8 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z + self.w * rhs.w
    }
}

/// SoA.
#[derive(Clone, Copy, Debug)]
pub struct Quaternionx8 {
    pub w: f32x8,
    pub x: f32x8,
    pub y: f32x8,
    pub z: f32x8,
}

impl Default for Quaternionx8 {
    fn default() -> Self {
        Self::new_identity()
    }
}

impl Add<Self> for Quaternionx8 {
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

impl Sub<Self> for Quaternionx8 {
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

impl Mul for Quaternionx8 {
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

impl Mul<Vec3x8> for Quaternionx8 {
    type Output = Self;

    /// Returns the multiplication of a Quaternion with a vector.  This is a
    /// normal Quaternion multiplication where the vector is treated a
    /// Quaternion with a W element value of zero.  The Quaternion is post-
    /// multiplied by the vector.
    fn mul(self, rhs: Vec3x8) -> Self::Output {
        Self {
            w: -self.x * rhs.x - self.y * rhs.y - self.z * rhs.z,
            x: self.w * rhs.x + self.y * rhs.z - self.z * rhs.y,
            y: self.w * rhs.y - self.x * rhs.z + self.z * rhs.x,
            z: self.w * rhs.z + self.x * rhs.y - self.y * rhs.x,
        }
    }
}

impl Mul<f32x8> for Quaternionx8 {
    type Output = Self;

    fn mul(self, rhs: f32x8) -> Self::Output {
        Self {
            w: self.w * rhs,
            x: self.x * rhs,
            y: self.y * rhs,
            z: self.z * rhs,
        }
    }
}

impl Mul<f32> for Quaternionx8 {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output {
        let s = f32x8::splat(rhs);
        Self {
            w: self.w * s,
            x: self.x * s,
            y: self.y * s,
            z: self.z * s,
        }
    }
}

impl Div<Self> for Quaternionx8 {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        self * rhs.inverse()
    }
}

impl Quaternionx8 {
    /// Create a new 8-lane, SoA, f32 Vec3
    pub fn from_array(slots: [Quaternion; 8]) -> Self {
        let mut w_arr = [0.; 8];
        let mut x_arr = [0.; 8];
        let mut y_arr = [0.; 8];
        let mut z_arr = [0.; 8];

        for i in 0..8 {
            w_arr[i] = slots[i].w;
            x_arr[i] = slots[i].x;
            y_arr[i] = slots[i].y;
            z_arr[i] = slots[i].z;
        }

        Self {
            w: f32x8::from_slice(&w_arr),
            x: f32x8::from_slice(&x_arr),
            y: f32x8::from_slice(&y_arr),
            z: f32x8::from_slice(&z_arr),
        }
    }

    pub fn from_slice(slice: &[Quaternion]) -> Self {
        let mut w_arr = [0.; 8];
        let mut x_arr = [0.; 8];
        let mut y_arr = [0.; 8];
        let mut z_arr = [0.; 8];

        for i in 0..8 {
            w_arr[i] = slice[i].w;
            x_arr[i] = slice[i].x;
            y_arr[i] = slice[i].y;
            z_arr[i] = slice[i].z;
        }

        Self {
            w: f32x8::from_slice(&w_arr),
            x: f32x8::from_slice(&x_arr),
            y: f32x8::from_slice(&y_arr),
            z: f32x8::from_slice(&z_arr),
        }
    }

    /// Convert the SoA data back into an array of eight Quaternions.
    pub fn to_array(self) -> [Quaternion; 8] {
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
        }; 8];
        for i in 0..8 {
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
            w: f32x8::splat(1.),
            x: f32x8::splat(0.),
            y: f32x8::splat(0.),
            z: f32x8::splat(0.),
        }
    }

    pub fn splat(val: Quaternion) -> Self {
        Self {
            w: f32x8::splat(val.w),
            x: f32x8::splat(val.x),
            y: f32x8::splat(val.y),
            z: f32x8::splat(val.z),
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
    pub fn to_vec(self) -> Vec3x8 {
        Vec3x8 {
            x: self.x,
            y: self.y,
            z: self.z,
        }
    }
    /// Returns the magnitude.
    pub fn magnitude(&self) -> f32x8 {
        (self.w.powi(2) + self.x.powi(2) + self.y.powi(2) + self.z.powi(2)).sqrt()
    }

    /// Returns the normalised version of the quaternion
    pub fn to_normalized(self) -> Self {
        let mag_recip = f32x8::splat(1.) / self.magnitude();
        self * mag_recip
    }

    /// Used by `slerp`.
    pub fn dot(&self, rhs: Self) -> f32x8 {
        self.w * rhs.w + self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }

    /// Rotate a vector using this quaternion. Note that our multiplication Q * v
    /// operation is effectively quaternion multiplication, with a quaternion
    /// created by a vec with w=0.
    /// Uses the right hand rule.
    pub fn rotate_vec(self, vec: Vec3x8) -> Vec3x8 {
        (self * vec * self.inverse()).to_vec()
    }
}

#[derive(Copy, Clone, Debug)]
#[allow(non_camel_case_types)]
#[repr(transparent)]
pub struct f32x8(__m256);

impl f32x8 {
    #[inline]
    pub fn from_slice(slice: &[f32]) -> Self {
        Self::load(slice.as_ptr())
    }

    #[inline]
    pub fn from_array(arr: [f32; 8]) -> Self {
        Self::load(arr.as_ptr())
    }

    #[inline]
    pub fn to_array(&self) -> [f32; 8] {
        unsafe { transmute(self.0) }
    }

    #[inline]
    pub fn load(ptr: *const f32) -> Self {
        unsafe { Self(_mm256_loadu_ps(ptr)) }
    }

    #[inline]
    pub fn splat(val: f32) -> Self {
        unsafe { Self(_mm256_set1_ps(val)) }
    }

    #[inline]
    pub fn sqrt(&self) -> Self {
        unsafe { Self(_mm256_sqrt_ps(self.0)) }
    }

    /// todo: This doesn't match the core::simd API. What's the equivalent there?
    /// todo: This is potentially a slow approach compared to using intrinsics
    pub fn replace(self, index: usize, value: f32) -> Self {
        let mut arr = self.to_array();
        // This will panic if index >= 8, similar to core::simd::Simd::replace.
        arr[index] = value;
        Self::from_array(arr)
    }

    pub fn powi(self, mut n: i32) -> Self {
        // todo: QC this one.
        // Handle the zero exponent case: x^0 = 1 for any x.
        if n == 0 {
            return unsafe { Self(_mm256_set1_ps(1.0)) };
        }

        let mut base = self;
        // If the exponent is negative, invert the base and use the positive exponent.
        if n < 0 {
            base = unsafe { Self(_mm256_div_ps(_mm256_set1_ps(1.0), base.0)) };
            n = -n;
        }

        // Initialize result as 1 for each lane.
        let mut result = unsafe { Self(_mm256_set1_ps(1.0)) };

        // Exponentiation by squaring.
        while n > 0 {
            if n & 1 == 1 {
                result = result * base;
            }
            base = base * base;
            n /= 2;
        }
        result
    }
}

impl Neg for f32x8 {
    type Output = Self;

    fn neg(self) -> Self::Output {
        f32x8::splat(0.) - self
    }
}

impl Add for f32x8 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        unsafe { Self(_mm256_add_ps(self.0, rhs.0)) }
    }
}

impl Sub for f32x8 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        unsafe { Self(_mm256_sub_ps(self.0, rhs.0)) }
    }
}

impl Mul for f32x8 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        unsafe { Self(_mm256_mul_ps(self.0, rhs.0)) }
    }
}

impl Div for f32x8 {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        unsafe { Self(_mm256_div_ps(self.0, rhs.0)) }
    }
}

impl AddAssign for f32x8 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        unsafe { *self = Self(_mm256_add_ps(self.0, rhs.0)) }
    }
}

impl SubAssign for f32x8 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        unsafe { *self = Self(_mm256_sub_ps(self.0, rhs.0)) }
    }
}

impl MulAssign for f32x8 {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        unsafe { *self = Self(_mm256_mul_ps(self.0, rhs.0)) }
    }
}

impl DivAssign for f32x8 {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        unsafe { *self = Self(_mm256_div_ps(self.0, rhs.0)) }
    }
}

/// Used for creating a set of `Vec3x8` from one of `Vec3`. Padded as required. The result
/// will have approximately 8x fewer elements than the input.
pub fn pack_vec3(vecs: &[Vec3]) -> Vec<Vec3x8> {
    let remainder = vecs.len() % 8;
    let padding_needed = if remainder == 0 { 0 } else { 8 - remainder };

    let mut padded = Vec::with_capacity(vecs.len() + padding_needed);
    padded.extend_from_slice(vecs);
    padded.extend((0..padding_needed).map(|_| Vec3::new_zero()));

    // Now `padded.len()` is a multiple of 8, so chunks_exact(8) will consume it fully.
    padded
        .chunks_exact(8)
        .map(|chunk| {
            // Convert the slice chunk into an array of 8 Vec3 elements.
            let arr: [Vec3; 8] = chunk.try_into().unwrap();
            Vec3x8::from_array(arr)
        })
        .collect()
}

/// Unpack an array of SIMD `Vec3x8` to normal `Vec3`. The result
/// will have approximately 8x as many elements as the input.
pub fn unpack_vec3(vecs: &[Vec3x8]) -> Vec<Vec3> {
    let mut result = Vec::new();
    for vec in vecs {
        result.extend(&vec.to_array());
    }
    result
}

/// Convert a slice of `f32` to an array of SIMD `f32` values, 8-wide. Padded as required. The result
/// will have approximately 8x fewer elements than the input.
pub fn pack_f32(vals: &[f32]) -> Vec<f32x8> {
    let remainder = vals.len() % 8;
    let padding_needed = if remainder == 0 { 0 } else { 8 - remainder };

    let mut padded = Vec::with_capacity(vals.len() + padding_needed);
    padded.extend_from_slice(vals);
    padded.extend((0..padding_needed).map(|_| 0.));

    // Now `padded.len()` is a multiple of 8, so chunks_exact(8) will consume it fully.
    padded
        .chunks_exact(8)
        // .map(|chunk| unsafe { _mm256_loadu_ps(chunk.as_ptr()) })
        .map(|chunk| f32x8::load(chunk.as_ptr()))
        .collect()
}

/// Convert a slice of SIMD `f32` values, 8-wide to an Vec of `f32`. The result
/// will have approximately 8x as many elements as the input.
pub fn unpack_f32(vals: &[f32x8]) -> Vec<f32> {
    let mut result = Vec::new();
    for val in vals {
        result.extend(&val.to_array());
    }
    result
}
