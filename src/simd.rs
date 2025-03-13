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
        __m256, __m256d, _mm256_add_ps, _mm256_div_ps, _mm256_loadu_ps, _mm256_mul_ps,
        _mm256_set1_ps, _mm256_setzero_ps, _mm256_sqrt_ps, _mm256_sub_ps,
    },
    convert::TryInto,
    mem::transmute,
    ops::{Add, Div, DivAssign, Mul, MulAssign, Neg, Sub},
};

use crate::f32::{Quaternion, Vec3};

/// SoA. Performs operations on 8 Vecs.
#[derive(Clone, Copy, Debug)]
pub struct Vec3S {
    pub x: __m256,
    pub y: __m256,
    pub z: __m256,
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
pub struct Vec4S {
    pub x: __m256,
    pub y: __m256,
    pub z: __m256,
    pub w: __m256,
}

/// SoA. Performs operations on 4 Vecs.
#[derive(Clone, Copy, Debug)]
pub struct Vec4sF64 {
    pub x: __m256d,
    pub y: __m256d,
    pub z: __m256d,
    pub w: __m256d,
}

impl Default for Vec3S {
    fn default() -> Self {
        Self::new_zero()
    }
}

impl Add for Vec3S {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        unsafe {
            Self {
                x: _mm256_add_ps(self.x, rhs.x),
                y: _mm256_add_ps(self.y, rhs.y),
                z: _mm256_add_ps(self.z, rhs.z),
            }
        }
    }
}

impl Sub for Vec3S {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        unsafe {
            Self {
                x: _mm256_sub_ps(self.x, rhs.x),
                y: _mm256_sub_ps(self.y, rhs.y),
                z: _mm256_sub_ps(self.z, rhs.z),
            }
        }
    }
}

impl Neg for Vec3S {
    type Output = Self;

    fn neg(self) -> Self::Output {
        unsafe {
            let zero = _mm256_setzero_ps();

            Self {
                x: _mm256_sub_ps(zero, self.x),
                y: _mm256_sub_ps(zero, self.y),
                z: _mm256_sub_ps(zero, self.z),
            }
        }
    }
}

impl Mul<f32> for Vec3S {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output {
        unsafe {
            let s = _mm256_set1_ps(rhs);

            Self {
                x: _mm256_mul_ps(self.x, s),
                y: _mm256_mul_ps(self.y, s),
                z: _mm256_mul_ps(self.z, s),
            }
        }
    }
}

impl Mul<[f32; 8]> for Vec3S {
    type Output = Self;

    fn mul(self, rhs: [f32; 8]) -> Self::Output {
        unsafe {
            // Load the array into a __m256 register for lane-wise multiplication:
            let r = _mm256_loadu_ps(rhs.as_ptr());
            Self {
                x: _mm256_mul_ps(self.x, r),
                y: _mm256_mul_ps(self.y, r),
                z: _mm256_mul_ps(self.z, r),
            }
        }
    }
}

impl Mul<__m256> for Vec3S {
    type Output = Self;

    fn mul(self, rhs: __m256) -> Self::Output {
        unsafe {
            // Load the array into a __m256 register for lane-wise multiplication:
            Self {
                x: _mm256_mul_ps(self.x, rhs),
                y: _mm256_mul_ps(self.y, rhs),
                z: _mm256_mul_ps(self.z, rhs),
            }
        }
    }
}

impl Div<f32> for Vec3S {
    type Output = Self;

    fn div(self, rhs: f32) -> Self::Output {
        unsafe {
            let s = _mm256_set1_ps(rhs);

            Self {
                x: _mm256_div_ps(self.x, s),
                y: _mm256_div_ps(self.y, s),
                z: _mm256_div_ps(self.z, s),
            }
        }
    }
}

impl Div<[f32; 8]> for Vec3S {
    type Output = Self;

    fn div(self, rhs: [f32; 8]) -> Self::Output {
        unsafe {
            // Load the array into a __m256 register for lane-wise divtiplication:
            let r = _mm256_loadu_ps(rhs.as_ptr());
            Self {
                x: _mm256_div_ps(self.x, r),
                y: _mm256_div_ps(self.y, r),
                z: _mm256_div_ps(self.z, r),
            }
        }
    }
}

impl Div<__m256> for Vec3S {
    type Output = Self;

    fn div(self, rhs: __m256) -> Self::Output {
        unsafe {
            // Load the array into a __m256 register for lane-wise divtiplication:
            Self {
                x: _mm256_div_ps(self.x, rhs),
                y: _mm256_div_ps(self.y, rhs),
                z: _mm256_div_ps(self.z, rhs),
            }
        }
    }
}

impl Add<f32> for Vec3S {
    type Output = Self;

    fn add(self, rhs: f32) -> Self::Output {
        unsafe {
            let s = _mm256_set1_ps(rhs);

            Self {
                x: _mm256_add_ps(self.x, s),
                y: _mm256_add_ps(self.y, s),
                z: _mm256_add_ps(self.z, s),
            }
        }
    }
}

impl Add<[f32; 8]> for Vec3S {
    type Output = Self;

    fn add(self, rhs: [f32; 8]) -> Self::Output {
        unsafe {
            // Load the array into a __m256 register for lane-wise addition
            let r = _mm256_loadu_ps(rhs.as_ptr());
            Self {
                x: _mm256_add_ps(self.x, r),
                y: _mm256_add_ps(self.y, r),
                z: _mm256_add_ps(self.z, r),
            }
        }
    }
}

impl Add<__m256> for Vec3S {
    type Output = Self;

    fn add(self, rhs: __m256) -> Self::Output {
        unsafe {
            // Load the array into a __m256 register for lane-wise addition
            Self {
                x: _mm256_add_ps(self.x, rhs),
                y: _mm256_add_ps(self.y, rhs),
                z: _mm256_add_ps(self.z, rhs),
            }
        }
    }
}

impl Sub<f32> for Vec3S {
    type Output = Self;

    fn sub(self, rhs: f32) -> Self::Output {
        unsafe {
            let s = _mm256_set1_ps(rhs);

            Self {
                x: _mm256_sub_ps(self.x, s),
                y: _mm256_sub_ps(self.y, s),
                z: _mm256_sub_ps(self.z, s),
            }
        }
    }
}

impl Sub<[f32; 8]> for Vec3S {
    type Output = Self;

    fn sub(self, rhs: [f32; 8]) -> Self::Output {
        unsafe {
            // Load the array into a __m256 register for lane-wise subtiplication:
            let r = _mm256_loadu_ps(rhs.as_ptr());
            Self {
                x: _mm256_sub_ps(self.x, r),
                y: _mm256_sub_ps(self.y, r),
                z: _mm256_sub_ps(self.z, r),
            }
        }
    }
}

impl Sub<__m256> for Vec3S {
    type Output = Self;

    fn sub(self, rhs: __m256) -> Self::Output {
        unsafe {
            // Load the array into a __m256 register for lane-wise subtiplication:
            Self {
                x: _mm256_sub_ps(self.x, rhs),
                y: _mm256_sub_ps(self.y, rhs),
                z: _mm256_sub_ps(self.z, rhs),
            }
        }
    }
}

impl Add for Vec4S {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        unsafe {
            Self {
                x: _mm256_add_ps(self.x, rhs.x),
                y: _mm256_add_ps(self.y, rhs.y),
                z: _mm256_add_ps(self.z, rhs.z),
                w: _mm256_add_ps(self.w, rhs.w),
            }
        }
    }
}

impl Sub for Vec4S {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        unsafe {
            Self {
                x: _mm256_sub_ps(self.x, rhs.x),
                y: _mm256_sub_ps(self.y, rhs.y),
                z: _mm256_sub_ps(self.z, rhs.z),
                w: _mm256_sub_ps(self.w, rhs.w),
            }
        }
    }
}

impl Neg for Vec4S {
    type Output = Self;

    fn neg(self) -> Self::Output {
        unsafe {
            let zero = _mm256_setzero_ps();

            Self {
                x: _mm256_sub_ps(zero, self.x),
                y: _mm256_sub_ps(zero, self.y),
                z: _mm256_sub_ps(zero, self.z),
                w: _mm256_sub_ps(zero, self.w),
            }
        }
    }
}

impl Mul<f32> for Vec4S {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output {
        unsafe {
            let s = _mm256_set1_ps(rhs);

            Self {
                x: _mm256_mul_ps(self.x, s),
                y: _mm256_mul_ps(self.y, s),
                z: _mm256_mul_ps(self.z, s),
                w: _mm256_mul_ps(self.w, s),
            }
        }
    }
}

impl MulAssign<f32> for Vec3S {
    fn mul_assign(&mut self, rhs: f32) {
        unsafe {
            let s = _mm256_set1_ps(rhs);
            self.x = _mm256_mul_ps(self.x, s);
            self.y = _mm256_mul_ps(self.y, s);
            self.z = _mm256_mul_ps(self.z, s);
        }
    }
}

/// DivAssign by scalar
impl DivAssign<f32> for Vec3S {
    fn div_assign(&mut self, rhs: f32) {
        unsafe {
            let s = _mm256_set1_ps(rhs);
            self.x = _mm256_div_ps(self.x, s);
            self.y = _mm256_div_ps(self.y, s);
            self.z = _mm256_div_ps(self.z, s);
        }
    }
}

impl Vec3S {
    /// Create a new 8-lane, SoA, f32 Vec3
    pub fn new(slots: [Vec3; 8]) -> Self {
        // unsafe {
        //     Self {
        //         // lane order is reversed in _mm256_set_ps: (a7, a6, ..., a0)
        //         x: _mm256_set_ps(
        //             slots[7].x, slots[6].x, slots[5].x, slots[4].x,
        //             slots[3].x, slots[2].x, slots[1].x, slots[0].x,
        //         ),
        //         y: _mm256_set_ps(
        //             slots[7].y, slots[6].y, slots[5].y, slots[4].y,
        //             slots[3].y, slots[2].y, slots[1].y, slots[0].y,
        //         ),
        //         z: _mm256_set_ps(
        //             slots[7].z, slots[6].z, slots[5].z, slots[4].z,
        //             slots[3].z, slots[2].z, slots[1].z, slots[0].z,
        //         ),
        //     }
        // }

        let mut x_arr = [0.; 8];
        let mut y_arr = [0.; 8];
        let mut z_arr = [0.; 8];

        for i in 0..8 {
            x_arr[i] = slots[i].x;
            y_arr[i] = slots[i].y;
            z_arr[i] = slots[i].z;
        }

        unsafe {
            Self {
                x: _mm256_loadu_ps(x_arr.as_ptr()),
                y: _mm256_loadu_ps(y_arr.as_ptr()),
                z: _mm256_loadu_ps(z_arr.as_ptr()),
            }
        }
    }

    fn new_zero() -> Self {
        unsafe {
            Self {
                x: _mm256_setzero_ps(),
                y: _mm256_setzero_ps(),
                z: _mm256_setzero_ps(),
            }
        }
    }

    /// Convert the SoA data back into an array of eight Vec3s.
    pub fn unpack(self) -> [Vec3; 8] {
        unsafe {
            // Extract each lane from the AVX registers into arrays of length 8
            let x_arr: [f32; 8] = transmute(self.x);
            let y_arr: [f32; 8] = transmute(self.y);
            let z_arr: [f32; 8] = transmute(self.z);

            // Reconstruct each Vec3 from corresponding lanes
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
    }

    /// Dot product across x, y, z lanes. Each lane result is x_i*x_j + y_i*y_j + z_i*z_j.
    /// Returns an __m256 of 8 dot products (one for each lane).
    pub fn dot(self, rhs: Self) -> __m256 {
        unsafe {
            let mut r = _mm256_mul_ps(self.x, rhs.x);
            r = _mm256_add_ps(r, _mm256_mul_ps(self.y, rhs.y));
            r = _mm256_add_ps(r, _mm256_mul_ps(self.z, rhs.z));
            r
        }
    }

    /// Similar to dot product, but unpacked to 8 f32s.
    pub fn dot_unpack(self, rhs: Self) -> [f32; 8] {
        unsafe { transmute(self.dot(rhs)) }
    }

    /// Hadamard product (lane-wise multiplication of x, y, z).
    pub fn hadamard_product(self, rhs: Self) -> Self {
        unsafe {
            Self {
                x: _mm256_mul_ps(self.x, rhs.x),
                y: _mm256_mul_ps(self.y, rhs.y),
                z: _mm256_mul_ps(self.z, rhs.z),
            }
        }
    }

    /// Magnitude squared per lane: x^2 + y^2 + z^2
    /// Returns an __m256 with the 8 results (one per lane).
    pub fn magnitude_squared(self) -> __m256 {
        unsafe {
            let mut r = _mm256_mul_ps(self.x, self.x);
            r = _mm256_add_ps(r, _mm256_mul_ps(self.y, self.y));
            r = _mm256_add_ps(r, _mm256_mul_ps(self.z, self.z));
            r
        }
    }

    /// Magnitude per lane = sqrt(x^2 + y^2 + z^2)
    /// This returns an __m256 of 8 magnitudes.
    pub fn magnitude(self) -> __m256 {
        unsafe {
            let msq = self.magnitude_squared();
            _mm256_sqrt_ps(msq)
        }
    }

    /// Normalize in place for each lane. That is: v /= magnitude(v).
    /// Lanes with zero magnitude will lead to division by zero, so handle with caution.
    pub fn normalize(&mut self) {
        unsafe {
            let ms = self.magnitude();
            // reciprocal of magnitude
            let recip = _mm256_div_ps(_mm256_set1_ps(1.0), ms);

            self.x = _mm256_mul_ps(self.x, recip);
            self.y = _mm256_mul_ps(self.y, recip);
            self.z = _mm256_mul_ps(self.z, recip);
        }
    }

    /// Return a new Vec3sF32 with each lane normalized
    pub fn to_normalized(self) -> Self {
        unsafe {
            let ms = self.magnitude();
            let recip = _mm256_div_ps(_mm256_set1_ps(1.0), ms);

            Self {
                x: _mm256_mul_ps(self.x, recip),
                y: _mm256_mul_ps(self.y, recip),
                z: _mm256_mul_ps(self.z, recip),
            }
        }
    }

    /// Lane-wise cross product
    /// cross.x = self.y * rhs.z - self.z * rhs.y
    /// cross.y = self.z * rhs.x - self.x * rhs.z
    /// cross.z = self.x * rhs.y - self.y * rhs.x
    pub fn cross(self, rhs: Self) -> Self {
        unsafe {
            let yz = _mm256_sub_ps(_mm256_mul_ps(self.y, rhs.z), _mm256_mul_ps(self.z, rhs.y)); // x-lane
            let zx = _mm256_sub_ps(_mm256_mul_ps(self.z, rhs.x), _mm256_mul_ps(self.x, rhs.z)); // y-lane
            let xy = _mm256_sub_ps(_mm256_mul_ps(self.x, rhs.y), _mm256_mul_ps(self.y, rhs.x)); // z-lane

            Self {
                x: yz,
                y: zx,
                z: xy,
            }
        }
    }

    /// Project a vector onto a plane defined by its normal vector. Assumes self and `plane_norm`
    /// are unit vectors.
    pub fn project_to_plane(self, plane_norm: Self) -> Self {
        self - plane_norm * self.dot(plane_norm)
    }
}

impl Vec4S {
    /// Dot product across x, y, z lanes. Each lane result is x_i*x_j + y_i*y_j + z_i*z_j.
    /// Returns an __m256 of 8 dot products (one for each lane).
    pub fn dot(self, rhs: Self) -> __m256 {
        unsafe {
            let mut r = _mm256_mul_ps(self.x, rhs.x);
            r = _mm256_add_ps(r, _mm256_mul_ps(self.y, rhs.y));
            r = _mm256_add_ps(r, _mm256_mul_ps(self.z, rhs.z));
            r = _mm256_add_ps(r, _mm256_mul_ps(self.w, rhs.w));
            r
        }
    }
}

/// SoA.
#[derive(Clone, Copy, Debug)]
pub struct QuaternionS {
    pub w: __m256,
    pub x: __m256,
    pub y: __m256,
    pub z: __m256,
}

impl Default for QuaternionS {
    fn default() -> Self {
        Self::new_identity()
    }
}

impl Add for QuaternionS {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        unsafe {
            Self {
                w: _mm256_add_ps(self.w, rhs.w),
                x: _mm256_add_ps(self.x, rhs.x),
                y: _mm256_add_ps(self.y, rhs.y),
                z: _mm256_add_ps(self.z, rhs.z),
            }
        }
    }
}

impl Sub for QuaternionS {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        unsafe {
            Self {
                w: _mm256_sub_ps(self.w, rhs.w),
                x: _mm256_sub_ps(self.x, rhs.x),
                y: _mm256_sub_ps(self.y, rhs.y),
                z: _mm256_sub_ps(self.z, rhs.z),
            }
        }
    }
}

impl Mul for QuaternionS {
    type Output = Self;

    // todo: QC This against your non-SIMD impl
    fn mul(self, rhs: Self) -> Self::Output {
        unsafe {
            let w = _mm256_sub_ps(
                _mm256_sub_ps(
                    _mm256_sub_ps(_mm256_mul_ps(self.w, rhs.w), _mm256_mul_ps(self.x, rhs.x)),
                    _mm256_mul_ps(self.y, rhs.y),
                ),
                _mm256_mul_ps(self.z, rhs.z),
            );

            let x = _mm256_add_ps(
                _mm256_add_ps(_mm256_mul_ps(self.w, rhs.x), _mm256_mul_ps(self.x, rhs.w)),
                _mm256_sub_ps(_mm256_mul_ps(self.y, rhs.z), _mm256_mul_ps(self.z, rhs.y)),
            );

            let y = _mm256_add_ps(
                _mm256_sub_ps(_mm256_mul_ps(self.w, rhs.y), _mm256_mul_ps(self.x, rhs.z)),
                _mm256_add_ps(_mm256_mul_ps(self.y, rhs.w), _mm256_mul_ps(self.z, rhs.x)),
            );

            let z = _mm256_add_ps(
                _mm256_add_ps(_mm256_mul_ps(self.w, rhs.z), _mm256_mul_ps(self.x, rhs.y)),
                _mm256_sub_ps(_mm256_mul_ps(self.z, rhs.w), _mm256_mul_ps(self.y, rhs.x)),
            );

            Self { w, x, y, z }
        }
    }
}

impl Mul<f32> for QuaternionS {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output {
        unsafe {
            // Broadcast the scalar value into an AVX register.
            let s = _mm256_set1_ps(rhs);
            Self {
                w: _mm256_mul_ps(self.w, s),
                x: _mm256_mul_ps(self.x, s),
                y: _mm256_mul_ps(self.y, s),
                z: _mm256_mul_ps(self.z, s),
            }
        }
    }
}

impl Mul<__m256> for QuaternionS {
    type Output = Self;

    fn mul(self, rhs: __m256) -> Self::Output {
        unsafe {
            Self {
                w: _mm256_mul_ps(self.w, rhs),
                x: _mm256_mul_ps(self.x, rhs),
                y: _mm256_mul_ps(self.y, rhs),
                z: _mm256_mul_ps(self.z, rhs),
            }
        }
    }
}

impl Mul<Vec3S> for QuaternionS {
    type Output = Self;

    /// Returns the multiplication of a Quaternion with a vector.  This is a
    /// normal Quaternion multiplication where the vector is treated a
    /// Quaternion with a W element value of zero.  The Quaternion is post-
    /// multiplied by the vector.
    fn mul(self, rhs: Vec3S) -> Self::Output {
        unsafe {
            // -self.x * rhs.x - self.y * rhs.y - self.z * rhs.z
            let prod_x = _mm256_mul_ps(self.x, rhs.x);
            let prod_y = _mm256_mul_ps(self.y, rhs.y);
            let prod_z = _mm256_mul_ps(self.z, rhs.z);
            let sum_xyz = _mm256_add_ps(_mm256_add_ps(prod_x, prod_y), prod_z);
            let w = _mm256_sub_ps(_mm256_setzero_ps(), sum_xyz);

            // self.w * rhs.x + self.y * rhs.z - self.z * rhs.y
            let wx = _mm256_mul_ps(self.w, rhs.x);
            let yz = _mm256_mul_ps(self.y, rhs.z);
            let zy = _mm256_mul_ps(self.z, rhs.y);
            let x = _mm256_sub_ps(_mm256_add_ps(wx, yz), zy);

            // self.w * rhs.y - self.x * rhs.z + self.z * rhs.x
            let wy = _mm256_mul_ps(self.w, rhs.y);
            let xz = _mm256_mul_ps(self.x, rhs.z);
            let zx = _mm256_mul_ps(self.z, rhs.x);
            let y = _mm256_add_ps(_mm256_sub_ps(wy, xz), zx);

            // self.w * rhs.z + self.x * rhs.y - self.y * rhs.x
            let wz = _mm256_mul_ps(self.w, rhs.z);
            let xy = _mm256_mul_ps(self.x, rhs.y);
            let yx = _mm256_mul_ps(self.y, rhs.x);
            let z = _mm256_sub_ps(_mm256_add_ps(wz, xy), yx);

            Self { w, x, y, z }
        }
    }
}

impl Div for QuaternionS {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        // Division is implemented as multiplication by the inverse.
        self * rhs.inverse()
    }
}

impl QuaternionS {
    /// Create a new 8-lane, SoA, f32 Vec3
    pub fn new(slots: [Quaternion; 8]) -> Self {
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

        unsafe {
            Self {
                w: _mm256_loadu_ps(w_arr.as_ptr()),
                x: _mm256_loadu_ps(x_arr.as_ptr()),
                y: _mm256_loadu_ps(y_arr.as_ptr()),
                z: _mm256_loadu_ps(z_arr.as_ptr()),
            }
        }
    }

    fn new_identity() -> Self {
        unsafe {
            Self {
                w: _mm256_set1_ps(1.),
                x: _mm256_setzero_ps(),
                y: _mm256_setzero_ps(),
                z: _mm256_setzero_ps(),
            }
        }
    }

    /// Convert the SoA data back into an array of eight Quatenrions.
    pub fn unpack(self) -> [Quaternion; 8] {
        unsafe {
            // Extract each lane from the AVX registers into arrays of length 8
            let w_arr: [f32; 8] = transmute(self.w);
            let x_arr: [f32; 8] = transmute(self.x);
            let y_arr: [f32; 8] = transmute(self.y);
            let z_arr: [f32; 8] = transmute(self.z);

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
    }

    pub fn inverse(self) -> Self {
        unsafe {
            let minus_one = _mm256_set1_ps(-1.0);
            Self {
                w: self.w,
                x: _mm256_mul_ps(self.x, minus_one),
                y: _mm256_mul_ps(self.y, minus_one),
                z: _mm256_mul_ps(self.z, minus_one),
            }
        }
    }

    /// Converts the SIMD quaternion to a SIMD 3D vector, discarding `w`.
    pub fn to_vec(self) -> Vec3S {
        Vec3S {
            x: self.x,
            y: self.y,
            z: self.z,
        }
    }

    /// Returns the magnitude (norm) of each quaternion lane as an __m256.
    pub fn magnitude(&self) -> __m256 {
        unsafe {
            let w2 = _mm256_mul_ps(self.w, self.w);
            let x2 = _mm256_mul_ps(self.x, self.x);
            let y2 = _mm256_mul_ps(self.y, self.y);
            let z2 = _mm256_mul_ps(self.z, self.z);

            let sum1 = _mm256_add_ps(w2, x2);
            let sum2 = _mm256_add_ps(y2, z2);
            let sum = _mm256_add_ps(sum1, sum2);

            // Take the square root of each lane.
            _mm256_sqrt_ps(sum)
        }
    }

    /// Returns the normalized quaternion for each lane.
    pub fn to_normalized(self) -> Self {
        unsafe {
            let mag = self.magnitude();
            let one = _mm256_set1_ps(1.0);
            let mag_recip = _mm256_div_ps(one, mag);

            Self {
                w: _mm256_mul_ps(self.w, mag_recip),
                x: _mm256_mul_ps(self.x, mag_recip),
                y: _mm256_mul_ps(self.y, mag_recip),
                z: _mm256_mul_ps(self.z, mag_recip),
            }
        }
    }

    /// Rotate a vector using this quaternion. Note that our multiplication Q * v
    /// operation is effectively quaternion multiplication, with a quaternion
    /// created by a vec with w=0.
    /// Uses the right hand rule.
    pub fn rotate_vec(self, vec: Vec3S) -> Vec3S {
        (self * vec * self.inverse()).to_vec()
    }
}

/// Used for creating a set of `Vec3S` from one of `Vec3`. Padded as required. The result
/// will have approximately 8x fewer elements than the input.
pub fn vec3s_to_simd(vecs: &[Vec3]) -> Vec<Vec3S> {
    let remainder = vecs.len() % 8;
    let padding_needed = if remainder == 0 { 0 } else { 8 - remainder };

    let mut padded = Vec::with_capacity(vecs.len() + padding_needed);
    padded.extend_from_slice(vecs);
    padded.extend((0..padding_needed).map(|_| Vec3::new(0.0, 0.0, 0.0)));

    // Now `padded.len()` is a multiple of 8, so chunks_exact(8) will consume it fully.
    padded
        .chunks_exact(8)
        .map(|chunk| {
            // Convert the slice chunk into an array of 8 Vec3 elements.
            let arr: [Vec3; 8] = chunk.try_into().unwrap();
            Vec3S::new(arr)
        })
        .collect()
}

/// Unpack an array of SIMD `Vec3S` to normal `Vec3`. The result
/// will have approximately 8x as many elements as the input.
pub fn simd_to_vec3s(vecs: &[Vec3S]) -> Vec<Vec3> {
    let mut result = Vec::new();
    for vec in vecs {
        result.extend(&vec.unpack());
    }
    result
}

/// Convert a slice of `f32` to an array of SIMD `f32` values, 8-wide. Padded as required. The result
/// will have approximately 8x fewer elements than the input.
pub fn f32s_to_simd(vals: &[f32]) -> Vec<__m256> {
    let remainder = vals.len() % 8;
    let padding_needed = if remainder == 0 { 0 } else { 8 - remainder };

    let mut padded = Vec::with_capacity(vals.len() + padding_needed);
    padded.extend_from_slice(vals);
    padded.extend((0..padding_needed).map(|_| 0.0f32));

    // Now `padded.len()` is a multiple of 8, so chunks_exact(8) will consume it fully.
    padded
        .chunks_exact(8)
        .map(|chunk| unsafe { _mm256_loadu_ps(chunk.as_ptr()) })
        .collect()
}

/// Convert a slice of SIMD `f32` values, 8-wide to an Vec of `f32`. The result
/// will have approximately 8x as many elements as the input.
pub fn simd_to_f32s(vals: &[__m256]) -> Vec<f32> {
    let mut result = Vec::new();
    for val in vals {
        let vals_f32: [f32; 8] = unsafe { transmute(*val) };
        result.extend(&vals_f32);
    }
    result
}
