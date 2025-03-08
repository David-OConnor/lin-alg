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
        _mm256_set_ps, _mm256_set1_ps, _mm256_setzero_ps, _mm256_sqrt_ps, _mm256_storeu_ps,
        _mm256_sub_ps,
    },
    mem::transmute,
    ops::{Add, DivAssign, Mul, MulAssign, Neg, Sub},
};

use crate::f32::Vec3;
// You could also implement Mul<f32> similarly, scaling all lanes by a constant

/// SoA. Performs operations on 8 Vecs.
#[derive(Clone, Copy, Debug)]
pub struct Vec3sF32 {
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
pub struct Vec4sF32 {
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

impl Add for Vec3sF32 {
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

impl Sub for Vec3sF32 {
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

impl Neg for Vec3sF32 {
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

// Scalar multiplication
impl Mul<f32> for Vec3sF32 {
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

impl Add for Vec4sF32 {
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

impl Sub for Vec4sF32 {
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

impl Neg for Vec4sF32 {
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

// Scalar multiplication
impl Mul<f32> for Vec4sF32 {
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

/// MulAssign by scalar
impl MulAssign<f32> for Vec3sF32 {
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
impl DivAssign<f32> for Vec3sF32 {
    fn div_assign(&mut self, rhs: f32) {
        unsafe {
            let s = _mm256_set1_ps(rhs);
            self.x = _mm256_div_ps(self.x, s);
            self.y = _mm256_div_ps(self.y, s);
            self.z = _mm256_div_ps(self.z, s);
        }
    }
}

impl Vec3sF32 {
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

    /// Convert the SoA data back into an array of eight Vec3's.
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

    /// Lane-wise cross product
    /// cross.x = self.y * rhs.z - self.z * rhs.y
    /// cross.y = self.z * rhs.x - self.x * rhs.z
    /// cross.z = self.x * rhs.y - self.y * rhs.x
    pub fn cross(self, rhs: Self) -> Self {
        unsafe {
            let yz = _mm256_sub_ps(_mm256_mul_ps(self.y, rhs.z), _mm256_mul_ps(self.z, rhs.y)); // x-lane
            let zx = _mm256_sub_ps(_mm256_mul_ps(self.z, rhs.x), _mm256_mul_ps(self.x, rhs.z)); // y-lane
            let xy = _mm256_sub_ps(_mm256_mul_ps(self.x, rhs.y), _mm256_mul_ps(self.y, rhs.x)); // z-lane
            Vec3sF32 {
                x: yz,
                y: zx,
                z: xy,
            }
        }
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
}

impl Vec4sF32 {
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
