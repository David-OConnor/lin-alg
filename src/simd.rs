#![allow(unused)] // todo temp

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



use std::arch::x86_64::{__m128, __m256d, _mm_add_ps, _mm_loadu_ps, _mm_mul_ps, _mm_set_ps, _mm_set1_ps, _mm_storeu_ps, _mm_sub_ps, _mm_setzero_ps};
use std::ops::{Add, Mul, MulAssign, Neg, Sub};
// You could also implement Mul<f32> similarly, scaling all lanes by a constant

/// SoA: 3 f32 types. Final position is 0.
#[derive(Clone, Debug)]
struct Vec3sF32 {
    x: __m128, // 4 x-values
    y: __m128, // ...
    z: __m128,
}

/// Soa: 4 f64 types. Final position is 0.
#[derive(Clone, Debug)]
struct Vec3sF64 {
    x: __m256d, // 4 x-values
    y: __m256d, // ...
    z: __m256d,
}

/// SoA: 4 f32 types
#[derive(Clone, Debug)]
struct Vec4sF32 {
    x: __m128, // 4 x-values
    y: __m128, // ...
    z: __m128,
    w: __m128,
}

/// Soa: 4 f64 types
#[derive(Clone, Debug)]
struct Vec4sF64 {
    x: __m256d, // 4 x-values
    y: __m256d, // ...
    z: __m256d,
    w: __m256d,
}

impl Add for Vec3sF32 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        unsafe {
            Self {
                x: _mm_add_ps(self.x, rhs.x),
                y: _mm_add_ps(self.y, rhs.y),
                z: _mm_add_ps(self.z, rhs.z),
            }
        }
    }
}

impl Sub for Vec3sF32 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        unsafe {
            Self {
                x: _mm_sub_ps(self.x, rhs.x),
                y: _mm_sub_ps(self.y, rhs.y),
                z: _mm_sub_ps(self.z, rhs.z),
            }
        }
    }
}

impl Neg for Vec3sF32 {
    type Output = Self;
    fn neg(self) -> Self::Output {
        unsafe {
        let zero = _mm_setzero_ps() ;
            Self {
                x: _mm_sub_ps(zero, self.x),
                y: _mm_sub_ps(zero, self.y),
                z: _mm_sub_ps(zero, self.z),
            }
        }
    }
}

// Scalar multiplication
impl Mul<f32> for Vec3sF32 {
    type Output = Self;
    fn mul(self, rhs: f32) -> Self::Output {
        unsafe {
        let s = _mm_set1_ps(rhs) ;
            Self {
                x: _mm_mul_ps(self.x, s),
                y: _mm_mul_ps(self.y, s),
                z: _mm_mul_ps(self.z, s),
            }
        }
    }
}

impl Vec3sF32 {
    /// Dot product across x, y, z lanes. Each lane result is x_i*x_j + y_i*y_j + z_i*z_j.
    /// Returns an __m128 of 4 dot products (one for each lane).
    pub fn dot(self, rhs: Self) -> __m128 {
        unsafe {
            let mut r = _mm_mul_ps(self.x, rhs.x);
            r = _mm_add_ps(r, _mm_mul_ps(self.y, rhs.y));
            r = _mm_add_ps(r, _mm_mul_ps(self.z, rhs.z));
            r
        }
    }
}