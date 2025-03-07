// todo: Experimenting
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

#[allow(unused)] // todo temp
use std::arch::x86_64::{__m128, __m256d, _mm_add_ps, _mm_set_ps};
use std::ops::{Add, Mul, Sub};

/// SoA: 4 f32 types
#[derive(Debug)]
struct Vec4Sf32 {
    x: __m128, // 4 x-values
    y: __m128, // ...
    z: __m128,
    w: __m128,
}

/// Soa: 4 f64 types
#[derive(Debug)]
struct Vec4Sf64 {
    x: __m256d, // 4 x-values
    y: __m256d, // ...
    z: __m256d,
    w: __m256d,
}

/// AoS.
/// A single 4D vector, each lane is x, y, z, w
#[repr(C, align(16))]
#[derive(Debug)]
pub struct Vec4Sf32Aos(pub __m128);

/// AoS.
#[repr(C, align(16))]
#[derive(Debug)]
pub struct Vec4f64(pub __m256d);

pub struct Vec3f32(pub __m128);

impl Vec3f32 {
    /// Store x, y, z, then 0.0 in the last lane.
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        unsafe {
            // Lane order: (0.0, z, y, x)
            Vec3f32(_mm_set_ps(0.0, z, y, x))
        }
    }
}

impl Add for Vec3f32 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        unsafe { Vec3f32(_mm_add_ps(self.0, rhs.0)) }
    }
}
