//! Fundamental SIMD floating point types. Placeholder until `core::simd` is in standard.

use std::{
    arch::x86_64::*,
    intrinsics::transmute,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

// todo: Stabilization PR soon on 512 as of 2025-03-22.

#[derive(Copy, Clone, Debug)]
#[allow(non_camel_case_types)]
#[repr(transparent)]
/// Similar to `core::simd`.
pub struct f32x8(__m256);

#[derive(Copy, Clone, Debug)]
#[allow(non_camel_case_types)]
#[repr(transparent)]
/// Similar to `core::simd`.
pub struct f32x16(__m512);

#[derive(Copy, Clone, Debug)]
#[allow(non_camel_case_types)]
#[repr(transparent)]
/// Similar to `core::simd`.
pub struct f64x4(__m256d);

#[derive(Copy, Clone, Debug)]
#[allow(non_camel_case_types)]
#[repr(transparent)]
/// Similar to `core::simd`.
pub struct f64x8(__m512d);

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

impl f64x4 {
    #[inline]
    pub fn from_slice(slice: &[f64]) -> Self {
        Self::load(slice.as_ptr())
    }

    #[inline]
    pub fn from_array(arr: [f64; 4]) -> Self {
        Self::load(arr.as_ptr())
    }

    #[inline]
    pub fn to_array(&self) -> [f64; 4] {
        unsafe { transmute(self.0) }
    }

    #[inline]
    pub fn load(ptr: *const f64) -> Self {
        unsafe { Self(_mm256_loadu_pd(ptr)) }
    }

    #[inline]
    pub fn splat(val: f64) -> Self {
        unsafe { Self(_mm256_set1_pd(val)) }
    }

    #[inline]
    pub fn sqrt(&self) -> Self {
        unsafe { Self(_mm256_sqrt_pd(self.0)) }
    }

    /// todo: This doesn't match the core::simd API. What's the equivalent there?
    /// todo: This is potentially a slow approach compared to using intrinsics
    pub fn replace(self, index: usize, value: f64) -> Self {
        let mut arr = self.to_array();
        // This will panic if index >= 8, similar to core::simd::Simd::replace.
        arr[index] = value;
        Self::from_array(arr)
    }

    pub fn powi(self, mut n: i32) -> Self {
        // todo: QC this one.
        // Handle the zero exponent case: x^0 = 1 for any x.
        if n == 0 {
            return unsafe { Self(_mm256_set1_pd(1.0)) };
        }

        let mut base = self;
        // If the exponent is negative, invert the base and use the positive exponent.
        if n < 0 {
            base = unsafe { Self(_mm256_div_pd(_mm256_set1_pd(1.0), base.0)) };
            n = -n;
        }

        // Initialize result as 1 for each lane.
        let mut result = unsafe { Self(_mm256_set1_pd(1.0)) };

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

impl Neg for f64x4 {
    type Output = Self;

    fn neg(self) -> Self::Output {
        f64x4::splat(0.) - self
    }
}

impl Add for f64x4 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        unsafe { Self(_mm256_add_pd(self.0, rhs.0)) }
    }
}

impl Sub for f64x4 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        unsafe { Self(_mm256_sub_pd(self.0, rhs.0)) }
    }
}

impl Mul for f64x4 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        unsafe { Self(_mm256_mul_pd(self.0, rhs.0)) }
    }
}

impl Div for f64x4 {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        unsafe { Self(_mm256_div_pd(self.0, rhs.0)) }
    }
}

impl AddAssign for f64x4 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        unsafe { *self = Self(_mm256_add_pd(self.0, rhs.0)) }
    }
}

impl SubAssign for f64x4 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        unsafe { *self = Self(_mm256_sub_pd(self.0, rhs.0)) }
    }
}

impl MulAssign for f64x4 {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        unsafe { *self = Self(_mm256_mul_pd(self.0, rhs.0)) }
    }
}

impl DivAssign for f64x4 {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        unsafe { *self = Self(_mm256_div_pd(self.0, rhs.0)) }
    }
}

// todo: Ipml f32x16 and f64x8 once 512 bit is standard.
#[cfg(feature = "nightly")]
impl f32x16 {
    #[inline]
    pub fn from_slice(slice: &[f32]) -> Self {
        Self::load(slice.as_ptr())
    }

    #[inline]
    pub fn from_array(arr: [f32; 16]) -> Self {
        Self::load(arr.as_ptr())
    }

    #[inline]
    pub fn to_array(&self) -> [f32; 16] {
        unsafe { transmute(self.0) }
    }

    #[inline]
    pub fn load(ptr: *const f32) -> Self {
        unsafe { Self(_mm512_loadu_ps(ptr)) }
    }

    #[inline]
    pub fn splat(val: f32) -> Self {
        unsafe { Self(_mm512_set1_ps(val)) }
    }

    #[inline]
    pub fn sqrt(&self) -> Self {
        unsafe { Self(_mm512_sqrt_ps(self.0)) }
    }

    /// todo: This doesn't match the core::simd API. What's the equivalent there?
    /// todo: This is potentially a slow approach compared to using intrinsics
    pub fn replace(self, index: usize, value: f32) -> Self {
        let mut arr = self.to_array();
        // This will panic if index >= 16, similar to core::simd::Simd::replace.
        arr[index] = value;
        Self::from_array(arr)
    }

    pub fn powi(self, mut n: i32) -> Self {
        // todo: QC this one.
        // Handle the zero exponent case: x^0 = 1 for any x.
        if n == 0 {
            return unsafe { Self(_mm512_set1_ps(1.0)) };
        }

        let mut base = self;
        // If the exponent is negative, invert the base and use the positive exponent.
        if n < 0 {
            base = unsafe { Self(_mm512_div_ps(_mm512_set1_ps(1.0), base.0)) };
            n = -n;
        }

        // Initialize result as 1 for each lane.
        let mut result = unsafe { Self(_mm512_set1_ps(1.0)) };

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

#[cfg(feature = "nightly")]
impl Neg for f32x16 {
    type Output = Self;

    fn neg(self) -> Self::Output {
        f32x16::splat(0.) - self
    }
}

#[cfg(feature = "nightly")]
impl Add for f32x16 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        unsafe { Self(_mm512_add_ps(self.0, rhs.0)) }
    }
}

#[cfg(feature = "nightly")]
impl Sub for f32x16 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        unsafe { Self(_mm512_sub_ps(self.0, rhs.0)) }
    }
}

#[cfg(feature = "nightly")]
impl Mul for f32x16 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        unsafe { Self(_mm512_mul_ps(self.0, rhs.0)) }
    }
}

#[cfg(feature = "nightly")]
impl Div for f32x16 {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        unsafe { Self(_mm512_div_ps(self.0, rhs.0)) }
    }
}

#[cfg(feature = "nightly")]
impl AddAssign for f32x16 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        unsafe { *self = Self(_mm512_add_ps(self.0, rhs.0)) }
    }
}

#[cfg(feature = "nightly")]
impl SubAssign for f32x16 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        unsafe { *self = Self(_mm512_sub_ps(self.0, rhs.0)) }
    }
}

#[cfg(feature = "nightly")]
impl MulAssign for f32x16 {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        unsafe { *self = Self(_mm512_mul_ps(self.0, rhs.0)) }
    }
}

#[cfg(feature = "nightly")]
impl DivAssign for f32x16 {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        unsafe { *self = Self(_mm512_div_ps(self.0, rhs.0)) }
    }
}

#[cfg(feature = "nightly")]
impl f64x8 {
    #[inline]
    pub fn from_slice(slice: &[f64]) -> Self {
        Self::load(slice.as_ptr())
    }

    #[inline]
    pub fn from_array(arr: [f64; 8]) -> Self {
        Self::load(arr.as_ptr())
    }

    #[inline]
    pub fn to_array(&self) -> [f64; 8] {
        unsafe { transmute(self.0) }
    }

    #[inline]
    pub fn load(ptr: *const f64) -> Self {
        unsafe { Self(_mm512_loadu_pd(ptr)) }
    }

    #[inline]
    pub fn splat(val: f64) -> Self {
        unsafe { Self(_mm512_set1_pd(val)) }
    }

    #[inline]
    pub fn sqrt(&self) -> Self {
        unsafe { Self(_mm512_sqrt_pd(self.0)) }
    }

    /// todo: This doesn't match the core::simd API. What's the equivalent there?
    /// todo: This is potentially a slow approach compared to using intrinsics
    pub fn replace(self, index: usize, value: f64) -> Self {
        let mut arr = self.to_array();
        // This will panic if index >= 8, similar to core::simd::Simd::replace.
        arr[index] = value;
        Self::from_array(arr)
    }

    pub fn powi(self, mut n: i32) -> Self {
        // todo: QC this one.
        // Handle the zero exponent case: x^0 = 1 for any x.
        if n == 0 {
            return unsafe { Self(_mm512_set1_pd(1.0)) };
        }

        let mut base = self;
        // If the exponent is negative, invert the base and use the positive exponent.
        if n < 0 {
            base = unsafe { Self(_mm512_div_pd(_mm512_set1_pd(1.0), base.0)) };
            n = -n;
        }

        // Initialize result as 1 for each lane.
        let mut result = unsafe { Self(_mm512_set1_pd(1.0)) };

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

#[cfg(feature = "nightly")]
impl Neg for f64x8 {
    type Output = Self;

    fn neg(self) -> Self::Output {
        f64x8::splat(0.) - self
    }
}

#[cfg(feature = "nightly")]
impl Add for f64x8 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        unsafe { Self(_mm512_add_pd(self.0, rhs.0)) }
    }
}

#[cfg(feature = "nightly")]
impl Sub for f64x8 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        unsafe { Self(_mm512_sub_pd(self.0, rhs.0)) }
    }
}

#[cfg(feature = "nightly")]
impl Mul for f64x8 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        unsafe { Self(_mm512_mul_pd(self.0, rhs.0)) }
    }
}

#[cfg(feature = "nightly")]
impl Div for f64x8 {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        unsafe { Self(_mm512_div_pd(self.0, rhs.0)) }
    }
}

#[cfg(feature = "nightly")]
impl AddAssign for f64x8 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        unsafe { *self = Self(_mm512_add_pd(self.0, rhs.0)) }
    }
}

#[cfg(feature = "nightly")]
impl SubAssign for f64x8 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        unsafe { *self = Self(_mm512_sub_pd(self.0, rhs.0)) }
    }
}

#[cfg(feature = "nightly")]
impl MulAssign for f64x8 {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        unsafe { *self = Self(_mm512_mul_pd(self.0, rhs.0)) }
    }
}

#[cfg(feature = "nightly")]
impl DivAssign for f64x8 {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        unsafe { *self = Self(_mm512_div_pd(self.0, rhs.0)) }
    }
}
