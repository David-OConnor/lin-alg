#![allow(non_camel_case_types)]
#![allow(clippy::not_unsafe_ptr_arg_deref)]

//! Fundamental SIMD floating point types. Placeholder until `core::simd` is in standard.

use std::{
    arch::x86_64::*,
    iter::Sum,
    mem::transmute,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
/// Similar to `core::simd`.
pub struct f32x8(__m256);

#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
/// Similar to `core::simd`.
pub struct f32x16(__m512);

#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
/// Similar to `core::simd`.
pub struct f64x4(__m256d);

#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
/// Similar to `core::simd`.
pub struct f64x8(__m512d);

impl f32x8 {
    pub fn from_slice(slice: &[f32]) -> Self {
        Self::load(slice.as_ptr())
    }

    pub fn from_array(arr: [f32; 8]) -> Self {
        Self::load(arr.as_ptr())
    }

    pub fn to_array(&self) -> [f32; 8] {
        unsafe { transmute(self.0) }
    }

    pub fn load(ptr: *const f32) -> Self {
        unsafe { Self(_mm256_loadu_ps(ptr)) }
    }

    pub fn splat(val: f32) -> Self {
        unsafe { Self(_mm256_set1_ps(val)) }
    }

    pub fn sqrt(&self) -> Self {
        unsafe { Self(_mm256_sqrt_ps(self.0)) }
    }

    // todo: Implement f64 variants of exp
    pub fn exp(&self) -> Self {
        unsafe {
            let max = _mm256_set1_ps(88.3762626647949_f32);
            let min = _mm256_set1_ps(-87.3365447502136_f32);
            let log2e = _mm256_set1_ps(1.4426950408889634_f32);
            let ln2_hi = _mm256_set1_ps(0.693359375_f32);
            let ln2_lo = _mm256_set1_ps(-2.12194440e-4_f32);

            let c1 = _mm256_set1_ps(1.9875691500e-4_f32);
            let c2 = _mm256_set1_ps(1.3981999507e-3_f32);
            let c3 = _mm256_set1_ps(8.3334519073e-3_f32);
            let c4 = _mm256_set1_ps(4.1665795894e-2_f32);
            let c5 = _mm256_set1_ps(1.6666665459e-1_f32);
            let c6 = _mm256_set1_ps(5.0000001201e-1_f32);

            let x = _mm256_max_ps(_mm256_min_ps(self.0, max), min);

            let y = _mm256_fmadd_ps(x, log2e, _mm256_set1_ps(0.5));
            let y_floor = _mm256_round_ps(y, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
            let n_i = _mm256_cvtps_epi32(y_floor);
            let n_f = _mm256_cvtepi32_ps(n_i);

            let r = _mm256_fnmadd_ps(n_f, ln2_hi, x);
            let r = _mm256_fnmadd_ps(n_f, ln2_lo, r);

            let mut p = _mm256_fmadd_ps(c1, r, c2);
            p = _mm256_fmadd_ps(p, r, c3);
            p = _mm256_fmadd_ps(p, r, c4);
            p = _mm256_fmadd_ps(p, r, c5);
            p = _mm256_fmadd_ps(p, r, c6);
            p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0));
            p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0));

            let bias = _mm256_set1_epi32(127);
            let e = _mm256_slli_epi32::<23>(_mm256_add_epi32(n_i, bias));
            let two_n = _mm256_castsi256_ps(e);

            Self(_mm256_mul_ps(p, two_n))
        }
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
                result *= base;
            }
            base = base * base;
            n /= 2;
        }
        result
    }

    /// Lane‑wise “less than” comparison.

    pub fn lt(self, other: Self) -> f32x8Mask {
        // Use the new const-generic intrinsic.
        unsafe { f32x8Mask(_mm256_cmp_ps::<{ _CMP_LT_OQ }>(self.0, other.0)) }
    }

    /// Lane‑wise “greater than” comparison.

    pub fn gt(self, other: Self) -> f32x8Mask {
        unsafe { f32x8Mask(_mm256_cmp_ps::<{ _CMP_GT_OQ }>(self.0, other.0)) }
    }

    /// Lane‑wise “equal to” comparison.

    pub fn eq(self, other: Self) -> f32x8Mask {
        unsafe { f32x8Mask(_mm256_cmp_ps::<{ _CMP_EQ_OQ }>(self.0, other.0)) }
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

    fn add(self, rhs: Self) -> Self::Output {
        unsafe { Self(_mm256_add_ps(self.0, rhs.0)) }
    }
}

impl Sub for f32x8 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        unsafe { Self(_mm256_sub_ps(self.0, rhs.0)) }
    }
}

impl Mul for f32x8 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        unsafe { Self(_mm256_mul_ps(self.0, rhs.0)) }
    }
}

impl Div for f32x8 {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        unsafe { Self(_mm256_div_ps(self.0, rhs.0)) }
    }
}

impl AddAssign for f32x8 {
    fn add_assign(&mut self, rhs: Self) {
        unsafe { *self = Self(_mm256_add_ps(self.0, rhs.0)) }
    }
}

impl SubAssign for f32x8 {
    fn sub_assign(&mut self, rhs: Self) {
        unsafe { *self = Self(_mm256_sub_ps(self.0, rhs.0)) }
    }
}

impl MulAssign for f32x8 {
    fn mul_assign(&mut self, rhs: Self) {
        unsafe { *self = Self(_mm256_mul_ps(self.0, rhs.0)) }
    }
}

impl DivAssign for f32x8 {
    fn div_assign(&mut self, rhs: Self) {
        unsafe { *self = Self(_mm256_div_ps(self.0, rhs.0)) }
    }
}

impl Sum for f32x8 {
    /// Sum up a sequence of `f32x8` values by lane.
    fn sum<I: Iterator<Item = f32x8>>(iter: I) -> Self {
        iter.fold(Self::splat(0.0), |acc, x| acc + x)
    }
}

impl<'a> Sum<&'a f32x8> for f32x8 {
    /// Sum up a sequence of `&f32x8` by lane.
    fn sum<I: Iterator<Item = &'a f32x8>>(iter: I) -> Self {
        iter.fold(Self::splat(0.0), |acc, &x| acc + x)
    }
}

impl f64x4 {
    pub fn from_slice(slice: &[f64]) -> Self {
        Self::load(slice.as_ptr())
    }

    pub fn from_array(arr: [f64; 4]) -> Self {
        Self::load(arr.as_ptr())
    }

    pub fn to_array(&self) -> [f64; 4] {
        unsafe { transmute(self.0) }
    }

    pub fn load(ptr: *const f64) -> Self {
        unsafe { Self(_mm256_loadu_pd(ptr)) }
    }

    pub fn splat(val: f64) -> Self {
        unsafe { Self(_mm256_set1_pd(val)) }
    }

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
                result *= base;
            }
            base = base * base;
            n /= 2;
        }
        result
    }

    /// Lane‑wise “less than” comparison.

    pub fn lt(self, other: Self) -> f64x4Mask {
        // Use the new const-generic intrinsic.
        unsafe { f64x4Mask(_mm256_cmp_pd::<{ _CMP_LT_OQ }>(self.0, other.0)) }
    }

    /// Lane‑wise “greater than” comparison.

    pub fn gt(self, other: Self) -> f64x4Mask {
        unsafe { f64x4Mask(_mm256_cmp_pd::<{ _CMP_GT_OQ }>(self.0, other.0)) }
    }

    /// Lane‑wise “equal to” comparison.

    pub fn eq(self, other: Self) -> f64x4Mask {
        unsafe { f64x4Mask(_mm256_cmp_pd::<{ _CMP_EQ_OQ }>(self.0, other.0)) }
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

    fn add(self, rhs: Self) -> Self::Output {
        unsafe { Self(_mm256_add_pd(self.0, rhs.0)) }
    }
}

impl Sub for f64x4 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        unsafe { Self(_mm256_sub_pd(self.0, rhs.0)) }
    }
}

impl Mul for f64x4 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        unsafe { Self(_mm256_mul_pd(self.0, rhs.0)) }
    }
}

impl Div for f64x4 {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        unsafe { Self(_mm256_div_pd(self.0, rhs.0)) }
    }
}

impl AddAssign for f64x4 {
    fn add_assign(&mut self, rhs: Self) {
        unsafe { *self = Self(_mm256_add_pd(self.0, rhs.0)) }
    }
}

impl SubAssign for f64x4 {
    fn sub_assign(&mut self, rhs: Self) {
        unsafe { *self = Self(_mm256_sub_pd(self.0, rhs.0)) }
    }
}

impl MulAssign for f64x4 {
    fn mul_assign(&mut self, rhs: Self) {
        unsafe { *self = Self(_mm256_mul_pd(self.0, rhs.0)) }
    }
}

impl DivAssign for f64x4 {
    fn div_assign(&mut self, rhs: Self) {
        unsafe { *self = Self(_mm256_div_pd(self.0, rhs.0)) }
    }
}

impl f32x16 {
    pub fn from_slice(slice: &[f32]) -> Self {
        Self::load(slice.as_ptr())
    }

    pub fn from_array(arr: [f32; 16]) -> Self {
        Self::load(arr.as_ptr())
    }

    pub fn to_array(&self) -> [f32; 16] {
        unsafe { transmute(self.0) }
    }

    pub fn load(ptr: *const f32) -> Self {
        unsafe { Self(_mm512_loadu_ps(ptr)) }
    }

    pub fn splat(val: f32) -> Self {
        unsafe { Self(_mm512_set1_ps(val)) }
    }

    pub fn sqrt(&self) -> Self {
        unsafe { Self(_mm512_sqrt_ps(self.0)) }
    }

    // todo: f64 variants of exp.
    pub fn exp(&self) -> Self {
        unsafe {
            let max = _mm512_set1_ps(88.3762626647949_f32);
            let min = _mm512_set1_ps(-87.3365447502136_f32);
            let log2e = _mm512_set1_ps(1.4426950408889634_f32);
            let ln2_hi = _mm512_set1_ps(0.693359375_f32);
            let ln2_lo = _mm512_set1_ps(-2.12194440e-4_f32);

            let c1 = _mm512_set1_ps(1.9875691500e-4_f32);
            let c2 = _mm512_set1_ps(1.3981999507e-3_f32);
            let c3 = _mm512_set1_ps(8.3334519073e-3_f32);
            let c4 = _mm512_set1_ps(4.1665795894e-2_f32);
            let c5 = _mm512_set1_ps(1.6666665459e-1_f32);
            let c6 = _mm512_set1_ps(5.0000001201e-1_f32);

            let x = _mm512_max_ps(_mm512_min_ps(self.0, max), min);

            let y = _mm512_fmadd_ps(x, log2e, _mm512_set1_ps(0.5));
            let y_floor = _mm512_roundscale_ps::<0x9>(y); // floor, no-exc
            let n_i = _mm512_cvtps_epi32(y_floor);
            let n_f = _mm512_cvtepi32_ps(n_i);

            let r = _mm512_fnmadd_ps(n_f, ln2_hi, x);
            let r = _mm512_fnmadd_ps(n_f, ln2_lo, r);

            let mut p = _mm512_fmadd_ps(c1, r, c2);
            p = _mm512_fmadd_ps(p, r, c3);
            p = _mm512_fmadd_ps(p, r, c4);
            p = _mm512_fmadd_ps(p, r, c5);
            p = _mm512_fmadd_ps(p, r, c6);
            p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.0));
            p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.0));

            let bias = _mm512_set1_epi32(127);
            let e = _mm512_slli_epi32::<23>(_mm512_add_epi32(n_i, bias));
            let two_n = _mm512_castsi512_ps(e);

            Self(_mm512_mul_ps(p, two_n))
        }
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

impl Neg for f32x16 {
    type Output = Self;

    fn neg(self) -> Self::Output {
        f32x16::splat(0.) - self
    }
}

impl Add for f32x16 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        unsafe { Self(_mm512_add_ps(self.0, rhs.0)) }
    }
}

impl Sub for f32x16 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        unsafe { Self(_mm512_sub_ps(self.0, rhs.0)) }
    }
}

impl Mul for f32x16 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        unsafe { Self(_mm512_mul_ps(self.0, rhs.0)) }
    }
}

impl Div for f32x16 {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        unsafe { Self(_mm512_div_ps(self.0, rhs.0)) }
    }
}

impl AddAssign for f32x16 {
    fn add_assign(&mut self, rhs: Self) {
        unsafe { *self = Self(_mm512_add_ps(self.0, rhs.0)) }
    }
}

impl SubAssign for f32x16 {
    fn sub_assign(&mut self, rhs: Self) {
        unsafe { *self = Self(_mm512_sub_ps(self.0, rhs.0)) }
    }
}

impl MulAssign for f32x16 {
    fn mul_assign(&mut self, rhs: Self) {
        unsafe { *self = Self(_mm512_mul_ps(self.0, rhs.0)) }
    }
}

impl DivAssign for f32x16 {
    fn div_assign(&mut self, rhs: Self) {
        unsafe { *self = Self(_mm512_div_ps(self.0, rhs.0)) }
    }
}

impl f64x8 {
    pub fn from_slice(slice: &[f64]) -> Self {
        Self::load(slice.as_ptr())
    }

    pub fn from_array(arr: [f64; 8]) -> Self {
        Self::load(arr.as_ptr())
    }

    pub fn to_array(&self) -> [f64; 8] {
        unsafe { transmute(self.0) }
    }

    pub fn load(ptr: *const f64) -> Self {
        unsafe { Self(_mm512_loadu_pd(ptr)) }
    }

    pub fn splat(val: f64) -> Self {
        unsafe { Self(_mm512_set1_pd(val)) }
    }

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
                result *= base;
            }
            base = base * base;
            n /= 2;
        }
        result
    }

    // todo: You need powf too.
}

impl Neg for f64x8 {
    type Output = Self;

    fn neg(self) -> Self::Output {
        f64x8::splat(0.) - self
    }
}

impl Add for f64x8 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        unsafe { Self(_mm512_add_pd(self.0, rhs.0)) }
    }
}

impl Sub for f64x8 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        unsafe { Self(_mm512_sub_pd(self.0, rhs.0)) }
    }
}

impl Mul for f64x8 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        unsafe { Self(_mm512_mul_pd(self.0, rhs.0)) }
    }
}

impl Div for f64x8 {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        unsafe { Self(_mm512_div_pd(self.0, rhs.0)) }
    }
}

impl AddAssign for f64x8 {
    fn add_assign(&mut self, rhs: Self) {
        unsafe { *self = Self(_mm512_add_pd(self.0, rhs.0)) }
    }
}

impl SubAssign for f64x8 {
    fn sub_assign(&mut self, rhs: Self) {
        unsafe { *self = Self(_mm512_sub_pd(self.0, rhs.0)) }
    }
}

impl MulAssign for f64x8 {
    fn mul_assign(&mut self, rhs: Self) {
        unsafe { *self = Self(_mm512_mul_pd(self.0, rhs.0)) }
    }
}

impl DivAssign for f64x8 {
    fn div_assign(&mut self, rhs: Self) {
        unsafe { *self = Self(_mm512_div_pd(self.0, rhs.0)) }
    }
}

/// A mask type for 8 lanes of f32 values. For compare operations.
#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub struct f32x8Mask(__m256);

impl f32x8Mask {
    /// Returns true if any lane is true.

    pub fn any(self) -> bool {
        unsafe { _mm256_movemask_ps(self.0) != 0 }
    }

    /// Returns true if all lanes are true.

    pub fn all(self) -> bool {
        unsafe { _mm256_movemask_ps(self.0) == 0xFF }
    }
}

/// A mask type for 4 lanes of f64 values. For compare operations.
#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub struct f64x4Mask(__m256d);

impl f64x4Mask {
    /// Returns true if any lane is true.

    pub fn any(self) -> bool {
        unsafe { _mm256_movemask_pd(self.0) != 0 }
    }

    /// Returns true if all lanes are true.

    pub fn all(self) -> bool {
        unsafe { _mm256_movemask_pd(self.0) == 0xFF }
    }
}
