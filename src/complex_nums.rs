// C+P from WF_LAB

//! This module contains a `Complex` number type, and methods for it.

use std::{
    f64::consts::E,
    fmt,
    ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign},
};

pub const IM: Cplx = Cplx { real: 0., im: 1. };

#[derive(Copy, Clone, Debug, Default)]
pub struct Cplx {
    // todo should probably just use num::Complex
    pub real: f64,
    pub im: f64,
}

impl Cplx {
    pub fn new(real: f64, im: f64) -> Self {
        Self { real, im }
    }

    pub const fn new_zero() -> Self {
        Self { real: 0., im: 0. }
    }

    pub fn conj(&self) -> Self {
        Self {
            real: self.real,
            im: -self.im,
        }
    }

    pub fn mag(&self) -> f64 {
        (self.real.powi(2) + self.im.powi(2)).sqrt()
    }

    pub fn phase(&self) -> f64 {
        (self.im).atan2(self.real)
    }

    /// Convert a real value into a complex number with 0 imaginary part.
    pub fn from_real(val_real: f64) -> Self {
        Self {
            real: val_real,
            im: 0.,
        }
    }

    /// e^this value
    pub fn exp(&self) -> Self {
        // todo: QC this
        (Self::from_real(self.im.cos()) + IM * self.im.sin()) * E.powf(self.real)
    }

    // /// Take a power to an integer
    // pub fn powi(&self, val: u32) -> Self {
    //     // todo: Better way?
    //
    //     let mut result = Self::new(1., 0.);
    //
    //     for _ in 0..val as usize {
    //         result *= *self;
    //     }
    //
    //     result
    // }

    /// Multiply this value's complex conjugate by it. Is a real number.
    pub fn abs_sq(&self) -> f64 {
        (self.conj() * *self).real
        // Another approach:
        // self.real.powi(2) + self.im.powi(2)
    }
}

impl From<f64> for Cplx {
    fn from(real_num: f64) -> Self {
        Self {
            real: real_num,
            im: 0.,
        }
    }
}

impl Add for Cplx {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            real: self.real + other.real,
            im: self.im + other.im,
        }
    }
}

impl AddAssign for Cplx {
    fn add_assign(&mut self, other: Self) {
        self.real = self.real + other.real;
        self.im = self.im + other.im;
    }
}

impl Sub for Cplx {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self {
            real: self.real - other.real,
            im: self.im - other.im,
        }
    }
}

impl SubAssign for Cplx {
    fn sub_assign(&mut self, other: Self) {
        self.real = self.real - other.real;
        self.im = self.im - other.im;
    }
}

impl Mul<Cplx> for Cplx {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        Self {
            real: self.real * other.real - self.im * other.im,
            im: self.real * other.im + self.im * other.real,
        }
    }
}

impl MulAssign<Cplx> for Cplx {
    fn mul_assign(&mut self, other: Self) {
        let result = *self * other;
        self.real = result.real;
        self.im = result.im;
    }
}

impl Mul<f64> for Cplx {
    type Output = Self;

    /// To verify, compare to `Mul<Cplx>`, where `other.im` is 0.
    fn mul(self, other: f64) -> Self {
        Self {
            real: self.real * other,
            im: self.im * other,
        }
    }
}

impl Div<Self> for Cplx {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        Self {
            real: (self.real * other.real + self.im * other.im)
                / (other.real.powi(2) + other.im.powi(2)),
            im: self.im * other.real
                - self.real * other.im / (other.real.powi(2) + other.im.powi(2)),
        }
    }
}

impl Div<f64> for Cplx {
    type Output = Self;

    fn div(self, other: f64) -> Self {
        Self {
            real: self.real / other,
            im: self.im / other,
        }
    }
}

impl Neg for Cplx {
    type Output = Self;

    fn neg(self) -> Self {
        Self {
            real: -self.real,
            im: -self.im,
        }
    }
}

impl fmt::Display for Cplx {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{} + {}i", self.real, self.im)
    }
}
