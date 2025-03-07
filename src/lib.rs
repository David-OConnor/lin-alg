#![cfg_attr(not(feature = "std"), no_std)]

//! Vector, matrix, and quaternion data structures and operations.
//!
//! Module for matrices, vectors, and quaternions, as used in 3D graphics, geometry,
//! robitics, and spacial embedded systems. Similar to the
//! `cgmath` and `glam` crates, but with a more transparent API.
//! [This elegant lib](https://github.com/MartinWeigel/Quaternion/blob/master/Quaternion.c)
//! may also be used as a reference for quaternion operations.
//!
//! Quaternion operations use the Hamilton (vice JPL) convention.

pub mod complex_nums;
mod matrix;
mod quaternion;
mod util;

#[cfg(feature = "std")]
mod simd;

#[cfg(test)]
mod tests;

pub use util::*;

#[derive(Debug)]
pub struct BufError {}

#[cfg(feature = "std")]
impl From<std::io::Error> for BufError {
    fn from(_: std::io::Error) -> Self {
        Self {}
    }
}

macro_rules! create {
    ($f:ident) => {
        use core::{
            ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
            $f::consts::TAU,
        };
        #[cfg(feature = "std")]
        use std::fmt;

        #[cfg(feature = "encode")]
        use bincode::{Decode, Encode};
        #[cfg(feature = "no_std")]
        use num_traits::float::Float;

        // This is "up" if Z is up, and "forward" if Y is up.
        pub const UP: Vec3 = Vec3 {
            x: 0.,
            y: 0.,
            z: 1.,
        };

        // This is "forward" if Z is up, and "up" if Y is up.
        pub const FORWARD: Vec3 = Vec3 {
            x: 0.,
            y: 1.,
            z: 0.,
        };

        pub const RIGHT: Vec3 = Vec3 {
            x: 1.,
            y: 0.,
            z: 0.,
        };

        /// A len-2 column vector.
        #[derive(Default, Clone, Copy)]
        #[cfg_attr(feature = "encode", derive(Encode, Decode))]
        pub struct Vec2 {
            pub x: $f,
            pub y: $f,
        }

        impl Vec2 {
            pub fn new(x: $f, y: $f) -> Self {
                Self { x, y }
            }

            pub fn magnitude(&self) -> $f {
                (self.x.powi(2) + self.y.powi(2)).sqrt()
            }

            /// Radians, CW from north.
            pub fn track(&self) -> $f {
                self.x.atan2(self.y)
            }
        }

        #[cfg(feature = "std")]
        impl fmt::Display for Vec2 {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                write!(f, "|{:.4}, {:.4}|", self.x, self.y)?;
                Ok(())
            }
        }

        #[derive(Clone, Copy, Default, Debug, PartialEq)]
        #[cfg_attr(feature = "encode", derive(Encode, Decode))]
        /// A len-3 column vector.
        pub struct Vec3 {
            pub x: $f,
            pub y: $f,
            pub z: $f,
        }

        impl From<[$f; 3]> for Vec3 {
            fn from(v: [$f; 3]) -> Self {
                Self {
                    x: v[0],
                    y: v[1],
                    z: v[2],
                }
            }
        }

        impl Add<Self> for Vec3 {
            type Output = Self;

            fn add(self, rhs: Self) -> Self::Output {
                Self {
                    x: self.x + rhs.x,
                    y: self.y + rhs.y,
                    z: self.z + rhs.z,
                }
            }
        }

        impl AddAssign<Self> for Vec3 {
            fn add_assign(&mut self, rhs: Self) {
                self.x = self.x + rhs.x;
                self.y = self.y + rhs.y;
                self.z = self.z + rhs.z;
            }
        }

        impl Sub<Self> for Vec3 {
            type Output = Self;

            fn sub(self, rhs: Self) -> Self::Output {
                Self {
                    x: self.x - rhs.x,
                    y: self.y - rhs.y,
                    z: self.z - rhs.z,
                }
            }
        }

        impl SubAssign<Self> for Vec3 {
            fn sub_assign(&mut self, rhs: Self) {
                self.x = self.x - rhs.x;
                self.y = self.y - rhs.y;
                self.z = self.z - rhs.z;
            }
        }

        impl Mul<$f> for Vec3 {
            type Output = Self;

            fn mul(self, rhs: $f) -> Self::Output {
                Self {
                    x: self.x * rhs,
                    y: self.y * rhs,
                    z: self.z * rhs,
                }
            }
        }

        impl MulAssign<$f> for Vec3 {
            fn mul_assign(&mut self, rhs: $f) {
                self.x *= rhs;
                self.y *= rhs;
                self.z *= rhs;
            }
        }

        impl Div<$f> for Vec3 {
            type Output = Self;

            fn div(self, rhs: $f) -> Self::Output {
                Self {
                    x: self.x / rhs,
                    y: self.y / rhs,
                    z: self.z / rhs,
                }
            }
        }

        impl DivAssign<$f> for Vec3 {
            fn div_assign(&mut self, rhs: $f) {
                self.x /= rhs;
                self.y /= rhs;
                self.z /= rhs;
            }
        }

        impl Neg for Vec3 {
            type Output = Self;

            fn neg(self) -> Self::Output {
                Self {
                    x: -self.x,
                    y: -self.y,
                    z: -self.z,
                }
            }
        }

        impl Vec3 {
            pub const fn new(x: $f, y: $f, z: $f) -> Self {
                Self { x, y, z }
            }

            pub const fn new_zero() -> Self {
                Self {
                    x: 0.,
                    y: 0.,
                    z: 0.,
                }
            }

            /// Construct from the first 3 values in a slice: &[x, y, z].
            pub fn from_slice(slice: &[$f]) -> Result<Self, crate::BufError> {
                if slice.len() < 3 {
                    return Err(crate::BufError {});
                }
                Ok(Self {
                    x: slice[0],
                    y: slice[1],
                    z: slice[2],
                })
            }

            /// Convert to a len-3 array: [x, y, z].
            pub fn to_arr(&self) -> [$f; 3] {
                [self.x, self.y, self.z]
            }

            /// Calculates the Hadamard product (element-wise multiplication).
            pub fn hadamard_product(self, rhs: Self) -> Self {
                Self {
                    x: self.x * rhs.x,
                    y: self.y * rhs.y,
                    z: self.z * rhs.z,
                }
            }

            /// Returns the vector magnitude squared
            pub fn magnitude_squared(self) -> $f {
                (self.hadamard_product(self)).sum()
            }

            /// Returns the vector magnitude
            pub fn magnitude(&self) -> $f {
                (self.x.powi(2) + self.y.powi(2) + self.z.powi(2)).sqrt()
            }

            /// Normalize, modifying in place
            pub fn normalize(&mut self) {
                let mag_recip = 1. / self.magnitude();

                self.x *= mag_recip;
                self.y *= mag_recip;
                self.z *= mag_recip;
            }

            /// Returns the normalised version of the vector
            pub fn to_normalized(self) -> Self {
                let mag_recip = 1. / self.magnitude();
                self * mag_recip
            }

            /// Returns the dot product with another vector.
            pub fn dot(&self, rhs: Self) -> $f {
                self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
            }

            /// Returns a sum of all elements
            pub fn sum(&self) -> $f {
                self.x + self.y + self.z
            }

            /// Calculate the cross product.
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

            /// Projects this vector onto another vector.
            pub fn project_to_vec(self, other: Self) -> Self {
                other * (self.dot(other) / other.magnitude_squared())
            }
        }

        #[cfg(feature = "std")]
        impl fmt::Display for Vec3 {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                write!(f, "|{:.4}, {:.4}, {:.4}|", self.x, self.y, self.z)?;
                Ok(())
            }
        }

        #[derive(Clone, Copy, Debug)]
        #[cfg_attr(feature = "encode", derive(Encode, Decode))]
        /// A len-4 column vector
        pub struct Vec4 {
            pub x: $f,
            pub y: $f,
            pub z: $f,
            pub w: $f,
        }

        impl Vec4 {
            pub fn new(x: $f, y: $f, z: $f, u: $f) -> Self {
                Self { x, y, z, w: u }
            }

            /// Returns the dot product with another vector.
            pub fn dot(&self, rhs: Self) -> $f {
                self.x * rhs.x + self.y * rhs.y + self.z * rhs.z + self.w * rhs.w
            }

            pub fn normalize(&mut self) {
                let len =
                    (self.x.powi(2) + self.y.powi(2) + self.z.powi(2) + self.w.powi(2)).sqrt();

                self.x /= len;
                self.y /= len;
                self.z /= len;
                self.w /= len;
            }

            /// Remove the nth element. Used in our inverse calulations.
            pub fn truncate_n(&self, n: usize) -> Vec3 {
                match n {
                    0 => Vec3::new(self.y, self.z, self.w),
                    1 => Vec3::new(self.x, self.z, self.w),
                    2 => Vec3::new(self.x, self.y, self.w),
                    3 => Vec3::new(self.x, self.y, self.z),
                    _ => panic!("{:?} is out of range", n),
                }
            }

            /// Remove the w element.
            pub fn xyz(&self) -> Vec3 {
                Vec3::new(self.x, self.y, self.z)
            }
        }

        #[cfg(feature = "std")]
        impl fmt::Display for Vec4 {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                write!(
                    f,
                    "|{:.4}, {:.4}, {:.4}, {:.4}|",
                    self.x, self.y, self.z, self.w
                )?;
                Ok(())
            }
        }

        #[derive(Clone, Debug)]
        #[cfg_attr(feature = "encode", derive(Encode, Decode))]
        /// Represents a set of Euler angles.
        pub struct EulerAngle {
            pub roll: $f,
            pub pitch: $f,
            pub yaw: $f,
        }
    };
}

pub mod f32 {
    create!(f32);
    create_quaternion!(f32);
    create_matrix!(f32);

    use super::f64;

    impl From<f64::Vec2> for Vec2 {
        fn from(other: f64::Vec2) -> Self {
            Self {
                x: other.x as f32,
                y: other.y as f32,
            }
        }
    }

    impl From<f64::Vec3> for Vec3 {
        fn from(other: f64::Vec3) -> Self {
            Self {
                x: other.x as f32,
                y: other.y as f32,
                z: other.z as f32,
            }
        }
    }

    impl From<f64::Vec4> for Vec4 {
        fn from(other: f64::Vec4) -> Self {
            Self {
                w: other.w as f32,
                x: other.x as f32,
                y: other.y as f32,
                z: other.z as f32,
            }
        }
    }

    impl From<f64::Quaternion> for Quaternion {
        fn from(other: f64::Quaternion) -> Self {
            Self {
                w: other.w as f32,
                x: other.x as f32,
                y: other.y as f32,
                z: other.z as f32,
            }
        }
    }

    impl From<f64::Mat3> for Mat3 {
        fn from(other: f64::Mat3) -> Self {
            Self {
                data: other.data.map(|x| x as f32),
            }
        }
    }

    impl From<f64::Mat4> for Mat4 {
        fn from(other: f64::Mat4) -> Self {
            Self {
                data: other.data.map(|x| x as f32),
            }
        }
    }

    impl Vec3 {
        #[cfg(feature = "computer_graphics")]
        /// Convert to a byte array, e.g. for sending to a GPU. Note that this function pads with an
        /// extra 4 bytes, IOC with the  hardware
        /// 16-byte alignment requirement. This assumes we're using this in a uniform; Vertexes
        /// don't use padding.
        pub fn to_bytes_uniform(&self) -> [u8; 4 * 4] {
            let mut result = [0; 4 * 4];

            result[0..4].clone_from_slice(&self.x.to_ne_bytes());
            result[4..8].clone_from_slice(&self.y.to_ne_bytes());
            result[8..12].clone_from_slice(&self.z.to_ne_bytes());
            result[12..16].clone_from_slice(&[0; 4]);

            result
        }

        /// Convert to a native-endian byte array, e.g. for sending to a GPU.
        pub fn to_bytes(&self) -> [u8; 3 * 4] {
            let mut result = [0; 3 * 4];

            result[0..4].clone_from_slice(&self.x.to_ne_bytes());
            result[4..8].clone_from_slice(&self.y.to_ne_bytes());
            result[8..12].clone_from_slice(&self.z.to_ne_bytes());

            result
        }
    }

    impl Vec4 {
        /// Convert to a native-endian byte array, e.g. for sending to a GPU. (x, y, z, w order)
        pub fn to_bytes(&self) -> [u8; 4 * 4] {
            let mut result = [0; 4 * 4];

            result[0..4].clone_from_slice(&self.x.to_ne_bytes());
            result[4..8].clone_from_slice(&self.y.to_ne_bytes());
            result[8..12].clone_from_slice(&self.z.to_ne_bytes());
            result[12..16].clone_from_slice(&self.w.to_ne_bytes());

            result
        }
    }

    impl Mat3 {
        #[cfg(feature = "computer_graphics")]
        pub fn to_bytes(&self) -> [u8; 9 * 4] {
            let mut result = [0; 9 * 4];

            for i in 0..self.data.len() {
                result[i * 4..i * 4 + 4].clone_from_slice(&self.data[i].to_ne_bytes());
            }

            result
        }
    }

    impl Mat4 {
        #[cfg(feature = "computer_graphics")]
        pub fn to_bytes(&self) -> [u8; 16 * 4] {
            let mut result = [0; 16 * 4];

            for i in 0..self.data.len() {
                result[i * 4..i * 4 + 4].clone_from_slice(&self.data[i].to_ne_bytes());
            }

            result
        }
    }
}

pub mod f64 {
    create!(f64);
    create_quaternion!(f64);
    create_matrix!(f64);

    use super::f32;

    impl From<f32::Vec2> for Vec2 {
        fn from(other: f32::Vec2) -> Self {
            Self {
                x: other.x as f64,
                y: other.y as f64,
            }
        }
    }

    impl From<f32::Vec3> for Vec3 {
        fn from(other: f32::Vec3) -> Self {
            Self {
                x: other.x as f64,
                y: other.y as f64,
                z: other.z as f64,
            }
        }
    }

    impl From<f32::Vec4> for Vec4 {
        fn from(other: f32::Vec4) -> Self {
            Self {
                w: other.w as f64,
                x: other.x as f64,
                y: other.y as f64,
                z: other.z as f64,
            }
        }
    }

    impl From<f32::Quaternion> for Quaternion {
        fn from(other: f32::Quaternion) -> Self {
            Self {
                w: other.w as f64,
                x: other.x as f64,
                y: other.y as f64,
                z: other.z as f64,
            }
        }
    }

    impl From<f32::Mat3> for Mat3 {
        fn from(other: f32::Mat3) -> Self {
            Self {
                data: other.data.map(|x| x as f64),
            }
        }
    }

    impl From<f32::Mat4> for Mat4 {
        fn from(other: f32::Mat4) -> Self {
            Self {
                data: other.data.map(|x| x as f64),
            }
        }
    }
}
