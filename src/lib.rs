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
mod vec;

// #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), target_feature = "avx"))]
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "std"))]
pub mod simd;

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
    create_vec!(f32);
    create_quaternion!(f32);
    create_matrix!(f32);

    use super::f64;
    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "std"))]
    pub use crate::simd::{Vec3S, Vec4S};

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
    create_vec!(f64);
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
