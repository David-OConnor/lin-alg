#![cfg_attr(not(feature = "std"), no_std)]
#![allow(clippy::suspicious_arithmetic_impl)] // E.g. quaternion division.

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
mod simd;

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "std"))]
mod simd_primitives;

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "std"))]
pub use simd::pack_slice;

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
            iter::Sum,
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
        pub const Z_VEC: Vec3 = Vec3 {
            x: 0.,
            y: 0.,
            z: 1.,
        };

        // This is "forward" if Z is up, and "up" if Y is up.
        pub const Y_VEC: Vec3 = Vec3 {
            x: 0.,
            y: 1.,
            z: 0.,
        };

        pub const X_VEC: Vec3 = Vec3 {
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
    #[cfg(feature = "cuda")]
    use std::sync::Arc;

    #[cfg(feature = "cuda")]
    use cudarc::driver::{CudaSlice, CudaStream};

    use super::f64;
    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "std"))]
    pub use crate::{
        simd::*,
        simd_primitives::{f32x8, f32x16},
    };

    create!(f32);

    create_vec!(f32);
    create_vec_shared!(f32, Vec3, Vec4);

    create_quaternion!(f32);
    create_quaternion_shared!(f32, Vec3, Quaternion);

    create_matrix!(f32);

    // 256-bit Vec and Quaternion
    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "std"))]
    create_vec_shared!(f32x8, Vec3x8, Vec4x8);
    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "std"))]
    create_quaternion_shared!(f32x8, Vec3x8, Quaternionx8);

    // 512-bit Vec and Quaternion
    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "std"))]
    create_vec_shared!(f32x16, Vec3x16, Vec4x16);
    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "std"))]
    create_quaternion_shared!(f32x16, Vec3x16, Quaternionx16);

    // Primitives (256-bit and 512-bit)
    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "std"))]
    create_simd!(
        f32,
        f32x8,
        Vec3x8,
        vec3x8,
        Vec4x8,
        Quaternionx8,
        quaternionx8,
        8
    );
    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "std"))]
    create_simd!(
        f32,
        f32x16,
        Vec3x16,
        vec3x16,
        Vec4x16,
        Quaternionx16,
        quaternionx16,
        16
    );

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

        pub fn to_le_bytes(&self) -> [u8; 3 * 4] {
            let mut result = [0; 3 * 4];

            result[0..4].clone_from_slice(&self.x.to_le_bytes());
            result[4..8].clone_from_slice(&self.y.to_le_bytes());
            result[8..12].clone_from_slice(&self.z.to_le_bytes());

            result
        }

        pub fn from_le_bytes(v: &[u8]) -> Self {
            Self {
                x: f32::from_le_bytes(v[0..4].try_into().unwrap()),
                y: f32::from_le_bytes(v[4..8].try_into().unwrap()),
                z: f32::from_le_bytes(v[8..12].try_into().unwrap()),
            }
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
        pub fn to_bytes(&self) -> [u8; 9 * 4] {
            let mut result = [0; 9 * 4];

            for i in 0..self.data.len() {
                result[i * 4..i * 4 + 4].clone_from_slice(&self.data[i].to_ne_bytes());
            }

            result
        }
    }

    impl Mat4 {
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
    #[cfg(feature = "cuda")]
    use std::sync::Arc;

    #[cfg(feature = "cuda")]
    use cudarc::driver::{CudaSlice, CudaStream};

    use super::f32;
    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "std"))]
    pub use crate::{
        simd::*,
        simd_primitives::{f64x4, f64x8},
    };

    create!(f64);

    create_vec!(f64);
    create_vec_shared!(f64, Vec3, Vec4);

    create_quaternion!(f64);
    create_quaternion_shared!(f64, Vec3, Quaternion);

    create_matrix!(f64);

    // 256-bit Vec and Quaternion
    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "std"))]
    create_vec_shared!(f64x4, Vec3x4, Vec4x4);
    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "std"))]
    create_quaternion_shared!(f64x4, Vec3x4, Quaternionx4);

    // 512-bit Vec and Quaternion
    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "std"))]
    create_vec_shared!(f64x8, Vec3x8, Vec4x8);
    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "std"))]
    create_quaternion_shared!(f64x8, Vec3x8, Quaternionx8);

    // Primitives (256-bit and 512-bit)
    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "std"))]
    create_simd!(
        f64,
        f64x4,
        Vec3x4,
        vec3x4,
        Vec4x4,
        Quaternionx4,
        quaternionx4,
        4
    );
    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "std"))]
    create_simd!(
        f64,
        f64x8,
        Vec3x8,
        vec3x8,
        Vec4x8,
        Quaternionx8,
        quaternionx8,
        8
    );

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

    impl Vec3 {
        /// Convert to a LE byte array, e.g. for sending to a GPU.
        pub fn to_le_bytes(&self) -> [u8; 3 * 8] {
            let mut result = [0; 3 * 8];

            result[0..8].clone_from_slice(&self.x.to_ne_bytes());
            result[8..16].clone_from_slice(&self.y.to_ne_bytes());
            result[16..24].clone_from_slice(&self.z.to_ne_bytes());

            result
        }

        /// Convert to a LE byte array, e.g. for sending to a GPU.
        pub fn from_le_bytes(v: &[u8]) -> Self {
            Self {
                x: f64::from_le_bytes(v[0..8].try_into().unwrap()),
                y: f64::from_le_bytes(v[8..16].try_into().unwrap()),
                z: f64::from_le_bytes(v[16..24].try_into().unwrap()),
            }
        }
    }
}
