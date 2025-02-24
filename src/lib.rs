#![cfg_attr(feature = "no_std", no_std)]

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
mod util;

pub use util::*;

pub struct BufError {}

macro_rules! create {
    ($f:ident) => {
        // Macro start

        use core::{
            ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
            $f::consts::TAU,
        };
        #[cfg(not(feature = "no_std"))]
        use std::fmt;

        #[cfg(feature = "no_std")]
        use num_traits::float::Float;

        #[cfg(feature = "encode")]
        use bincode::{Decode, Encode};

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
        #[cfg_attr(feature = "bincode", derive(Encode, Decode))]
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

        #[cfg(not(feature = "no_std"))]
        impl fmt::Display for Vec2 {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                write!(f, "|{:.4}, {:.4}|", self.x, self.y)?;
                Ok(())
            }
        }

        #[derive(Clone, Copy, Default, Debug, PartialEq)]
        #[cfg_attr(feature = "bincode", derive(Encode, Decode))]
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
                    return Err(crate::BufError {})
                }
                Ok(Self { x: slice[0], y: slice[1], z: slice[2] })
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

            #[cfg(feature = "computer_graphics")]
            /// Note that this function pads with an extra 4 bytes, IOC with the  hardware
            /// 16-byte alignment requirement. This assumes we're using this in a uniform; Vertexes
            /// don't use padding.
            pub fn to_bytes_uniform(&self) -> [u8; 4 * 4] {
                let mut result = [0; 4 * 4];

                result[0..4].clone_from_slice(&self.x.to_ne_bytes());
                result[4..8].clone_from_slice(&self.y.to_ne_bytes());
                result[8..12].clone_from_slice(&self.z.to_ne_bytes());
                result[12..16].clone_from_slice(&[0_u8; 4]);

                result
            }

            #[cfg(feature = "computer_graphics")]
            pub fn to_bytes_vertex(&self) -> [u8; 3 * 4] {
                let mut result = [0; 3 * 4];

                result[0..4].clone_from_slice(&self.x.to_ne_bytes());
                result[4..8].clone_from_slice(&self.y.to_ne_bytes());
                result[8..12].clone_from_slice(&self.z.to_ne_bytes());

                result
            }
        }

        #[cfg(not(feature = "no_std"))]
        impl fmt::Display for Vec3 {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                write!(f, "|{:.4}, {:.4}, {:.4}|", self.x, self.y, self.z)?;
                Ok(())
            }
        }

        #[derive(Clone, Copy, Debug)]
        #[cfg_attr(feature = "bincode", derive(Encode, Decode))]
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

        #[cfg(not(feature = "no_std"))]
        impl fmt::Display for Vec4 {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                write!(f, "|{:.4}, {:.4}, {:.4}, {:.4}|", self.x, self.y, self.z, self.w)?;
                Ok(())
            }
        }

        /// A quaternion using Hamilton (not JPL) transformation conventions. The most common operations
        /// usedful for representing orientations and rotations are defined, including for operations
        /// with `Vec3`.
        #[derive(Clone, Copy, Debug)]
        #[cfg_attr(feature = "bincode", derive(Encode, Decode))]
        pub struct Quaternion {
            pub w: $f,
            pub x: $f,
            pub y: $f,
            pub z: $f,
        }

        impl Default for Quaternion {
            fn default() -> Self {
                Self::new_identity()
            }
        }

        impl Add<Self> for Quaternion {
            type Output = Self;

            fn add(self, rhs: Self) -> Self::Output {
                Self {
                    w: self.w + rhs.w,
                    x: self.x + rhs.x,
                    y: self.y + rhs.y,
                    z: self.z + rhs.z,
                }
            }
        }

        impl Sub<Self> for Quaternion {
            type Output = Self;

            fn sub(self, rhs: Self) -> Self::Output {
                Self {
                    w: self.w - rhs.w,
                    x: self.x - rhs.x,
                    y: self.y - rhs.y,
                    z: self.z - rhs.z,
                }
            }
        }

        impl Mul<Self> for Quaternion {
            type Output = Self;

            fn mul(self, rhs: Self) -> Self::Output {
                Self {
                    w: self.w * rhs.w - self.x * rhs.x - self.y * rhs.y - self.z * rhs.z,
                    x: self.w * rhs.x + self.x * rhs.w + self.y * rhs.z - self.z * rhs.y,
                    y: self.w * rhs.y - self.x * rhs.z + self.y * rhs.w + self.z * rhs.x,
                    z: self.w * rhs.z + self.x * rhs.y - self.y * rhs.x + self.z * rhs.w,
                }
            }
        }

        impl Mul<Vec3> for Quaternion {
            type Output = Self;

            /// Returns the multiplication of a Quaternion with a vector.  This is a
            /// normal Quaternion multiplication where the vector is treated a
            /// Quaternion with a W element value of zero.  The Quaternion is post-
            /// multiplied by the vector.
            fn mul(self, rhs: Vec3) -> Self::Output {
                Self {
                    w: -self.x * rhs.x - self.y * rhs.y - self.z * rhs.z,
                    x: self.w * rhs.x + self.y * rhs.z - self.z * rhs.y,
                    y: self.w * rhs.y - self.x * rhs.z + self.z * rhs.x,
                    z: self.w * rhs.z + self.x * rhs.y - self.y * rhs.x,
                }
            }
        }

        impl Mul<$f> for Quaternion {
            type Output = Self;

            fn mul(self, rhs: $f) -> Self::Output {
                Self {
                    w: self.w * rhs,
                    x: self.x * rhs,
                    y: self.y * rhs,
                    z: self.z * rhs,
                }
            }
        }

        impl Div<Self> for Quaternion {
            type Output = Self;

            fn div(self, rhs: Self) -> Self::Output {
                self * rhs.inverse()
            }
        }

        impl Quaternion {
            pub fn new_identity() -> Self {
                Self {
                    w: 1.,
                    x: 0.,
                    y: 0.,
                    z: 0.,
                }
            }

            pub fn new(w: $f, x: $f, y: $f, z: $f) -> Self {
                Self { w, x, y, z }
            }

            /// Construct from the first 4 values in a slice: &[w, x, y, z].
            pub fn from_slice(slice: &[$f]) -> Result<Self, crate::BufError> {
                if slice.len() < 4 {
                    return Err(crate::BufError {})
                }
                Ok(Self { w: slice[0], x: slice[1], y: slice[2], z: slice[3] })
            }

            /// Convert to a len-4 array: [w, x, y, z].
            pub fn to_arr(&self) -> [$f; 4] {
                [self.w, self.x, self.y, self.z]
            }

            /// Create the quaternion that creates the shortest (great circle) rotation from vec0
            /// to vec1.
            pub fn from_unit_vecs(v0: Vec3, v1: Vec3) -> Self {
                const ONE_MINUS_EPS: $f = 1.0 - $f::EPSILON;

                let dot = v0.dot(v1);
                if dot > ONE_MINUS_EPS {
                    return Self::new_identity();
                } else if dot < -ONE_MINUS_EPS {
                    // Rotate along any orthonormal vec to vec1 or vec2 as the axis.
                    return Self::from_axis_angle(Vec3::new(1., 0., 0.).cross(v0), TAU / 2.);
                }

                let w = 1. + dot;
                let v = v0.cross(v1);

                (Self {
                    w,
                    x: v.x,
                    y: v.y,
                    z: v.z,
                })
                .to_normalized()
            }

            /// Convert Euler angles to a quaternion.
            /// https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
            pub fn from_euler(euler: &EulerAngle) -> Self {
                let cr = (euler.roll * 0.5).cos();
                let sr = (euler.roll * 0.5).sin();
                let cp = (euler.pitch * 0.5).cos();
                let sp = (euler.pitch * 0.5).sin();
                let cy = (euler.yaw * 0.5).cos();
                let sy = (euler.yaw * 0.5).sin();

                Self {
                    w: cr * cp * cy + sr * sp * sy,
                    x: sr * cp * cy - cr * sp * sy,
                    y: cr * sp * cy + sr * cp * sy,
                    z: cr * cp * sy - sr * sp * cy,
                }
            }

            /// Convert this quaternion to Tait-Bryon (Euler) angles.
            /// We use this reference: http://marc-b-reynolds.github.io/math/2017/04/18/TaitEuler.html.
            /// Most sources online provide us with results that do not behave how we expect.
            pub fn to_euler(&self) -> EulerAngle {
                let w = self.w;
                let x = self.y;
                let y = self.x;
                let z = self.z;

                // half z-component of x' (negated)
                let xz = w * y - x * z;

                let yaw = (x * y + w * z).atan2(0.5 - (y * y + z * z));
                let pitch = -(xz / (0.25 - xz * xz).sqrt()).atan();

                const YPR_GIMBAL_LOCK: $f = 100.; // todo: Currently always uses logic A.

                let roll = if (xz.abs()) < YPR_GIMBAL_LOCK {
                    y * z + w * x.atan2(0.5 - (x * x + y * y))
                } else {
                    2.0 * x.atan2(w) + xz.signum() * yaw
                } * -1.;

                EulerAngle { pitch, roll, yaw }
            }

            pub fn inverse(self) -> Self {
                Self {
                    w: self.w,
                    x: -self.x,
                    y: -self.y,
                    z: -self.z,
                }
            }

            /// Rotate a vector using this quaternion. Note that our multiplication Q * v
            /// operation is effectively quaternion multiplication, with a quaternion
            /// created by a vec with w=0.
            /// Uses the right hand rule.
            pub fn rotate_vec(self, vec: Vec3) -> Vec3 {
                (self * vec * self.inverse()).to_vec()
            }

            /// Create a rotation quaternion from an axis and angle.
            pub fn from_axis_angle(axis: Vec3, angle: $f) -> Self {
                // Here we calculate the sin( theta / 2) once for optimization
                let c = (angle / 2.).sin();

                Self {
                    // Calcualte the w value by cos( theta / 2 )
                    w: (angle / 2.).cos(),
                    // Calculate the x, y and z of the quaternion
                    x: axis.x * c,
                    y: axis.y * c,
                    z: axis.z * c,
                }
            }

            /// Extract the axis of rotation.
            pub fn axis(&self) -> Vec3 {
                if self.w.abs() > 1. - $f::EPSILON {
                    return Vec3 {
                        x: 1.,
                        y: 0.,
                        z: 0.,
                    }; // arbitrary.
                }

                let denom = (1. - self.w.powi(2)).sqrt();

                if denom.abs() < $f::EPSILON {
                    return Vec3 {
                        x: 1.,
                        y: 0.,
                        z: 0.,
                    }; // Arbitrary normalized vector
                }

                Vec3 {
                    x: self.x / denom,
                    y: self.y / denom,
                    z: self.z / denom,
                }
            }

            /// Extract the axis of rotation.
            pub fn angle(&self) -> $f {
                // Generally, this will be due to it being slightly higher than 1,
                // but any value > 1 will return NaN from acos.
                if self.w.abs() > 1. - $f::EPSILON {
                    return 0.;
                }

                2. * self.w.acos()
            }

            /// Convert an attitude to rotations around individual axes.
            /// Assumes X is left, Z is up, and Y is forward.
            pub fn to_axes(&self) -> ($f, $f, $f) {
                let axis = self.axis();
                let angle = self.angle();

                let sign_x = -axis.x.signum();
                let sign_y = -axis.y.signum();
                let sign_z = -axis.z.signum();

                let x_component = (axis.project_to_vec(RIGHT) * angle).magnitude() * sign_x;
                let y_component = (axis.project_to_vec(FORWARD) * angle).magnitude() * sign_y;
                let z_component = (axis.project_to_vec(UP) * angle).magnitude() * sign_z;

                (x_component, y_component, z_component)
            }

            /// Convert to a 3D vector, discarding `w`.
            pub fn to_vec(self) -> Vec3 {
                Vec3 {
                    x: self.x,
                    y: self.y,
                    z: self.z,
                }
            }

            /// Returns the magnitude.
            pub fn magnitude(&self) -> $f {
                (self.w.powi(2) + self.x.powi(2) + self.y.powi(2) + self.z.powi(2)).sqrt()
            }

            /// Returns the normalised version of the quaternion
            pub fn to_normalized(self) -> Self {
                let mag_recip = 1. / self.magnitude();
                self * mag_recip
            }

            /// Used by `slerp`.
            pub fn dot(&self, rhs: Self) -> $f {
                self.w * rhs.w + self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
            }

            /// Used as part of `slerp`.
            /// https://github.com/bitshifter/glam-rs/blob/main/src/$f/scalar/quat.rs#L546
            fn lerp(self, end: Self, amount: $f) -> Self {
                let start = self;
                let dot = start.dot(end);

                let bias = if dot >= 0. { 1. } else { -1. };

                let interpolated = start.add(end.mul(bias).sub(start).mul(amount));
                interpolated.to_normalized()
            }

            /// Performs Spherical Linear Interpolation between this and another quaternion. A high
            /// `amount` will make the result more towards `end`. An `amount` of 0 will result in
            /// this quaternion.
            /// Ref: https://github.com/bitshifter/glam-rs/blob/main/src/$f/scalar/quat.rs#L567
            /// Alternative implementation to examine: https://github.com/MartinWeigel/Quaternion/blob/master/Quaternion.c
            pub fn slerp(&self, mut end: Quaternion, amount: $f) -> Quaternion {
                const DOT_THRESHOLD: $f = 0.9995;

                // Note that a rotation can be represented by two quaternions: `q` and
                // `-q`. The slerp path between `q` and `end` will be different from the
                // path between `-q` and `end`. One path will take the long way around and
                // one will take the short way. In order to correct for this, the `dot`
                // product between `self` and `end` should be positive. If the `dot`
                // product is negative, slerp between `self` and `-end`.
                let mut dot = self.dot(end);
                if dot < 0.0 {
                    end = end * -1.;
                    dot = dot * -1.;
                }

                if dot > DOT_THRESHOLD {
                    // assumes lerp returns a normalized quaternion
                    self.lerp(end, amount)
                } else {
                    let theta = dot.acos();

                    let scale1 = (theta * (1. - amount)).sin();
                    let scale2 = (theta * amount).sin();
                    let theta_sin = theta.sin();

                    self.mul(scale1).add(end.mul(scale2)).mul(1. / theta_sin)
                }
            }

            /// Converts a Quaternion to a rotation matrix
                            #[rustfmt::skip]
            pub fn to_matrix(&self) -> Mat4 {
                                // https://docs.rs/glam/latest/src/glam/$f/mat3.rs.html#159-180
                                let x2 = self.x + self.x;
                                let y2 = self.y + self.y;
                                let z2 = self.z + self.z;

                                let xx = self.x * x2;
                                let xy = self.x * y2;
                                let xz = self.x * z2;

                                let yy = self.y * y2;
                                let yz = self.y * z2;
                                let zz = self.z * z2;
                                let wx = self.w * x2;
                                let wy = self.w * y2;
                                let wz = self.w * z2;

                                Mat4 {
                                    data: [
                                        1.0 - (yy + zz), xy + wz, xz - wy, 0.,
                                        xy - wz, 1.0 - (xx + zz), yz + wx, 0.,
                                        xz + wy, yz - wx, 1.0 - (xx + yy), 0.,
                                        0., 0., 0., 1.,
                                    ]
                                }
                            }

            /// Converts a Quaternion to a rotation matrix
            #[rustfmt::skip]
            pub fn to_matrix3(&self) -> Mat3 {
                // https://docs.rs/glam/latest/src/glam/$f/mat3.rs.html#159-180
                let x2 = self.x + self.x;
                let y2 = self.y + self.y;
                let z2 = self.z + self.z;

                let xx = self.x * x2;
                let xy = self.x * y2;
                let xz = self.x * z2;

                let yy = self.y * y2;
                let yz = self.y * z2;
                let zz = self.z * z2;

                let wx = self.w * x2;
                let wy = self.w * y2;
                let wz = self.w * z2;

                Mat3{
                    data: [
                        1.0 - (yy + zz), xy + wz, xz - wy,
                        xy - wz, 1.0 - (xx + zz), yz + wx,
                        xz + wy, yz - wx, 1.0 - (xx + yy),
                    ]
                }
            }
        }

        #[cfg(not(feature = "no_std"))]
        impl fmt::Display for Quaternion {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                write!(f, "Q|{:.4}, {:.4}, {:.4}, {:.4}|", self.w, self.x, self.y, self.z)?;
                Ok(())
            }
        }

        #[derive(Clone, Debug)]
        #[cfg_attr(feature = "bincode", derive(Encode, Decode))]
        /// Represents a set of Euler angles.
        pub struct EulerAngle {
            pub roll: $f,
            pub pitch: $f,
            pub yaw: $f,
        }

        #[derive(Clone, Debug)]
        #[cfg_attr(feature = "bincode", derive(Encode, Decode))]
        /// A 3x3 matrix. Data and operations are column-major.
        pub struct Mat3 {
            pub data: [$f; 9],
        }

        // todo: temp?
        impl From<[[$f; 3]; 3]> for Mat3 {
            #[rustfmt::skip]
            fn from(m: [[$f; 3]; 3]) -> Self {
                Self {
                    data: [
                        m[0][0], m[0][1], m[0][2],
                        m[1][0], m[1][1], m[1][2],
                        m[2][0], m[2][1], m[2][2],
                    ]
                }
            }
        }

        // todo: temp?
        impl From<Mat3> for [[$f; 3]; 3] {
            #[rustfmt::skip]
            fn from(m: Mat3) -> Self {
                let d = m.data;
                [
                    [d[0], d[1], d[2]],
                    [d[3], d[4], d[5]],
                    [d[6], d[7], d[8]]
                ]
            }
        }

        impl Mat3 {
            pub fn new(data: [$f; 9]) -> Self {
                Self { data }
            }

            /// Create a matrix from column vectors
            #[rustfmt::skip]
            pub fn from_cols(x: Vec3, y: Vec3, z: Vec3) -> Self {
            Self::new([
                x.x, x.y, x.z,
                y.x, y.y, y.z,
                z.x, z.y, z.z
            ])
            }


            #[rustfmt::skip]
            /// Calculate the matrix's determinant.
            pub fn determinant(&self) -> $f {
                let d = self.data; // code shortener.

                d[0] * d[4] * d[8] +
                d[3] * d[7] * d[2] +
                d[6] * d[1] * d[5] -
                d[0] * d[7] * d[5] -
                d[3] * d[1] * d[8] -
                d[6] * d[4] * d[2]
            }

            #[rustfmt::skip]
            pub fn new_identity() -> Self {
                Self {
                    data: [
                        1., 0., 0.,
                        0., 1., 0.,
                        0., 0., 1.,
                    ]
                }
            }

            #[cfg(feature = "computer_graphics")]
            pub fn to_bytes(&self) -> [u8; 9 * 4] {
                // todo: f64 vs f32 serialization.
                let mut result = [0; 9 * 4];

                for i in 0..self.data.len() {
                    result[i * 4..i * 4 + 4].clone_from_slice(&self.data[i].to_ne_bytes());
                }

                result
            }
        }

        impl Mul<Self> for Mat3 {
            type Output = Self;

            fn mul(self, rhs: Self) -> Self::Output {
                let d = self.data; // code shortener.
                let rd = rhs.data;

                // `acr` means a(column)(row)
                let a00 = d[0] * rd[0] + d[3] * rd[1] + d[6] * rd[2];
                let a10 = d[0] * rd[3] + d[3] * rd[4] + d[6] * rd[5];
                let a20 = d[0] * rd[6] + d[3] * rd[7] + d[6] * rd[8];

                let a01 = d[1] * rd[0] + d[4] * rd[1] + d[7] * rd[2];
                let a11 = d[1] * rd[3] + d[4] * rd[4] + d[7] * rd[5];
                let a21 = d[1] * rd[6] + d[4] * rd[7] + d[7] * rd[8];

                let a02 = d[2] * rd[0] + d[5] * rd[1] + d[8] * rd[2];
                let a12 = d[2] * rd[3] + d[5] * rd[4] + d[8] * rd[5];
                let a22 = d[2] * rd[6] + d[5] * rd[7] + d[8] * rd[8];

                Self {
                    data: [a00, a01, a02, a10, a11, a12, a20, a21, a22],
                }
            }
        }

        impl Mul<Vec3> for Mat3 {
            type Output = Vec3;

            fn mul(self, rhs: Vec3) -> Self::Output {
                Vec3 {
                    x: rhs.x * self.data[0] + rhs.y * self.data[3] + rhs.z * self.data[6],
                    y: rhs.x * self.data[1] + rhs.y * self.data[4] + rhs.z * self.data[7],
                    z: rhs.x * self.data[2] + rhs.y * self.data[5] + rhs.z * self.data[8],
                }
            }
        }

        impl Mul<$f> for Mat3 {
            type Output = Self;

            fn mul(self, rhs: $f) -> Self::Output {
                let d = self.data; // code shortener.
                Self {
                    data: [
                        d[0] * rhs,
                        d[1] * rhs,
                        d[2] * rhs,
                        d[3] * rhs,
                        d[4] * rhs,
                        d[5] * rhs,
                        d[6] * rhs,
                        d[7] * rhs,
                        d[8] * rhs,
                    ],
                }
            }
        }

        impl Add<Self> for Mat3 {
            type Output = Self;

            fn add(self, rhs: Self) -> Self::Output {
                let d = self.data; // code shortener.
                let rd = rhs.data;

                Self {
                    data: [
                        d[0] + rd[0],
                        d[1] + rd[1],
                        d[2] + rd[2],
                        d[3] + rd[3],
                        d[4] + rd[4],
                        d[5] + rd[5],
                        d[6] + rd[6],
                        d[7] + rd[7],
                        d[8] + rd[8],
                    ],
                }
            }
        }

        #[derive(Clone, Debug)]
        #[cfg_attr(feature = "bincode", derive(Encode, Decode))]
        /// A 4x4 matrix. Data and operations are column-major.
        pub struct Mat4 {
            pub data: [$f; 16],
        }

        // todo temp?
        impl From<[[$f; 4]; 4]> for Mat4 {
            #[rustfmt::skip]
            fn from(m: [[$f; 4]; 4]) -> Self {
                Self {
                    data: [
                        m[0][0], m[0][1], m[0][2], m[0][3],
                        m[1][0], m[1][1], m[1][2], m[0][3],
                        m[2][0], m[2][1], m[2][2], m[0][3],
                        m[3][0], m[3][1], m[3][2], m[3][3],
                    ]
                }
            }
        }

        // todo temp?
        impl From<Mat4> for [[$f; 4]; 4] {
            #[rustfmt::skip]
            fn from(m: Mat4) -> Self {
                let d = m.data;
                [
                    [d[0], d[1], d[2], d[3]],
                    [d[4], d[5], d[6], d[7]],
                    [d[8], d[9], d[10], d[11]],
                    [d[12], d[13], d[14], d[15]],
                ]
            }
        }

        // todo: DRY. Call above instead?
        // todo temp?
        impl From<&Mat4> for [[$f; 4]; 4] {
            #[rustfmt::skip]
            fn from(m: &Mat4) -> Self {
                let d = m.data;
                [
                    [d[0], d[1], d[2], d[3]],
                    [d[4], d[5], d[6], d[7]],
                    [d[8], d[9], d[10], d[11]],
                    [d[12], d[13], d[14], d[15]],
                ]
            }
        }

        impl Mat4 {
            pub fn new(data: [$f; 16]) -> Self {
                Self { data }
            }

            /// Creates a left-hand perspective projection matrix with 0-1 depth range.
            /// Field of view is in radians. Aspect is width / height.
            /// https://docs.rs/glam/latest/src/glam/$f/sse2/mat4.rs.html#818-830
            #[cfg(feature = "computer_graphics")]
            #[rustfmt::skip]
            pub fn new_perspective_lh(fov_y: $f, aspect_ratio: $f, z_near: $f, z_far: $f) -> Self {
                let (sin_fov, cos_fov) = (0.5 * fov_y).sin_cos();
                let h = cos_fov / sin_fov;
                let w = h / aspect_ratio;
                let r = z_far / (z_far - z_near);

                Self {
                    data: [
                        w, 0., 0., 0.,
                        0., h, 0., 0.,
                        0., 0., r, 1.,
                        0., 0., -r * z_near, 0.
                    ]
                }
            }

            // "Note that we first do a translation and then a scale transformation when multiplying matrices.
            // Matrix multiplication is not commutative, which means their order is important. When
            // multiplying matrices the right-most matrix is first multiplied with the vector so you should
            // read the multiplications from right to left. It is advised to first do scaling operations,
            // then rotations and lastly translations when combining matrices otherwise they may (negatively)
            // affect each other. For example, if you would first do a translation and then scale, the translation
            // vector would also scale!"

            /// https://learnopengl.com/Getting-started/Transformations
            #[rustfmt::skip]
            #[cfg(feature = "computer_graphics")]
            pub fn new_rotation(val: Vec3) -> Self {
                let (sin_x, cos_x) = val.x.sin_cos();
                let (sin_y, cos_y) = val.y.sin_cos();
                let (sin_z, cos_z) = val.z.sin_cos();

                let rot_x = Self {
                    data: [
                        1., 0., 0., 0.,
                        0., cos_x, sin_x, 0.,
                        0., -sin_x, cos_x, 0.,
                        0., 0., 0., 1.
                    ]
                };

                let rot_y = Self {
                    data: [
                        cos_y, 0., -sin_y, 0.,
                        0., 1., 0., 0.,
                        sin_y, 0., cos_y, 0.,
                        0., 0., 0., 1.
                    ]
                };

                let rot_z = Self {
                    data: [
                        cos_z, sin_z, 0., 0.,
                        -sin_z, cos_z, 0., 0.,
                        0., 0., 1., 0.,
                        0., 0., 0., 1.
                    ]
                };

                // todo: What order to apply these three ?
                // todo: TO avoid gimbal lock, consider rotating aroudn an arbitrary unit axis immediately.

                rot_x * rot_y * rot_z
            }

            #[cfg(feature = "computer_graphics")]
            #[rustfmt::skip]
            pub fn new_scaler(scale: $f) -> Self {
                Self {
                    data: [
                        scale, 0., 0., 0.,
                        0., scale, 0., 0.,
                        0., 0., scale, 0.,
                        0., 0., 0., 1.,
                    ]
                }
            }


            #[cfg(feature = "computer_graphics")]
            #[rustfmt::skip]
            pub fn new_scaler_partial(scale: Vec3) -> Self {
                Self {
                    data: [
                        scale.x, 0., 0., 0.,
                        0., scale.y, 0., 0.,
                        0., 0., scale.z, 0.,
                        0., 0., 0., 1.,
                    ]
                }
            }

            #[cfg(feature = "computer_graphics")]
            #[rustfmt::skip]
            /// Create a translation matrix. Note that the matrix is 4x4, but it takes len-3 vectors -
            /// this is so we can compose it with other 4x4 matrices.
            pub fn new_translation(val: Vec3) -> Self {
                Self {
                    data: [
                        1., 0., 0., 0.,
                        0., 1., 0., 0.,
                        0., 0., 1., 0.,
                        val.x, val.y, val.z, 1.
                    ]
                }
            }

            #[rustfmt::skip]
            pub fn new_identity() -> Self {
                Self {
                    data: [
                        1., 0., 0., 0.,
                        0., 1., 0., 0.,
                        0., 0., 1., 0.,
                        0., 0., 0., 1.,
                    ]
                }
            }

            #[rustfmt::skip]
            /// Calculate the matrix's determinant.
            pub fn determinant(&self) -> $f {
                let d = self.data; // code shortener.

                d[0] * d[5] * d[10] * d[15] +
                d[4] * d[9] * d[14] * d[3] +
                d[8] * d[13] * d[2] * d[7] +
                d[12] * d[1] * d[6] * d[11] -
                d[0] * d[13] * d[10] * d[7] -
                d[4] * d[1] * d[14] * d[11] -
                d[8] * d[5] * d[2] * d[15] -
                d[12] * d[9] * d[6] * d[3]
            }

            /// Transpose the matrix
            #[rustfmt::skip]
            pub fn transpose(&self) -> Self {
                let d = self.data; // code shortener.
                Self {
                    data: [
                        d[0], d[4], d[8], d[12],
                        d[1], d[5], d[9], d[13],
                        d[2], d[6], d[10], d[14],
                        d[3], d[7], d[11], d[15]
                    ]
                }
            }

            /// Returns cols: x, y, z, w
            pub fn to_cols(&self) -> (Vec4, Vec4, Vec4, Vec4) {
                let d = self.data; // code shortener.
                (
                    Vec4::new(d[0], d[1], d[2], d[3]),
                    Vec4::new(d[4], d[5], d[6], d[7]),
                    Vec4::new(d[8], d[9], d[10], d[11]),
                    Vec4::new(d[12], d[13], d[14], d[15]),
                )
            }

            /// See cgmath's impl.
            #[rustfmt::skip]
            pub fn inverse(&self) -> Option<Self> {
                let det = self.determinant();
                if det == 0. {
                    None
                } else {
                    let inv_det = 1. / det;
                    let t = self.transpose();
                    let (t_x, t_y, t_z, t_w) = t.to_cols();

                    // todo!!
                    let cf = |i, j| {
                    let mat = match i {
                    0 => {
                    Mat3::from_cols(t_y.truncate_n(j), t_z.truncate_n(j), t_w.truncate_n(j))
                }
                1 => {
                    Mat3::from_cols(t_x.truncate_n(j), t_z.truncate_n(j), t_w.truncate_n(j))
                }
                2 => {
                    Mat3::from_cols(t_x.truncate_n(j), t_y.truncate_n(j), t_w.truncate_n(j))
                }
                3 => {
                    Mat3::from_cols(t_x.truncate_n(j), t_y.truncate_n(j), t_z.truncate_n(j))
                }
                _ => panic!("out of range"),
                };
                let sign = if (i + j) & 1 == 1 {
                -1.
                } else {
                1.
                };
                mat.determinant() * sign * inv_det
                };

                Some(Mat4::new([
                    cf(0, 0), cf(0, 1), cf(0, 2), cf(0, 3),
                    cf(1, 0), cf(1, 1), cf(1, 2), cf(1, 3),
                    cf(2, 0), cf(2, 1), cf(2, 2), cf(2, 3),
                    cf(3, 0), cf(3, 1), cf(3, 2), cf(3, 3),
                    ]))
                }
            }

            #[cfg(feature = "computer_graphics")]
            pub fn to_bytes(&self) -> [u8; 16 * 4] {
                // todo: f64 vs f32 serialization.
                let mut result = [0; 16 * 4];

                for i in 0..self.data.len() {
                    result[i * 4..i * 4 + 4].clone_from_slice(&self.data[i].to_ne_bytes());
                }

                result
            }
        }

        impl Mul<Self> for Mat4 {
            type Output = Self;

            fn mul(self, rhs: Self) -> Self::Output {
                let d = self.data; // code shortener
                let rd = rhs.data;

                // acr means a(column)(row)
                let a00 = d[0] * rd[0] + d[4] * rd[1] + d[8] * rd[2] + d[12] * rd[3];
                let a10 = d[0] * rd[4] + d[4] * rd[5] + d[8] * rd[6] + d[12] * rd[7];
                let a20 = d[0] * rd[8] + d[4] * rd[9] + d[8] * rd[10] + d[12] * rd[11];
                let a30 = d[0] * rd[12] + d[4] * rd[13] + d[8] * rd[14] + d[12] * rd[15];

                let a01 = d[1] * rd[0] + d[5] * rd[1] + d[9] * rd[2] + d[13] * rd[3];
                let a11 = d[1] * rd[4] + d[5] * rd[5] + d[9] * rd[6] + d[13] * rd[7];
                let a21 = d[1] * rd[8] + d[5] * rd[9] + d[9] * rd[10] + d[13] * rd[11];
                let a31 = d[1] * rd[12] + d[5] * rd[13] + d[9] * rd[14] + d[13] * rd[15];

                let a02 = d[2] * rd[0] + d[6] * rd[1] + d[10] * rd[2] + d[14] * rd[3];
                let a12 = d[2] * rd[4] + d[6] * rd[5] + d[10] * rd[6] + d[14] * rd[7];
                let a22 = d[2] * rd[8] + d[6] * rd[9] + d[10] * rd[10] + d[14] * rd[11];
                let a32 = d[2] * rd[12] + d[6] * rd[13] + d[10] * rd[14] + d[14] * rd[15];

                let a03 = d[3] * rd[0] + d[7] * rd[1] + d[11] * rd[2] + d[15] * rd[3];
                let a13 = d[3] * rd[4] + d[7] * rd[5] + d[11] * rd[6] + d[15] * rd[7];
                let a23 = d[3] * rd[8] + d[7] * rd[9] + d[11] * rd[10] + d[15] * rd[11];
                let a33 = d[3] * rd[12] + d[7] * rd[13] + d[11] * rd[14] + d[15] * rd[15];

                Self {
                    data: [
                        a00, a01, a02, a03, a10, a11, a12, a13, a20, a21, a22, a23, a30, a31, a32,
                        a33,
                    ],
                }
            }
        }

        impl Mul<Vec4> for Mat4 {
            type Output = Vec4;

            fn mul(self, rhs: Vec4) -> Self::Output {
                Vec4 {
                    x: rhs.x * self.data[0]
                        + rhs.y * self.data[4]
                        + rhs.z * self.data[8]
                        + self.data[12] * rhs.w,
                    y: rhs.x * self.data[1]
                        + rhs.y * self.data[5]
                        + rhs.z * self.data[9]
                        + self.data[13] * rhs.w,
                    z: rhs.x * self.data[2]
                        + rhs.y * self.data[6]
                        + rhs.z * self.data[10]
                        + self.data[14] * rhs.w,
                    w: rhs.x * self.data[3]
                        + rhs.y * self.data[7]
                        + rhs.z * self.data[11]
                        + self.data[15] * rhs.w,
                }
            }
        }

        impl Mul<$f> for Mat4 {
            type Output = Self;

            fn mul(self, rhs: $f) -> Self::Output {
                Self {
                    data: [
                        self.data[0] * rhs,
                        self.data[1] * rhs,
                        self.data[2] * rhs,
                        self.data[3] * rhs,
                        self.data[4] * rhs,
                        self.data[5] * rhs,
                        self.data[6] * rhs,
                        self.data[7] * rhs,
                        self.data[8] * rhs,
                        self.data[9] * rhs,
                        self.data[10] * rhs,
                        self.data[11] * rhs,
                        self.data[12] * rhs,
                        self.data[13] * rhs,
                        self.data[14] * rhs,
                        self.data[15] * rhs,
                    ],
                }
            }
        }

        #[cfg(not(feature = "no_std"))]
        impl fmt::Display for Mat4 {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                let d = self.data;
                writeln!(f, "\n|{:.2} {:.2} {:.2} {:.2}|", d[0], d[4], d[8], d[12])?;
                writeln!(f, "|{:.2} {:.2} {:.2} {:.2}|", d[1], d[5], d[9], d[13])?;
                writeln!(f, "|{:.2} {:.2} {:.2} {:.2}|", d[2], d[6], d[10], d[14])?;
                writeln!(f, "|{:.2} {:.2} {:.2} {:.2}|", d[3], d[7], d[11], d[15])?;

                Ok(())
            }
        }

        /// Calculate the determinate of a matrix defined by its columns.
        /// We use this for determining the full 0 - tau angle between bonds.
        #[rustfmt::skip]
        pub fn det_from_cols(c0: Vec3, c1: Vec3, c2: Vec3) -> $f {
            c0.x * c1.y * c2.z +
    c1.x * c2.y * c0.z +
    c2.x * c0.y * c1.z -
    c0.x * c2.y * c1.z -
    c1.x * c0.y * c2.z -
    c2.x * c1.y * c0.z
}
    };
}

pub mod f32 {
    create!(f32);

    use super::f64;

    impl From<f64::Vec2> for Vec2 {
        fn from(other: f64::Vec2) -> Self {
            Self{ x: other.x as f32, y: other.y as f32 }
        }
    }

    impl From<f64::Vec3> for Vec3 {
        fn from(other: f64::Vec3) -> Self {
            Self { x: other.x as f32, y: other.y as f32, z: other.z as f32 }
        }
    }

    impl From<f64::Vec4> for Vec4 {
        fn from(other: f64::Vec4) -> Self {
            Self{ w: other.w as f32, x: other.x as f32, y: other.y as f32, z: other.z as f32 }
        }
    }

    impl From<f64::Quaternion> for Quaternion {
        fn from(other: f64::Quaternion) -> Self {
            Self{ w: other.w as f32, x: other.x as f32, y: other.y as f32, z: other.z as f32 }
        }
    }

    // todo: Matrix type conversions as well.
}

pub mod f64 {
    create!(f64);

    use super::f32;

    impl From<f32::Vec2> for Vec2 {
        fn from(other: f32::Vec2) -> Self {
            Self{ x: other.x as f64, y: other.y as f64 }
        }
    }

    impl From<f32::Vec3> for Vec3 {
        fn from(other: f32::Vec3) -> Self {
            Self{ x: other.x as f64, y: other.y as f64, z: other.z as f64 }
        }
    }

    impl From<f32::Vec4> for Vec4 {
        fn from(other: f32::Vec4) -> Self {
            Self{ w: other.w as f64, x: other.x as f64, y: other.y as f64, z: other.z as f64 }
        }
    }

    impl From<f32::Quaternion> for Quaternion {
        fn from(other: f32::Quaternion) -> Self {
            Self{ w: other.w as f64, x: other.x as f64, y: other.y as f64, z: other.z as f64 }
        }
    }

    // todo: Matrix type conversions as well.
}
