//! Vector, matrix, and quatenrion operations on f32
//!
//! Module for matrices, vectors, and quaternions, as used in 3d graphics, geometry,
//! and aircraft attitude systems. Similar to the
//! `cgmath` and `glam` crates, but with a more transparent API.

use core::{
    f32::consts::TAU,
    ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub},
};

#[cfg(not(feature = "no_std"))]
use std::fmt;

#[cfg(feature = "no_std")]
use num_traits::float::Float;

const EPS: f32 = 0.0000001;

#[derive(Clone, Copy, Default, Debug, PartialEq)]
/// A len-3 column vector
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl From<[f32; 3]> for Vec3 {
    fn from(v: [f32; 3]) -> Self {
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

impl AddAssign<Self> for Vec3 {
    fn add_assign(&mut self, rhs: Self) {
        self.x = self.x + rhs.x;
        self.y = self.y + rhs.y;
        self.z = self.z + rhs.z;
    }
}

impl Mul<f32> for Vec3 {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output {
        Self {
            x: self.x * rhs,
            y: self.y * rhs,
            z: self.z * rhs,
        }
    }
}

impl Div<f32> for Vec3 {
    type Output = Self;

    fn div(self, rhs: f32) -> Self::Output {
        Self {
            x: self.x / rhs,
            y: self.y / rhs,
            z: self.z / rhs,
        }
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

impl MulAssign<f32> for Vec3 {
    fn mul_assign(&mut self, rhs: f32) {
        self.x *= rhs;
        self.y *= rhs;
        self.z *= rhs;
    }
}

impl Vec3 {
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    pub fn new_zero() -> Self {
        Self {
            x: 0.,
            y: 0.,
            z: 0.,
        }
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
    pub fn magnitude_squared(self) -> f32 {
        (self.hadamard_product(self)).sum()
    }

    pub fn magnitude(&self) -> f32 {
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

    pub fn dot(&self, rhs: Self) -> f32 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }

    /// Returns a sum of all elements
    pub fn sum(&self) -> f32 {
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

#[derive(Clone, Copy, Debug)]
/// A len-4 column vector
pub struct Vec4 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl Vec4 {
    pub fn new(x: f32, y: f32, z: f32, u: f32) -> Self {
        Self { x, y, z, w: u }
    }

    pub fn normalize(&mut self) {
        let len = (self.x.powi(2) + self.y.powi(2) + self.z.powi(2) + self.w.powi(2)).sqrt();

        self.x /= len;
        self.y /= len;
        self.z /= len;
        self.w /= len;
    }

    /// Remove the 3rd element. Used in our inverse calulations.
    pub fn truncate_n(&self, n: usize) -> Vec3 {
        match n {
            0 => Vec3::new(self.y, self.z, self.w),
            1 => Vec3::new(self.x, self.z, self.w),
            2 => Vec3::new(self.x, self.y, self.w),
            3 => Vec3::new(self.x, self.y, self.z),
            _ => panic!("{:?} is out of range", n),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Quaternion {
    pub w: f32,
    pub x: f32,
    pub y: f32,
    pub z: f32,
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

impl Mul<f32> for Quaternion {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output {
        Self {
            w: self.w * rhs,
            x: self.x * rhs,
            y: self.y * rhs,
            z: self.z * rhs,
        }
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

    pub fn new(w: f32, x: f32, y: f32, z: f32) -> Self {
        Self { w, x, y, z }
    }

    /// Create the quaternion that creates the shortest (great circle) rotation from vec0
    /// to vec1.
    pub fn from_unit_vecs(v0: Vec3, v1: Vec3) -> Self {
        const ONE_MINUS_EPS: f32 = 1.0 - 2.0 * EPS;

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

    /// Convert this quaternion to Euler angles.
    /// https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
    pub fn to_euler(&self) -> EulerAngle {
        // roll (z-axis rotation)
        let sinr_cosp = 2. * (self.w * self.x + self.y * self.z);
        let cosr_cosp = 1. - 2. * (self.x * self.x + self.y * self.y);

        let roll = sinr_cosp.atan2(cosr_cosp);

        // // pitch (x-axis rotation)
        // let sinp = 2. * (self.w * self.y - self.z * self.x);
        // let pitch = if sinp.abs() >= 1. {
        //     (TAU / 4.).copysign(sinp) // use 90 degrees if out of range
        // } else {
        //     sinp.asin()
        // };

        // todo: Above, or below? Gimbal lock?
        let c = 2. * (self.w * self.y - self.z * self.x);
        let sinp = (1. + c).sqrt();
        let cosp = (1. - c).sqrt();
        let pitch = 2. * sinp.atan2(cosp) - TAU / 4.;

        // yaw (y-axis rotation)
        let siny_cosp = 2. * (self.w * self.z + self.x * self.y);
        let cosy_cosp = 1. - 2. * (self.y * self.y + self.z * self.z);
        let yaw = siny_cosp.atan2(cosy_cosp);

        EulerAngle { roll, pitch, yaw }
    }

    // /// Converts a Quaternion to ZYX Euler angles, in radians.
    // /// todo: THis is from AHR fusion. Which do you want?
    // pub fn to_euler2(self) -> (f32, f32, f32) {
    //     let half_minus_qy_squared = 0.5 - self.y * self.y; // calculate common terms to avoid repeated operations
    //
    //     (
    //         (self.w * self.x + self.y * self.z)
    //             .atan2(half_minus_qy_squared - self.x * self.x),
    //         fusion_asin(2.0 * (self.w * self.y - self.z * self.x)),
    //         (self.w * self.z + self.x * self.y).atan2(half_minus_qy_squared - self.z * self.z),
    //     )
    // }

    // /// Creates an orientation that point towards a vector, with a given up direction defined.
    // pub fn from_vec_direction(dir: Vec3, up: Vec3) -> Self {
    //     let forward_vector = dir;
    //
    //     let forward = Vec3::new(0., 0., 1.);
    //
    //     let dot = forward.dot(forward_vector);
    //
    //     if (dot - (-1.0)).abs() < 0.000001 {
    //         // return Self: { x:  Quaternion(Vector3.up.x, Vector3.up.y, Vector3.up.z, 3.1415926535897932f);
    //         Self::new_identity(); // todo! adapt the above.
    //     }
    //     if (dot - (1.0)).abs() < 0.000001 {
    //         return Self::new_identity();
    //     }
    //
    //     let rot_angle = dot.acos();
    //     let rot_axis = forward.cross(forward_vector).to_normalized();
    //
    //     Self::from_axis_angle(rot_axis, rot_angle)
    // }

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
    pub fn rotate_vec(self, vec: Vec3) -> Vec3 {
        (self * vec * self.inverse()).to_vec()
    }

    /// Create a rotation quaternion from an axis and angle.
    pub fn from_axis_angle(axis: Vec3, angle: f32) -> Self {
        // Here we calculate the sin( theta / 2) once for optimization
        let factor = (angle / 2.).sin();

        Self {
            // Calcualte the w value by cos( theta / 2 )
            w: (angle / 2.).cos(),
            // Calculate the x, y and z of the quaternion
            x: axis.x * factor,
            y: axis.y * factor,
            z: axis.z * factor,
        }
    }

    /// Extract the axis of rotation.
    pub fn axis(&self) -> Vec3 {
        if self.w.abs() > 1. - EPS {
            return Vec3 {
                x: 1.,
                y: 0.,
                z: 0.,
            }; // arbitrary.
        }

        let denom = (1. - self.w.powi(2)).sqrt();

        // if denom.abs() < EPS {
        //     return Vec3 { x: 1., y: 0., z: 0. } // arbitrary.
        // }

        Vec3 {
            x: self.x / denom,
            y: self.y / denom,
            z: self.z / denom,
        }
    }

    /// Extract the axis of rotation.
    pub fn angle(&self) -> f32 {
        // Generally, this will be due to it being slightly higher than 1,
        // but any value > 1 will return NaN from acos.
        if self.w.abs() > 1. - EPS {
            return 0.;
        }

        2. * self.w.acos()
    }

    /// Convert to a 3D vector, discarding `w`.
    pub fn to_vec(self) -> Vec3 {
        Vec3 {
            x: self.x,
            y: self.y,
            z: self.z,
        }
    }

    /// Returns the vector magnitude.
    pub fn magnitude(&self) -> f32 {
        (self.w.powi(2) + self.x.powi(2) + self.y.powi(2) + self.z.powi(2)).sqrt()
    }

    /// Returns the normalised version of the vector
    pub fn to_normalized(self) -> Self {
        let mag_recip = 1. / self.magnitude();
        self * mag_recip
    }

    /// Converts a Quaternion to a rotation matrix
    #[rustfmt::skip]
    pub fn to_matrix(&self) -> Mat4 {
        // https://docs.rs/glam/latest/src/glam/f32/mat3.rs.html#159-180
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
        // https://docs.rs/glam/latest/src/glam/f32/mat3.rs.html#159-180
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

#[derive(Clone, Debug)]
/// Euler angles.
pub struct EulerAngle {
    pub roll: f32,
    pub pitch: f32,
    pub yaw: f32,
}

#[derive(Clone, Debug)]
/// A 3x3 matrix. Data and operations are column-major.
pub struct Mat3 {
    pub data: [f32; 9],
}

// todo: temp?
impl From<[[f32; 3]; 3]> for Mat3 {
    #[rustfmt::skip]
    fn from(m: [[f32; 3]; 3]) -> Self {
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
impl From<Mat3> for [[f32; 3]; 3] {
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
    pub fn new(data: [f32; 9]) -> Self {
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
    pub fn determinant(&self) -> f32 {
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

#[derive(Clone, Debug)]
/// A 4x4 matrix. Data and operations are column-major.
pub struct Mat4 {
    pub data: [f32; 16],
}

// todo temp?
impl From<[[f32; 4]; 4]> for Mat4 {
    #[rustfmt::skip]
    fn from(m: [[f32; 4]; 4]) -> Self {
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
impl From<Mat4> for [[f32; 4]; 4] {
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
impl From<&Mat4> for [[f32; 4]; 4] {
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
    pub fn new(data: [f32; 16]) -> Self {
        Self { data }
    }

    #[cfg(feature = "computer_graphics")]
    /// Creates a left-hand perspective projection matrix with 0-1 depth range.
    /// Field of view is in radians. Aspect is width / height.
    /// https://docs.rs/glam/latest/src/glam/f32/sse2/mat4.rs.html#818-830
    #[rustfmt::skip]
    pub fn new_perspective_lh(fov_y: f32, aspect_ratio: f32, z_near: f32, z_far: f32) -> Self {
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

    #[cfg(feature = "computer_graphics")]
    /// https://learnopengl.com/Getting-started/Transformations
    #[rustfmt::skip]
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
    pub fn new_scaler(scale: f32) -> Self {
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
    pub fn determinant(&self) -> f32 {
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
                a00, a01, a02, a03, a10, a11, a12, a13, a20, a21, a22, a23, a30, a31, a32, a33,
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

impl Mul<f32> for Mat4 {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output {
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
        write!(f, "\n|{:.2} {:.2} {:.2} {:.2}|\n", d[0], d[4], d[8], d[12])?;
        write!(f, "|{:.2} {:.2} {:.2} {:.2}|\n", d[1], d[5], d[9], d[13])?;
        write!(f, "|{:.2} {:.2} {:.2} {:.2}|\n", d[2], d[6], d[10], d[14])?;
        write!(f, "|{:.2} {:.2} {:.2} {:.2}|\n", d[3], d[7], d[11], d[15])?;

        Ok(())
    }
}

/// Calculate the determinate of a matrix defined by its columns.
/// We use this for determining the full 0 - tau angle between bonds.
#[rustfmt::skip]
pub fn det_from_cols(c0: Vec3, c1: Vec3, c2: Vec3) -> f32 {
    c0.x * c1.y * c2.z +
        c1.x * c2.y * c0.z +
        c2.x * c0.y * c1.z -
        c0.x * c2.y * c1.z -
        c1.x * c0.y * c2.z -
        c2.x * c1.y * c0.z
}
