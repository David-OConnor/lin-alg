#![macro_use]

//! Handles Quaternion operations.

// Agnostic to SIMD and non-simd
// `$f` here could be a primitive like `f32`, or a SIMD primitive like `f32x8`.
macro_rules! create_quaternion_shared {
    ($f:ident, $vec3_ty:ident, $quat_ty:ident) => {
        impl $quat_ty {
            /// Returns the magnitude.
            pub fn magnitude(&self) -> $f {
                (self.w.powi(2) + self.x.powi(2) + self.y.powi(2) + self.z.powi(2)).sqrt()
            }

            /// Normalize, modifying in place.
            pub fn normalize(&mut self) {
                let mag = self.magnitude();

                self.x /= mag;
                self.y /= mag;
                self.z /= mag;
            }

            /// Returns the normalized version of the vector.
            pub fn to_normalized(self) -> Self {
                self / self.magnitude()
            }

            /// Used by `slerp`.
            pub fn dot(&self, rhs: Self) -> $f {
                self.w * rhs.w + self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
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
            /// Uses the X_VEC hand rule.
            pub fn rotate_vec(self, vec: $vec3_ty) -> $vec3_ty {
                (self * vec * self.inverse()).to_vec()
            }
        }

        impl Default for $quat_ty {
            fn default() -> Self {
                Self::new_identity()
            }
        }

        impl Add for $quat_ty {
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

        impl Sub for $quat_ty {
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

        impl Mul for $quat_ty {
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

        impl Mul<$vec3_ty> for $quat_ty {
            type Output = Self;

            /// Returns the multiplication of a Quaternion with a vector.  This is a
            /// normal Quaternion multiplication where the vector is treated as a
            /// Quaternion with a W element value of zero. The Quaternion is post-
            /// multiplied by the vector.
            fn mul(self, rhs: $vec3_ty) -> Self::Output {
                Self {
                    w: -self.x * rhs.x - self.y * rhs.y - self.z * rhs.z,
                    x: self.w * rhs.x + self.y * rhs.z - self.z * rhs.y,
                    y: self.w * rhs.y - self.x * rhs.z + self.z * rhs.x,
                    z: self.w * rhs.z + self.x * rhs.y - self.y * rhs.x,
                }
            }
        }

        impl Div for $quat_ty {
            type Output = Self;

            fn div(self, rhs: Self) -> Self::Output {
                self * rhs.inverse()
            }
        }

        impl Mul<$f> for $quat_ty {
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

        // todo: Mul and div assign are not compiling for simd?
        // impl MulAssign<$quat_ty> for $quat_ty {
        //     fn mul_assign(&mut self, rhs: $quat_ty) {
        //         self = self * rhs;
        //     }
        // }
        //
        // impl DivAssign<$quat_ty> for $quat_ty {
        //     fn div_assign(&mut self, rhs: $quat_ty) {
        //         self = self  rhs;
        //     }
        // }

        impl MulAssign<$f> for $quat_ty {
            fn mul_assign(&mut self, rhs: $f) {
                self.w = self.w * rhs;
                self.x = self.x * rhs;
                self.y = self.y * rhs;
                self.z = self.z * rhs;
            }
        }

        impl Div<$f> for $quat_ty {
            type Output = Self;

            fn div(self, rhs: $f) -> Self::Output {
                Self {
                    w: self.w / rhs,
                    x: self.x / rhs,
                    y: self.y / rhs,
                    z: self.z / rhs,
                }
            }
        }

        impl DivAssign<$f> for $quat_ty {
            fn div_assign(&mut self, rhs: $f) {
                self.w = self.w / rhs;
                self.x = self.x / rhs;
                self.y = self.y / rhs;
                self.z = self.z / rhs;
            }
        }

        // todo: Consider this, and add A/R
        // impl Div<$vec3_ty> for $quat_ty {
        //     type Output = Self;
        //
        //     /// Returns the division of a Quaternion with a vector. This is a
        //     /// normal Quaternion division where the vector is treated a
        //     /// Quaternion with a W element value of zero. The Quaternion is post-
        //     /// divided by the vector.
        //     fn div(self, rhs: $vec3_ty) -> Self::Output {
        //         let as_vec = Self {
        //             w: 0.0,
        //             x: rhs.x,
        //             y: rhs.y,
        //             z: rhs.z,
        //         };
        //
        //         self * as_vec.inverse()
        //     }
        // }
    };
}

macro_rules! create_quaternion {
    ($f:ident) => {
        /// A quaternion using Hamilton (not JPL) transformation conventions. The most common operations
        /// usedful for representing orientations and rotations are defined, including for operations
        /// with `Vec3`.
        #[derive(Clone, Copy, Debug, PartialEq)]
        #[cfg_attr(feature = "encode", derive(Encode, Decode))]
        pub struct Quaternion {
            pub w: $f,
            pub x: $f,
            pub y: $f,
            pub z: $f,
        }

        impl Quaternion {
            pub const fn new_identity() -> Self {
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
                    return Err(crate::BufError {});
                }
                Ok(Self {
                    w: slice[0],
                    x: slice[1],
                    y: slice[2],
                    z: slice[3],
                })
            }

            /// Convert to a len-4 array: [w, x, y, z].
            pub fn to_arr(&self) -> [$f; 4] {
                [self.w, self.x, self.y, self.z]
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
            /// Assumes X is left, Z is Z_VEC, and Y is Y_VEC.
            pub fn to_axes(&self) -> ($f, $f, $f) {
                let axis = self.axis();
                let angle = self.angle();

                let sign_x = -axis.x.signum();
                let sign_y = -axis.y.signum();
                let sign_z = -axis.z.signum();

                let x_component = (axis.project_to_vec(X_VEC) * angle).magnitude() * sign_x;
                let y_component = (axis.project_to_vec(Y_VEC) * angle).magnitude() * sign_y;
                let z_component = (axis.project_to_vec(Z_VEC) * angle).magnitude() * sign_z;

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
                    end *= -1.;
                    dot *= -1.;
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
                        1.0 - (yy + zz),
                        xy + wz,
                        xz - wy,
                        0.,
                        xy - wz,
                        1.0 - (xx + zz),
                        yz + wx,
                        0.,
                        xz + wy,
                        yz - wx,
                        1.0 - (xx + yy),
                        0.,
                        0.,
                        0.,
                        0.,
                        1.,
                    ],
                }
            }

            /// Converts a Quaternion to a rotation matrix
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

                Mat3 {
                    data: [
                        1.0 - (yy + zz),
                        xy + wz,
                        xz - wy,
                        xy - wz,
                        1.0 - (xx + zz),
                        yz + wx,
                        xz + wy,
                        yz - wx,
                        1.0 - (xx + yy),
                    ],
                }
            }
        }

        #[cfg(feature = "std")]
        impl fmt::Display for Quaternion {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                write!(
                    f,
                    "Q|{:.4}, {:.4}, {:.4}, {:.4}|",
                    self.w, self.x, self.y, self.z
                )?;
                Ok(())
            }
        }

        #[cfg(feature = "cuda")]
        /// Convert a collection of `Quaternion`s into Cuda arrays of their components.
        pub fn quaternions_to_dev(stream: &Arc<CudaStream>, data: &[Quaternion]) -> CudaSlice<$f> {
            let mut result = Vec::new();
            // todo: Ref etcs A/R; you are making a double copy here.
            for v in data {
                result.push(v.w as $f);
                result.push(v.x as $f);
                result.push(v.y as $f);
                result.push(v.z as $f);
            }
            stream.memcpy_stod(&result).unwrap()
        }

        #[cfg(feature = "cuda")]
        pub fn quaternions_from_dev(
            stream: &Arc<CudaStream>,
            data_dev: &CudaSlice<$f>,
        ) -> Vec<Quaternion> {
            let data_host = stream.memcpy_dtov(data_dev).unwrap();

            data_host
                .chunks_exact(4)
                .map(|chunk| Quaternion::new(chunk[0], chunk[1], chunk[2], chunk[3]))
                .collect()
        }

        #[cfg(feature = "cuda")]
        unsafe impl cudarc::driver::DeviceRepr for Quaternion {}
    };
}
