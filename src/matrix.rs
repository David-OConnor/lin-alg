//! Haandles Matrix operations

#![macro_use]
macro_rules! create_matrix {
    ($f:ident) => {
        #[derive(Clone, Debug)]
        #[cfg_attr(feature = "encode", derive(Encode, Decode))]
        /// A 3x3 matrix. Data and operations are column-major.
        pub struct Mat3 {
            pub data: [$f; 9],
        }

        // todo: temp?
        impl From<[[$f; 3]; 3]> for Mat3 {
            fn from(m: [[$f; 3]; 3]) -> Self {
                Self {
                    data: [
                        m[0][0], m[0][1], m[0][2], m[1][0], m[1][1], m[1][2], m[2][0], m[2][1],
                        m[2][2],
                    ],
                }
            }
        }

        // todo: temp?
        impl From<Mat3> for [[$f; 3]; 3] {
            fn from(m: Mat3) -> Self {
                let d = m.data;
                [[d[0], d[1], d[2]], [d[3], d[4], d[5]], [d[6], d[7], d[8]]]
            }
        }

        impl Mat3 {
            pub fn new(data: [$f; 9]) -> Self {
                Self { data }
            }

            /// Create a matrix from column vectors
            pub fn from_cols(x: Vec3, y: Vec3, z: Vec3) -> Self {
                Self::new([x.x, x.y, x.z, y.x, y.y, y.z, z.x, z.y, z.z])
            }

            /// Calculate the matrix's determinant.
            pub fn determinant(&self) -> $f {
                let d = self.data; // code shortener.

                d[0] * d[4] * d[8] + d[3] * d[7] * d[2] + d[6] * d[1] * d[5]
                    - d[0] * d[7] * d[5]
                    - d[3] * d[1] * d[8]
                    - d[6] * d[4] * d[2]
            }

            pub fn new_identity() -> Self {
                Self {
                    data: [1., 0., 0., 0., 1., 0., 0., 0., 1.],
                }
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
        #[cfg_attr(feature = "encode", derive(Encode, Decode))]
        /// A 4x4 matrix. Data and operations are column-major.
        pub struct Mat4 {
            pub data: [$f; 16],
        }

        impl From<[[$f; 4]; 4]> for Mat4 {
            fn from(m: [[$f; 4]; 4]) -> Self {
                Self {
                    data: [
                        m[0][0], m[0][1], m[0][2], m[0][3], m[1][0], m[1][1], m[1][2], m[0][3],
                        m[2][0], m[2][1], m[2][2], m[0][3], m[3][0], m[3][1], m[3][2], m[3][3],
                    ],
                }
            }
        }

        impl From<Mat4> for [[$f; 4]; 4] {
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
        impl From<&Mat4> for [[$f; 4]; 4] {
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
            pub fn new_perspective_lh(fov_y: $f, aspect_ratio: $f, z_near: $f, z_far: $f) -> Self {
                let (sin_fov, cos_fov) = (0.5 * fov_y).sin_cos();
                let h = cos_fov / sin_fov;
                let w = h / aspect_ratio;
                let r = z_far / (z_far - z_near);

                Self {
                    data: [
                        w,
                        0.,
                        0.,
                        0.,
                        0.,
                        h,
                        0.,
                        0.,
                        0.,
                        0.,
                        r,
                        1.,
                        0.,
                        0.,
                        -r * z_near,
                        0.,
                    ],
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
            #[cfg(feature = "computer_graphics")]
            pub fn new_rotation(val: Vec3) -> Self {
                let (sin_x, cos_x) = val.x.sin_cos();
                let (sin_y, cos_y) = val.y.sin_cos();
                let (sin_z, cos_z) = val.z.sin_cos();

                let rot_x = Self {
                    data: [
                        1., 0., 0., 0., 0., cos_x, sin_x, 0., 0., -sin_x, cos_x, 0., 0., 0., 0., 1.,
                    ],
                };

                let rot_y = Self {
                    data: [
                        cos_y, 0., -sin_y, 0., 0., 1., 0., 0., sin_y, 0., cos_y, 0., 0., 0., 0., 1.,
                    ],
                };

                let rot_z = Self {
                    data: [
                        cos_z, sin_z, 0., 0., -sin_z, cos_z, 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.,
                    ],
                };

                // todo: What order to apply these three ?
                // todo: TO avoid gimbal lock, consider rotating aroudn an arbitrary unit axis immediately.

                rot_x * rot_y * rot_z
            }

            #[cfg(feature = "computer_graphics")]
            pub fn new_scaler(scale: $f) -> Self {
                Self {
                    data: [
                        scale, 0., 0., 0., 0., scale, 0., 0., 0., 0., scale, 0., 0., 0., 0., 1.,
                    ],
                }
            }

            #[cfg(feature = "computer_graphics")]
            pub fn new_scaler_partial(scale: Vec3) -> Self {
                Self {
                    data: [
                        scale.x, 0., 0., 0., 0., scale.y, 0., 0., 0., 0., scale.z, 0., 0., 0., 0.,
                        1.,
                    ],
                }
            }

            #[cfg(feature = "computer_graphics")]
            /// Create a translation matrix. Note that the matrix is 4x4, but it takes len-3 vectors -
            /// this is so we can compose it with other 4x4 matrices.
            pub fn new_translation(val: Vec3) -> Self {
                Self {
                    data: [
                        1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., val.x, val.y, val.z, 1.,
                    ],
                }
            }

            pub fn new_identity() -> Self {
                Self {
                    data: [
                        1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.,
                    ],
                }
            }

            /// Calculate the matrix's determinant.
            pub fn determinant(&self) -> $f {
                let d = self.data; // code shortener.

                d[0] * d[5] * d[10] * d[15]
                    + d[4] * d[9] * d[14] * d[3]
                    + d[8] * d[13] * d[2] * d[7]
                    + d[12] * d[1] * d[6] * d[11]
                    - d[0] * d[13] * d[10] * d[7]
                    - d[4] * d[1] * d[14] * d[11]
                    - d[8] * d[5] * d[2] * d[15]
                    - d[12] * d[9] * d[6] * d[3]
            }

            /// Transpose the matrix
            pub fn transpose(&self) -> Self {
                let d = self.data; // code shortener.
                Self {
                    data: [
                        d[0], d[4], d[8], d[12], d[1], d[5], d[9], d[13], d[2], d[6], d[10], d[14],
                        d[3], d[7], d[11], d[15],
                    ],
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
                            0 => Mat3::from_cols(
                                t_y.truncate_n(j),
                                t_z.truncate_n(j),
                                t_w.truncate_n(j),
                            ),
                            1 => Mat3::from_cols(
                                t_x.truncate_n(j),
                                t_z.truncate_n(j),
                                t_w.truncate_n(j),
                            ),
                            2 => Mat3::from_cols(
                                t_x.truncate_n(j),
                                t_y.truncate_n(j),
                                t_w.truncate_n(j),
                            ),
                            3 => Mat3::from_cols(
                                t_x.truncate_n(j),
                                t_y.truncate_n(j),
                                t_z.truncate_n(j),
                            ),
                            _ => panic!("out of range"),
                        };
                        let sign = if (i + j) & 1 == 1 { -1. } else { 1. };
                        mat.determinant() * sign * inv_det
                    };

                    Some(Mat4::new([
                        cf(0, 0),
                        cf(0, 1),
                        cf(0, 2),
                        cf(0, 3),
                        cf(1, 0),
                        cf(1, 1),
                        cf(1, 2),
                        cf(1, 3),
                        cf(2, 0),
                        cf(2, 1),
                        cf(2, 2),
                        cf(2, 3),
                        cf(3, 0),
                        cf(3, 1),
                        cf(3, 2),
                        cf(3, 3),
                    ]))
                }
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

        #[cfg(feature = "std")]
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
        pub fn det_from_cols(c0: Vec3, c1: Vec3, c2: Vec3) -> $f {
            c0.x * c1.y * c2.z + c1.x * c2.y * c0.z + c2.x * c0.y * c1.z
                - c0.x * c2.y * c1.z
                - c1.x * c0.y * c2.z
                - c2.x * c1.y * c0.z
        }
    };
}
