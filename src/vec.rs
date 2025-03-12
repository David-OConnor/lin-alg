#![macro_use]
macro_rules! create_vec {
    ($f:ident) => {
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

            /// Calculate the cross product.
            pub fn cross(&self, rhs: Self) -> Self {
                Self {
                    x: self.y * rhs.z - self.z * rhs.y,
                    y: self.z * rhs.x - self.x * rhs.z,
                    z: self.x * rhs.y - self.y * rhs.x,
                }
            }

            /// Returns a sum of all elements
            pub fn sum(&self) -> $f {
                self.x + self.y + self.z
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

        impl From<[$f; 4]> for Vec4 {
            fn from(v: [$f; 4]) -> Self {
                Self {
                    x: v[0],
                    y: v[1],
                    z: v[2],
                    w: v[3],
                }
            }
        }

        impl Add<Self> for Vec4 {
            type Output = Self;

            fn add(self, rhs: Self) -> Self::Output {
                Self {
                    x: self.x + rhs.x,
                    y: self.y + rhs.y,
                    z: self.z + rhs.z,
                    w: self.w + rhs.w,
                }
            }
        }

        impl AddAssign<Self> for Vec4 {
            fn add_assign(&mut self, rhs: Self) {
                self.x = self.x + rhs.x;
                self.y = self.y + rhs.y;
                self.z = self.z + rhs.z;
                self.w = self.w + rhs.w;
            }
        }

        impl Sub<Self> for Vec4 {
            type Output = Self;

            fn sub(self, rhs: Self) -> Self::Output {
                Self {
                    x: self.x - rhs.x,
                    y: self.y - rhs.y,
                    z: self.z - rhs.z,
                    w: self.w - rhs.w,
                }
            }
        }

        impl SubAssign<Self> for Vec4 {
            fn sub_assign(&mut self, rhs: Self) {
                self.x = self.x - rhs.x;
                self.y = self.y - rhs.y;
                self.z = self.z - rhs.z;
                self.w = self.w - rhs.w;
            }
        }

        impl Mul<$f> for Vec4 {
            type Output = Self;

            fn mul(self, rhs: $f) -> Self::Output {
                Self {
                    x: self.x * rhs,
                    y: self.y * rhs,
                    z: self.z * rhs,
                    w: self.w * rhs,
                }
            }
        }

        impl MulAssign<$f> for Vec4 {
            fn mul_assign(&mut self, rhs: $f) {
                self.x *= rhs;
                self.y *= rhs;
                self.z *= rhs;
                self.w *= rhs;
            }
        }

        impl Div<$f> for Vec4 {
            type Output = Self;

            fn div(self, rhs: $f) -> Self::Output {
                Self {
                    x: self.x / rhs,
                    y: self.y / rhs,
                    z: self.z / rhs,
                    w: self.w / rhs,
                }
            }
        }

        impl DivAssign<$f> for Vec4 {
            fn div_assign(&mut self, rhs: $f) {
                self.x /= rhs;
                self.y /= rhs;
                self.z /= rhs;
                self.w /= rhs;
            }
        }

        impl Neg for Vec4 {
            type Output = Self;

            fn neg(self) -> Self::Output {
                Self {
                    x: -self.x,
                    y: -self.y,
                    z: -self.z,
                    w: -self.w,
                }
            }
        }

        impl Vec4 {
            pub fn new(x: $f, y: $f, z: $f, u: $f) -> Self {
                Self { x, y, z, w: u }
            }

            pub const fn new_zero() -> Self {
                Self {
                    x: 0.,
                    y: 0.,
                    z: 0.,
                    w: 0.,
                }
            }

            /// Construct from the first 4 values in a slice: &[x, y, z, w].
            pub fn from_slice(slice: &[$f]) -> Result<Self, crate::BufError> {
                if slice.len() < 4 {
                    return Err(crate::BufError {});
                }
                Ok(Self {
                    x: slice[0],
                    y: slice[1],
                    z: slice[2],
                    w: slice[3],
                })
            }

            /// Convert to a len-4 array: [x, y, z, w].
            pub fn to_arr(&self) -> [$f; 4] {
                [self.x, self.y, self.z, self.w]
            }

            /// Calculates the Hadamard product (element-wise multiplication).
            pub fn hadamard_product(self, rhs: Self) -> Self {
                Self {
                    x: self.x * rhs.x,
                    y: self.y * rhs.y,
                    z: self.z * rhs.z,
                    w: self.w * rhs.w,
                }
            }

            /// Returns the vector magnitude squared
            pub fn magnitude_squared(self) -> $f {
                (self.hadamard_product(self)).sum()
            }

            /// Returns the vector magnitude
            pub fn magnitude(&self) -> $f {
                (self.x.powi(2) + self.y.powi(2) + self.z.powi(2) + self.w.powi(2)).sqrt()
            }

            pub fn normalize(&mut self) {
                let len =
                    (self.x.powi(2) + self.y.powi(2) + self.z.powi(2) + self.w.powi(2)).sqrt();

                self.x /= len;
                self.y /= len;
                self.z /= len;
                self.w /= len;
            }

            /// Returns the normalised version of the vector
            pub fn to_normalized(self) -> Self {
                let mag_recip = 1. / self.magnitude();
                self * mag_recip
            }

            /// Remove the nth element. Used in our inverse calculations.
            pub fn truncate_n(&self, n: usize) -> Vec3 {
                match n {
                    0 => Vec3::new(self.y, self.z, self.w),
                    1 => Vec3::new(self.x, self.z, self.w),
                    2 => Vec3::new(self.x, self.y, self.w),
                    3 => Vec3::new(self.x, self.y, self.z),
                    _ => panic!("{:?} is out of range", n),
                }
            }

            /// Returns the dot product with another vector.
            pub fn dot(&self, rhs: Self) -> $f {
                self.x * rhs.x + self.y * rhs.y + self.z * rhs.z + self.w * rhs.w
            }

            /// Returns a sum of all elements
            pub fn sum(&self) -> $f {
                self.x + self.y + self.z + self.w
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

        #[cfg(feature = "cuda")]
        /// Convert a collection of `Vec3`s into Cuda arrays of their components. Always outputs
        /// to an `f32` variant, for now.
        pub fn alloc_vec3s(dev: &Arc<CudaDevice>, data: &[Vec3]) -> CudaSlice<f32> {
            let mut result = Vec::new();
            // todo: Ref etcs A/R; you are making a double copy here.
            for v in data {
                result.push(v.x as f32);
                result.push(v.y as f32);
                result.push(v.z as f32);
            }
            dev.htod_copy(result).unwrap()
        }
    };
}
