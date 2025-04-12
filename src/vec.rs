#![macro_use]

// Agnostic to SIMD and non-simd
// `$f` here could be a primitive like `f32`, or a SIMD primitive like `f32x8`.
macro_rules! create_vec_shared {
    ($f:ident, $vec3_ty:ident, $vec4_ty:ident) => {
        impl $vec3_ty {
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
                self.x.powi(2) + self.y.powi(2) + self.z.powi(2)
            }

            /// Returns the vector magnitude
            pub fn magnitude(&self) -> $f {
                (self.x.powi(2) + self.y.powi(2) + self.z.powi(2)).sqrt()
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

        impl Default for $vec3_ty {
            fn default() -> Self {
                Self::new_zero()
            }
        }

        impl Add for $vec3_ty {
            type Output = Self;

            fn add(self, rhs: Self) -> Self::Output {
                Self {
                    x: self.x + rhs.x,
                    y: self.y + rhs.y,
                    z: self.z + rhs.z,
                }
            }
        }

        impl AddAssign for $vec3_ty {
            fn add_assign(&mut self, rhs: Self) {
                self.x += rhs.x;
                self.y += rhs.y;
                self.z += rhs.z;
            }
        }

        impl Add<$f> for $vec3_ty {
            type Output = Self;

            fn add(self, rhs: $f) -> Self::Output {
                Self {
                    x: self.x + rhs,
                    y: self.y + rhs,
                    z: self.z + rhs,
                }
            }
        }

        impl AddAssign<$f> for $vec3_ty {
            fn add_assign(&mut self, rhs: $f) {
                self.x += rhs;
                self.y += rhs;
                self.z += rhs;
            }
        }

        impl Sub for $vec3_ty {
            type Output = Self;

            fn sub(self, rhs: Self) -> Self::Output {
                Self {
                    x: self.x - rhs.x,
                    y: self.y - rhs.y,
                    z: self.z - rhs.z,
                }
            }
        }

        impl SubAssign for $vec3_ty {
            fn sub_assign(&mut self, rhs: Self) {
                self.x -= rhs.x;
                self.y -= rhs.y;
                self.z -= rhs.z;
            }
        }

        impl Sub<$f> for $vec3_ty {
            type Output = Self;

            fn sub(self, rhs: $f) -> Self::Output {
                Self {
                    x: self.x - rhs,
                    y: self.y - rhs,
                    z: self.z - rhs,
                }
            }
        }

        impl SubAssign<$f> for $vec3_ty {
            fn sub_assign(&mut self, rhs: $f) {
                self.x -= rhs;
                self.y -= rhs;
                self.z -= rhs;
            }
        }

        impl Mul<$f> for $vec3_ty {
            type Output = Self;

            fn mul(self, rhs: $f) -> Self::Output {
                Self {
                    x: self.x * rhs,
                    y: self.y * rhs,
                    z: self.z * rhs,
                }
            }
        }

        impl MulAssign<$f> for $vec3_ty {
            fn mul_assign(&mut self, rhs: $f) {
                self.x = self.x * rhs;
                self.y = self.y * rhs;
                self.z = self.z * rhs;
            }
        }

        impl Div<$f> for $vec3_ty {
            type Output = Self;

            fn div(self, rhs: $f) -> Self::Output {
                Self {
                    x: self.x / rhs,
                    y: self.y / rhs,
                    z: self.z / rhs,
                }
            }
        }

        impl DivAssign<$f> for $vec3_ty {
            fn div_assign(&mut self, rhs: $f) {
                self.x = self.x / rhs;
                self.y = self.y / rhs;
                self.z = self.z / rhs;
            }
        }

        impl $vec4_ty {
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
                self.x.powi(2) + self.y.powi(2) + self.z.powi(2) + self.w.powi(2)
            }

            /// Returns the vector magnitude
            pub fn magnitude(&self) -> $f {
                (self.x.powi(2) + self.y.powi(2) + self.z.powi(2) + self.w.powi(2)).sqrt()
            }

            /// Normalize, modifying in place.
            pub fn normalize(&mut self) {
                let mag = self.magnitude();

                self.x /= mag;
                self.y /= mag;
                self.z /= mag;
                self.w /= mag;
            }

            /// Returns the normalized version of the vector.
            pub fn to_normalized(self) -> Self {
                self / self.magnitude()
            }

            /// Returns the dot product with another vector.
            pub fn dot(&self, rhs: Self) -> $f {
                self.x * rhs.x + self.y * rhs.y + self.z * rhs.z + self.w * rhs.w
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

        impl Default for $vec4_ty {
            fn default() -> Self {
                Self::new_zero()
            }
        }

        impl Add for $vec4_ty {
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

        impl AddAssign for $vec4_ty {
            fn add_assign(&mut self, rhs: Self) {
                self.x += rhs.x;
                self.y += rhs.y;
                self.z += rhs.z;
                self.w += rhs.w;
            }
        }

        impl Add<$f> for $vec4_ty {
            type Output = Self;

            fn add(self, rhs: $f) -> Self::Output {
                Self {
                    x: self.x + rhs,
                    y: self.y + rhs,
                    z: self.z + rhs,
                    w: self.w + rhs,
                }
            }
        }

        impl AddAssign<$f> for $vec4_ty {
            fn add_assign(&mut self, rhs: $f) {
                self.x += rhs;
                self.y += rhs;
                self.z += rhs;
                self.w += rhs;
            }
        }

        impl Sub for $vec4_ty {
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

        impl SubAssign for $vec4_ty {
            fn sub_assign(&mut self, rhs: Self) {
                self.x -= rhs.x;
                self.y -= rhs.y;
                self.z -= rhs.z;
                self.w -= rhs.w;
            }
        }

        impl Sub<$f> for $vec4_ty {
            type Output = Self;

            fn sub(self, rhs: $f) -> Self::Output {
                Self {
                    x: self.x - rhs,
                    y: self.y - rhs,
                    z: self.z - rhs,
                    w: self.w - rhs,
                }
            }
        }

        impl SubAssign<$f> for $vec4_ty {
            fn sub_assign(&mut self, rhs: $f) {
                self.x -= rhs;
                self.y -= rhs;
                self.z -= rhs;
                self.w -= rhs;
            }
        }

        impl Mul<$f> for $vec4_ty {
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

        impl MulAssign<$f> for $vec4_ty {
            fn mul_assign(&mut self, rhs: $f) {
                self.x = self.x * rhs;
                self.y = self.y * rhs;
                self.z = self.z * rhs;
                self.w = self.w * rhs;
            }
        }

        impl Div<$f> for $vec4_ty {
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

        impl DivAssign<$f> for $vec4_ty {
            fn div_assign(&mut self, rhs: $f) {
                self.x = self.x / rhs;
                self.y = self.y / rhs;
                self.z = self.z / rhs;
                self.w = self.w / rhs;
            }
        }
    };
}

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

            /// Returns the remaining scalar after removing the nth element (0-based).
            /// For example:
            /// truncate_n(0) => drops .x, returns y
            /// truncate_n(1) => drops .y, returns x
            pub fn truncate_n(&self, n: usize) -> $f {
                match n {
                    0 => self.y,
                    1 => self.x,
                    _ => panic!("{} is out of range for Vec2", n),
                }
            }
        }

        #[cfg(feature = "std")]
        impl fmt::Display for Vec2 {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                write!(f, "|{:.4}, {:.4}|", self.x, self.y)?;
                Ok(())
            }
        }

        #[derive(Clone, Copy, Debug, PartialEq)]
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

        impl Sum<Vec3> for Vec3 {
            fn sum<I: Iterator<Item = Vec3>>(iter: I) -> Vec3 {
                iter.fold(Vec3::new_zero(), |a, b| a + b)
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

            /// Returns this vector with the nth element removed (0-based).
            /// For example:
            /// truncate_n(0) => drops .x, returns Vec2(y, z)
            /// truncate_n(1) => drops .y, returns Vec2(x, z)
            /// truncate_n(2) => drops .z, returns Vec2(x, y)
            pub fn truncate_n(&self, n: usize) -> Vec2 {
                match n {
                    0 => Vec2::new(self.y, self.z),
                    1 => Vec2::new(self.x, self.z),
                    2 => Vec2::new(self.x, self.y),
                    _ => panic!("{} is out of range for Vec3", n),
                }
            }

            /// Returns a sum of all elements
            pub fn sum(&self) -> $f {
                self.x + self.y + self.z
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

        impl Sum<Vec4> for Vec4 {
            fn sum<I: Iterator<Item = Vec4>>(iter: I) -> Vec4 {
                iter.fold(Vec4::new_zero(), |a, b| a + b)
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

            /// Returns a sum of all elements
            pub fn sum(&self) -> $f {
                self.x + self.y + self.z + self.w
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
        /// Convert a collection of `Vec3`s into Cuda arrays of their components.
        pub fn alloc_vec3s(stream: &Arc<CudaStream>, data: &[Vec3]) -> CudaSlice<$f> {
            let mut result = Vec::new();
            // todo: Ref etcs A/R; you are making a double copy here.
            for v in data {
                result.push(v.x as $f);
                result.push(v.y as $f);
                result.push(v.z as $f);
            }
            stream.memcpy_stod(&result).unwrap()
        }
    };
}
