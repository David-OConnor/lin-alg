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
        #[derive(Default, Clone, Copy, PartialEq)]
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

            pub fn splat(val: $f) -> Self {
                Self {
                    x: val,
                    y: val,
                    z: val,
                }
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

            /// Returns an arbitrary vector perpendicular to it.]
            pub fn any_perpendicular(&self) -> Self {
                let a = Self::new(1., 0., 0.); // This determines the resulting direction.
                self.cross(a)
            }

            pub fn min(self, other: Self) -> Self {
                Self {
                    x: self.x.min(other.x),
                    y: self.y.min(other.y),
                    z: self.z.min(other.z),
                }
            }

            pub fn max(self, other: Self) -> Self {
                Self {
                    x: self.x.max(other.x),
                    y: self.y.max(other.y),
                    z: self.z.max(other.z),
                }
            }
        }

        #[cfg(feature = "std")]
        impl fmt::Display for Vec3 {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                write!(f, "|{:.4}, {:.4}, {:.4}|", self.x, self.y, self.z)?;
                Ok(())
            }
        }

        #[derive(Clone, Copy, Debug, PartialEq)]
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

            pub fn splat(val: $f) -> Self {
                Self {
                    x: val,
                    y: val,
                    z: val,
                    w: val,
                }
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

        /// Calculate the dihedral angle between 4 positions. The positions must be in order by
        /// connection/bond, although both directions produce identical results. Compared to `calc_dihedral_angle()`,
        /// this function's API is more clear if you have the set of positions directly.
        pub fn calc_dihedral_angle_v2(posits: &(Vec3, Vec3, Vec3, Vec3)) -> $f {
            let mid = posits.1 - posits.2;
            let adj_next = posits.2 - posits.3;
            let adj_prev = posits.0 - posits.1;
            calc_dihedral_angle(mid, adj_next, adj_prev)
        }

        /// Calculate the dihedral angle between 4 positions (3 bonds). Compared to `calc_dihedral_angle_v2()`,
        /// this function's API is more clear if you have the bonds/connections, but not the positions.
        /// this might happen if you are doing certain vector operations, for example.
        ///
        /// The `bonds` are one position, subtracted from the next.
        ///
        /// Order matters. For posits 0-1-2-3, with connections 0-1, 1-2, 2-3:
        /// middle: posit 1 - posit 2.
        /// adj next: posit 2 - posit 3
        /// adj prev: posit 0 - posit 1
        pub fn calc_dihedral_angle(
            bond_middle: Vec3,
            bond_adj_next: Vec3,
            bond_adj_prev: Vec3,
        ) -> $f {
            // Project the next and previous bonds onto the plane that has this bond as its normal.
            // Re-normalize after projecting.
            let mid_norm = bond_middle.to_normalized();

            let bond1_on_plane = bond_adj_next.project_to_plane(mid_norm).to_normalized();
            let bond2_on_plane = -bond_adj_prev.project_to_plane(mid_norm).to_normalized();

            let result = bond1_on_plane.dot(bond2_on_plane).acos();

            // This can happen perhaps due to numerical precision problems on systems
            // where all 4 points are coplanar.
            // An alternative we could use instead of this check, starting at the line above:
            // x = cos φ,  y = sin φ  (see Allen & Tildesley §4.5)
            //let x = b1.dot(b2);
            //let y = b1.cross(b2).dot(mid_norm);

            //let mut φ = y.atan2(x);          // range (-π, π]
            //if φ < 0.0 { φ += TAU; }         // put it into [0, τ) if you prefer

            if result.is_nan() {
                return 0.;
            }

            // The dot product approach to angles between vectors only covers half of possible
            // rotations; use a determinant of the 3 vectors as matrix columns to determine if what we
            // need to modify is on the second half.
            let det = det_from_cols(bond1_on_plane, bond2_on_plane, mid_norm);

            if det < 0. { result } else { TAU - result }
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
        /// Convert a collection of `Vec3` into Cuda arrays of float3.
        /// Note: Ignore resources that say you need to pad to 16-bytes to map to float3; they're wrong.
        pub fn vec3s_to_dev(stream: &Arc<CudaStream>, data_host: &[Vec3]) -> CudaSlice<$f> {
            let mut result = Vec::with_capacity(data_host.len() * 3);

            // todo: Ref etcs A/R; you are making a double copy here.
            for v in data_host {
                result.push(v.x as $f);
                result.push(v.y as $f);
                result.push(v.z as $f);
            }
            stream.memcpy_stod(&result).unwrap()
        }

        #[cfg(feature = "cuda")]
        /// Convert a Cuda array of `float3` into Vec3.
        /// Note: Ignore resources that say you need to pad to 16-bytes to map to float3; they're wrong.
        pub fn vec3s_from_dev(stream: &Arc<CudaStream>, data_dev: &CudaSlice<$f>) -> Vec<Vec3> {
            let data_host = stream.memcpy_dtov(data_dev).unwrap();

            data_host
                .chunks_exact(3)
                .map(|chunk| Vec3::new(chunk[0], chunk[1], chunk[2]))
                .collect()
        }

        #[cfg(feature = "cuda")]
        unsafe impl cudarc::driver::DeviceRepr for Vec3 {}
    };
}
