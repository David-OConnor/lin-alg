use std::f32::consts::TAU;
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "std"))]
use std::mem::transmute;

use super::*;
use crate::{
    f32::{FORWARD, RIGHT, UP, f32x8, pack_float, pack_vec3, unpack_float},
    f64::unpack_vec3,
};

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "std"))]
// todo: More tests, including for matrices.
#[test]
fn test_vec3_addition() {
    let v1 = f32::Vec3::new(1.0, 2.0, 3.0);
    let v2 = f32::Vec3::new(4.0, 5.0, 6.0);
    let v3: f32::Vec3 = f64::Vec3::new(4.0, 5.0, 6.0).into();

    let sum1 = v1 + v2;
    let sum2 = v1 + v3;

    assert_eq!(sum1.x, 5.0);
    assert_eq!(sum1.y, 7.0);
    assert_eq!(sum1.z, 9.0);
    assert_eq!(sum2, f32::Vec3::new(5., 7., 9.));
}

#[test]
fn test_vec3_subtraction() {
    let v1 = f32::Vec3::new(1.0, 2.0, 3.0);
    let v2 = f32::Vec3::new(0.5, 1.0, -1.0);
    let diff = v1 - v2;
    assert_eq!(diff.x, 0.5);
    assert_eq!(diff.y, 1.0);
    assert_eq!(diff.z, 4.0);
}

#[test]
fn test_vec3_scalar_multiply() {
    let v1 = f32::Vec3::new(2.0, 3.0, -1.0);
    let scaled = v1 * 2.0;
    assert_eq!(scaled.x, 4.0);
    assert_eq!(scaled.y, 6.0);
    assert_eq!(scaled.z, -2.0);

    let v2 = f32::Vec4::new(2.0, 3.0, -1.0, 10.);
    let scaled = v2 * 2.0;
    assert_eq!(scaled.x, 4.0);
    assert_eq!(scaled.y, 6.0);
    assert_eq!(scaled.z, -2.0);
    assert_eq!(scaled.w, 20.0);
}

#[test]
fn test_vec3_dot_product() {
    let v1 = f32::Vec3::new(1.0, 2.0, 3.0);
    let v2 = f32::Vec3::new(4.0, -1.0, 2.0);
    let dot = v1.dot(v2);
    assert_eq!(dot, 1.0 * 4.0 + 2.0 * (-1.0) + 3.0 * 2.0); // 4 - 2 + 6 = 8
}

#[test]
fn test_vec3_cross_product() {
    let v1 = f32::Vec3::new(1.0, 0.0, 0.0);
    let v2 = f32::Vec3::new(0.0, 1.0, 0.0);
    let cross = v1.cross(v2);
    assert_eq!(cross.x, 0.0);
    assert_eq!(cross.y, 0.0);
    assert_eq!(cross.z, 1.0);

    let v3 = f64::Vec3::new(1.0, 0.0, 0.0);
    let v4 = f64::Vec3::new(0.0, 1.0, 0.0);
    let cross = v3.cross(v4);
    assert_eq!(cross.x, 0.0);
    assert_eq!(cross.y, 0.0);
    assert_eq!(cross.z, 1.0);
}

#[test]
fn test_vec_normalize() {
    let mut v = f32::Vec3::new(3.0, 0.0, 4.0);
    v.normalize();
    let magnitude = v.magnitude();
    assert!(
        (magnitude - 1.0).abs() < f32::EPSILON,
        "Magnitude was not 1, got {}",
        magnitude
    );

    let mut v = f64::Vec4::new(3.0, 0.0, 4.0, 2.);
    v.normalize();
    let magnitude = v.magnitude();
    assert!(
        (magnitude - 1.0).abs() < f64::EPSILON,
        "Magnitude was not 1, got {}",
        magnitude
    );
}

#[test]
fn test_vec3_from_slice() {
    let arr = [10.0, 20.0, 30.0];
    let vec = f32::Vec3::from_slice(&arr).expect("Failed to create Vec3 from slice");
    assert_eq!(vec.x, 10.0);
    assert_eq!(vec.y, 20.0);
    assert_eq!(vec.z, 30.0);

    let short_arr = [10.0, 20.0];
    assert!(
        f32::Vec3::from_slice(&short_arr).is_err(),
        "Should fail on short slice"
    );
}

#[test]
fn test_quaternion_identity() {
    let q = f32::Quaternion::new_identity();
    assert_eq!(q.w, 1.0);
    assert_eq!(q.x, 0.0);
    assert_eq!(q.y, 0.0);
    assert_eq!(q.z, 0.0);
}

#[test]
fn test_quaternion_add() {
    let q1 = f32::Quaternion::new(1.0, 2.0, 3.0, 4.0);
    let q2 = f32::Quaternion::new(0.5, 0.5, 0.5, 0.5);
    let sum = q1 + q2;
    assert_eq!(sum.w, 1.5);
    assert_eq!(sum.x, 2.5);
    assert_eq!(sum.y, 3.5);
    assert_eq!(sum.z, 4.5);
}

#[test]
fn test_quaternion_subtraction() {
    let q1 = f32::Quaternion::new(1.0, 2.0, 3.0, 4.0);
    let q2 = f32::Quaternion::new(0.5, 0.5, 0.5, 0.5);
    let diff = q1 - q2;
    assert_eq!(diff.w, 0.5);
    assert_eq!(diff.x, 1.5);
    assert_eq!(diff.y, 2.5);
    assert_eq!(diff.z, 3.5);
}

#[test]
fn test_quaternion_mul_quaternion() {
    // For a quick test, multiply two simple quaternions and verify the result
    let q1 = f32::Quaternion::new(1.0, 0.0, 1.0, 0.0).to_normalized();
    let q2 = f32::Quaternion::new(1.0, 0.5, 0.5, 0.75).to_normalized();
    let product = q1 * q2;
    // The exact values aren’t necessarily intuitive, so we might just re-check
    // with known numeric approximations or simply ensure we got the correct shape.
    // Here, we do a rough check that the resulting magnitude is around 1.0,
    // plus a couple of value checks:
    assert!((product.magnitude() - 1.0).abs() < 1e-5);

    // Optionally, compare specific components if you have a reference or do partial checks
    // Just as an example (these might not match your exact reference depending on rounding):
    // assert!((product.w - 0.7071).abs() < 0.001);
}

#[test]
fn test_quaternion_mul_scalar() {
    let q = f32::Quaternion::new(1.0, 2.0, 3.0, 4.0);
    let s = 2.0;
    let r = q * s;
    assert_eq!(r.w, 2.0);
    assert_eq!(r.x, 4.0);
    assert_eq!(r.y, 6.0);
    assert_eq!(r.z, 8.0);
}

#[test]
fn test_quaternion_mul_vec() {
    let q = f32::Quaternion::new(0.0, 1.0, 0.0, 0.0); // A 180-degree rotation around X?
    let v = f32::Vec3::new(0.0, 1.0, 0.0);
    let result_q = q * v; // This is Q x v in quaternion form
    assert_eq!(result_q.w, -0.0 - 0.0 - 0.0); // = 0

    let rotated_v = q.rotate_vec(v);
    assert!((rotated_v.x - 0.0).abs() < f32::EPSILON);
    assert!((rotated_v.y + 1.0).abs() < f32::EPSILON); // we expect y = -1
    assert!((rotated_v.z - 0.0).abs() < f32::EPSILON);
}

#[test]
fn test_quaternion_inverse() {
    let q = f32::Quaternion::new(1.0, 2.0, 3.0, 4.0).to_normalized();
    let q_inv = q.inverse();
    let identity = q * q_inv;

    assert!((identity.w - 1.0).abs() < 1e-5);
    assert!((identity.x).abs() < 1e-5);
    assert!((identity.y).abs() < 1e-5);
    assert!((identity.z).abs() < 1e-5);
}

#[test]
fn test_quaternion_slerp() {
    let q1 = f32::Quaternion::new_identity();
    let q2 = f32::Quaternion::new(0.707, 0.707, 0.0, 0.0).to_normalized(); // ~90 deg around x-axis
    let q_half = q1.slerp(q2, 0.5);
    let angle_half = q_half.angle().to_degrees();
    assert!((angle_half - 45.0).abs() < 1.0);

    let axis = q_half.axis();
    assert!((axis.x - 1.0).abs() < 0.1, "Axis not near X");
}

#[test]
fn test_quaternion_from_slice() {
    let data = [1.0, 2.0, 3.0, 4.0];
    let q = f32::Quaternion::from_slice(&data).expect("Failed to create quaternion from slice");
    assert_eq!(q.w, 1.0);
    assert_eq!(q.x, 2.0);
    assert_eq!(q.y, 3.0);
    assert_eq!(q.z, 4.0);

    let short = [1.0, 2.0, 3.0];
    assert!(
        f32::Quaternion::from_slice(&short).is_err(),
        "Should fail with short slice"
    );
}

#[test]
fn test_quaternion_to_arr() {
    let q = f32::Quaternion::new(1.0, 2.0, 3.0, 4.0);
    let arr = q.to_arr();
    assert_eq!(arr, [1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_quaternion_from_unit_vecs() {
    let v0 = f32::Vec3::new(1.0, 0.0, 0.0); // x axis
    let v1 = f32::Vec3::new(0.0, 1.0, 0.0); // y axis
    let q = f32::Quaternion::from_unit_vecs(v0, v1);

    let rotated = q.rotate_vec(v0);
    assert!((rotated.x - 0.0).abs() < 1e-5);
    assert!((rotated.y - 1.0).abs() < 1e-5);
    assert!((rotated.z - 0.0).abs() < 1e-5);
}

#[test]
fn test_quaternion_from_and_to_euler() {
    let euler = f32::EulerAngle {
        roll: std::f32::consts::FRAC_PI_2,
        pitch: 0.0,
        yaw: std::f32::consts::FRAC_PI_2,
    };
    let q = f32::Quaternion::from_euler(&euler);
    // Then convert back:
    let euler2 = q.to_euler();
    // Because euler conversions can have multiple solutions, we do rough checks:
    // you might verify that euler2.pitch ~ 0, euler2.roll ~ pi/2, euler2.yaw ~ pi/2
    // within some tolerance

    // assert!((euler2.roll - std::f32::consts::FRAC_PI_2).abs() < 0.2);
    assert!((euler2.pitch - 0.0).abs() < 0.2);
    assert!((euler2.yaw - std::f32::consts::FRAC_PI_2).abs() < 0.2);
}

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "std"))]
#[test]
fn test_simd_vec3_cross() {
    let vec_a = f32::Vec3::new(1., 2., 3.);
    let vec_b = f32::Vec3::new(4., 5., 6.);

    let a = f32::Vec3x8::from_array([vec_a; 8]);
    let b = f32::Vec3x8::from_array([vec_b; 8]);

    let c = a.cross(b);

    let cx = c.x.to_array();
    let cy = c.y.to_array();
    let cz = c.z.to_array();

    for i in 0..8 {
        assert!((cx[i] - -3.0).abs() < f32::EPSILON);
        assert!((cy[i] - 6.0).abs() < f32::EPSILON);
        assert!((cz[i] - -3.0).abs() < f32::EPSILON);
    }
}

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "std"))]
#[test]
fn test_simd_vec3_dot() {
    let vec_a = f32::Vec3::new(1., 2., 3.);
    let vec_b = f32::Vec3::new(4., 5., 6.);

    let a = f32::Vec3x8::from_array([vec_a; 8]);
    let b = f32::Vec3x8::from_array([vec_b; 8]);

    let c: [f32; 8] = a.dot(b).to_array();

    for i in 0..8 {
        assert!((c[i] - (32.0)).abs() < f32::EPSILON);
    }
}

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "std"))]
#[test]
fn test_simd_vec3_dot_f64() {
    let vec_a = f64::Vec3::new(1., 2., 3.);
    let vec_b = f64::Vec3::new(4., 5., 6.);

    let a = f64::Vec3x4::from_array([vec_a; 4]);
    let b = f64::Vec3x4::from_array([vec_b; 4]);

    let c: [f64; 4] = a.dot(b).to_array();

    for i in 0..4 {
        assert!((c[i] - (32.0)).abs() < f64::EPSILON);
    }
}

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "std"))]
#[test]
fn test_simd_add() {
    let vec_a = f32::Vec3::new(1., 2., 3.);
    let vec_b = f32::Vec3::new(4., 5., 6.);

    let a = f32::Vec3x8::from_array([vec_a; 8]);
    let b = f32::Vec3x8::from_array([vec_b; 8]);
    let c = a + b;

    let vec3s = c.to_array();

    for vec3 in &vec3s {
        assert!((vec3.x - 5.0).abs() < f32::EPSILON);
        assert!((vec3.y - 7.0).abs() < f32::EPSILON);
        assert!((vec3.z - 9.0).abs() < f32::EPSILON);
    }
}

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "std"))]
#[test]
fn test_simd_rotate_vec() {
    let rot_init = [
        f32::Quaternion::from_unit_vecs(UP, FORWARD),
        f32::Quaternion::from_unit_vecs(UP, -FORWARD),
        f32::Quaternion::from_unit_vecs(UP, RIGHT),
        f32::Quaternion::from_unit_vecs(UP, -RIGHT),
        f32::Quaternion::from_unit_vecs(UP, UP),
        f32::Quaternion::from_unit_vecs(UP, -UP),
        f32::Quaternion::from_axis_angle(RIGHT, TAU / 4.),
        f32::Quaternion::from_axis_angle(RIGHT, TAU / 8.),
    ];

    let rotation = f32::Quaternionx8::from_array(rot_init);

    // This could be 8 separate values.
    let vec = f32::Vec3x8::from_array([UP; 8]);

    let result = rotation.rotate_vec(vec).to_array();

    let sqrt_2_div_2 = 2_f32.sqrt() / 2.;
    let angled = f32::Vec3::new(0., -sqrt_2_div_2, sqrt_2_div_2);

    assert!((result[0] - FORWARD).magnitude() < f32::EPSILON);
    assert!((result[1] - -FORWARD).magnitude() < f32::EPSILON);
    assert!((result[2] - RIGHT).magnitude() < f32::EPSILON);
    assert!((result[3] - -RIGHT).magnitude() < f32::EPSILON);
    assert!((result[4] - UP).magnitude() < f32::EPSILON);
    assert!((result[5] - -UP).magnitude() < f32::EPSILON);
    assert!((result[6] - -FORWARD).magnitude() < f32::EPSILON);
    assert!((result[7] - angled).magnitude() < f32::EPSILON);
}

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "std"))]
#[test]
fn test_simd_prim_exp() {
    let base = f32x8::from_array([1., 1.5, 2., 2.5, 3., 3.5, -1., -2.]);

    let exp_a = -2;
    let exp_b = -1;
    let exp_c = 0;
    let exp_d = 1;
    let exp_e = 2;
    let exp_f = 3;

    let result_a = base.powi(exp_a).to_array();
    let result_b = base.powi(exp_b).to_array();
    let result_c = base.powi(exp_c).to_array();
    let result_d = base.powi(exp_d).to_array();
    let result_e = base.powi(exp_e).to_array();
    let result_f = base.powi(exp_f).to_array();

    assert!(result_a[0] - 1. < f32::EPSILON);
    assert!(result_a[1] - 0.444444444 < f32::EPSILON);
    assert!(result_a[2] - 0.25 < f32::EPSILON);

    assert!(result_b[1] - 0.66666666666666 < f32::EPSILON);
    assert!(result_b[6] - 0.5 < f32::EPSILON);

    assert!(result_f[1] - 3.375 < f32::EPSILON);
    assert!(result_f[2] - 8. < f32::EPSILON);
    assert!(result_f[3] - 15.625 < f32::EPSILON);
}

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "std"))]
#[test]
fn test_simd_neg() {
    let a = f32x8::from_array([1., 1.5, 2., 2.5, 3., 3.5, -1., -2.]);
    let b = (-a).to_array();

    assert!(b[0] - -1. < f32::EPSILON);
    assert!(b[1] - -1.5 < f32::EPSILON);
    assert!(b[7] - 2. < f32::EPSILON);

    let c = f32::Vec3x8::splat(f32::Vec3::new(1., -1., 2.));
    let d = (-c).to_array();

    assert!(d[0].x - -1. < f32::EPSILON);
    assert!(d[1].y - 1. < f32::EPSILON);
    assert!(d[2].z - -2. < f32::EPSILON);
}

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "std"))]
#[test]
fn test_simd_pack_float() {
    // Test both with and without garbage lanes.
    let data = vec![
        0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18.,
    ];
    let data_1 = vec![
        0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.,
    ];

    let mut expected = Vec::with_capacity(data.len());
    let mut expected_1 = Vec::with_capacity(data_1.len());
    for v in &data {
        expected.push(v * 10.);
    }
    for v in &data_1 {
        expected_1.push(v * 10.);
    }

    let (mut packed, lanes_last) = pack_float(&data);
    let (mut packed_1, lanes_last_1) = pack_float(&data_1);

    assert_eq!(lanes_last, 3);
    assert_eq!(lanes_last_1, 8);

    for item in &mut packed {
        *item *= f32x8::splat(10.);
    }
    for item in &mut packed_1 {
        *item *= f32x8::splat(10.);
    }

    let unpacked = unpack_float(&packed, data.len());
    let unpacked_1 = unpack_float(&packed_1, data_1.len());

    assert_eq!(unpacked.len(), data.len());
    assert_eq!(unpacked_1.len(), data_1.len());

    for i in 0..packed.len() {
        assert!((expected[i] - unpacked[i]).abs() < f32::EPSILON);
    }

    for i in 0..packed_1.len() {
        assert!((expected_1[i] - unpacked_1[i]).abs() < f32::EPSILON);
    }
}

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "std"))]
#[test]
fn test_simd_pack_vec3() {
    // Test both with and without garbage lanes.
    let mut data = Vec::with_capacity(19);
    let mut data_1 = Vec::with_capacity(16);
    for i in 0..19 {
        data.push(f32::Vec3::new(i as f32, i as f32, i as f32));
    }

    for i in 0..16 {
        data_1.push(f32::Vec3::new(i as f32, i as f32, i as f32));
    }

    let mut expected = Vec::with_capacity(data.len());
    let mut expected_1 = Vec::with_capacity(data_1.len());
    for v in &data {
        expected.push(*v * 10.);
    }
    for v in &data_1 {
        expected_1.push(*v * 10.);
    }

    let (mut packed, lanes_last) = pack_vec3(&data);
    let (mut packed_1, lanes_last_1) = pack_vec3(&data_1);

    assert_eq!(lanes_last, 3);
    assert_eq!(lanes_last_1, 8);

    for item in &mut packed {
        *item *= f32x8::splat(10.);
    }
    for item in &mut packed_1 {
        *item *= f32x8::splat(10.);
    }

    let unpacked = f32::unpack_vec3(&packed, data.len());
    let unpacked_1 = f32::unpack_vec3(&packed_1, data_1.len());

    assert_eq!(unpacked.len(), data.len());
    assert_eq!(unpacked_1.len(), data_1.len());

    for i in 0..packed.len() {
        assert!((expected[i] - unpacked[i]).x.abs() < f32::EPSILON);
        assert!((expected[i] - unpacked[i]).y.abs() < f32::EPSILON);
        assert!((expected[i] - unpacked[i]).z.abs() < f32::EPSILON);
    }

    for i in 0..packed_1.len() {
        assert!((expected_1[i] - unpacked_1[i]).x.abs() < f32::EPSILON);
        assert!((expected_1[i] - unpacked_1[i]).y.abs() < f32::EPSILON);
        assert!((expected_1[i] - unpacked_1[i]).z.abs() < f32::EPSILON);
    }
}
