/// Create a set of values in a given range, with a given number of values.
/// Similar to `numpy.linspace`.
/// The result terminates one step before the end of the range.
pub fn linspace(start: f64, stop: f64, num_points: usize) -> Vec<f64> {
    if num_points < 2 {
        return vec![start];
    }

    let step = (stop - start) / (num_points - 1) as f64;
    (0..num_points).map(|i| start + i as f64 * step).collect()
}

// todo: Consider if these functions should be aliased to f32 and f64, user generics, or otherwise.

/// Linearly map an input value to an output.
pub fn map_linear(val: f32, range_in: (f32, f32), range_out: (f32, f32)) -> f32 {
    // todo: You may be able to optimize calls to this by having the ranges pre-store
    // todo the total range vals.
    let portion = (val - range_in.0) / (range_in.1 - range_in.0);

    portion * (range_out.1 - range_out.0) + range_out.0
}

// /// Linearly map an input value to an output
// pub fn map_linear<T>(val: T, range_in: (T, T), range_out: (T, T)) -> T
// where
//     T: num_traits::Float,
// {
//     let portion = (val - range_in.0) / (range_in.1 - range_in.0);
//     portion * (range_out.1 - range_out.0) + range_out.0
// }
