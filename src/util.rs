use num_traits::Float;

/// Create a set of values in a given range, with a given number of values.
/// Similar to `numpy.linspace`.
/// The result terminates one step before the end of the range.
pub fn linspace<T>(start: T, stop: T, num_points: usize) -> Vec<T> where T: Float{
    if num_points < 2 {
        return vec![start];
    }

    let step = (stop - start) / T::from(num_points - 1).unwrap();
    (0..num_points).map(|i| start + T::from(i).unwrap() * step).collect()
}

pub fn logspace(mut start: f64, stop: f64, num_points: usize) -> Vec<f64> {
    if num_points < 2 {
        return vec![start.exp()];
    }

    if start < f64::EPSILON {
        start = 0.0001; // todo?
    }

    let log_start = start.ln();
    let log_stop = stop.ln();
    let step = (log_stop - log_start) / (num_points - 1) as f64;

    (0..num_points)
        .map(|i| (log_start + i as f64 * step).exp())
        .collect()
}


// todo: Consider if these functions should be aliased to f32 and f64, user generics, or otherwise.

// /// Linearly map an input value to an output.
// pub fn map_linear(val: f32, range_in: (f32, f32), range_out: (f32, f32)) -> f32 {
//     // todo: You may be able to optimize calls to this by having the ranges pre-store
//     // todo the total range vals.
//     let portion = (val - range_in.0) / (range_in.1 - range_in.0);
//
//     portion * (range_out.1 - range_out.0) + range_out.0
// }

/// Linearly map an input value to an output
pub fn map_linear<T>(val: T, range_in: (T, T), range_out: (T, T)) -> T
where
    T: Float,
{
    let portion = (val - range_in.0) / (range_in.1 - range_in.0);
    portion * (range_out.1 - range_out.0) + range_out.0
}