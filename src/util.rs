pub fn linspace(start: f64, stop: f64, num_points: usize) -> Vec<f64> {
    if num_points < 2 {
        return vec![start];
    }

    let step = (stop - start) / (num_points - 1) as f64;
    (0..num_points).map(|i| start + i as f64 * step).collect()
}