use pyo3::prelude::*;
use rayon::prelude::*;
use sha2::{Sha256, Digest};
const PRIMES: [u64; 32] = [
    2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,
    59,61,67,71,73,79,83,89,97,101,103,107,109,113,127,131
];
const LOG_PRIMES: [f64; 32] = [
    0.6931471805599453, 1.0986122886681098, 1.6094379124341003,
    1.9459101090932196, 2.3978952727983706, 2.5649493574615367,
    2.833213344056216, 2.9444389791664407, 3.1354942159291497,
    3.367295829986474, 3.4339872044851267, 3.6109179126442243,
    3.713572066704308, 3.7612001156935624, 3.8501476017100584,
    3.970291913552122, 4.07753744390572, 4.110873864173311,
    4.204692619390966, 4.2626798770413155, 4.290459441148391,
    4.3694478524670215, 4.4188406077965983, 4.48863636522361,
    4.574710978503479, 4.615120516934824, 4.634728988229636,
    4.672828834461906, 4.6913478822291435, 4.727387818712341,
    4.844187086458591, 4.875197323201151
];
fn spectrum_bytes_inner(hash_bytes: &[u8], n_primes: usize) -> Vec<f64> {
    let np = n_primes.min(32);
    let mut spectrum = vec![0.0f64; np];
    for &byte_val in hash_bytes {
        if byte_val <= 1 { continue; }
        let mut remaining = byte_val as u64;
        for i in 0..np {
            let p = PRIMES[i];
            if p > remaining { break; }
            let mut count = 0u32;
            while remaining % p == 0 {
                count += 1;
                remaining /= p;
            }
            if count > 0 {
                spectrum[i] += (1.0 + count as f64).ln() / LOG_PRIMES[i];
            }
        }
    }
    let norm = spectrum.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > 1e-10 { spectrum.iter_mut().for_each(|x| *x /= norm); }
    spectrum
}
#[pyfunction]
fn prime_log_spectrum_bytes<'py>(
    py: Python<'py>,
    hash_bytes: Vec<u8>,
    n_primes: usize,
) -> Bound<'py, numpy::PyArray1<f64>> {
    let spec = spectrum_bytes_inner(&hash_bytes, n_primes);
    numpy::PyArray1::from_vec(py, spec)
}
#[pyfunction]
fn batch_prime_spectra<'py>(
    py: Python<'py>,
    words: Vec<String>,
    n_primes: usize,
    seed: u32,
) -> Bound<'py, numpy::PyArray1<f64>> {
    let np = n_primes.min(32);
    let results: Vec<Vec<f64>> = words.par_iter().map(|w| {
        let input = format!("{}:{}", seed, w.to_lowercase().trim());
        let hash = Sha256::digest(input.as_bytes());
        spectrum_bytes_inner(&hash, np)
    }).collect();
    let flat: Vec<f64> = results.into_iter().flatten().collect();
    numpy::PyArray1::from_vec(py, flat)
}
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(prime_log_spectrum_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(batch_prime_spectra, m)?)?;
    Ok(())
}
