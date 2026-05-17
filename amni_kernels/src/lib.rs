use pyo3::prelude::*;
pub mod nonce;
pub mod tokenizer;
pub mod tex_io;
pub mod resonance;
pub mod septidecimal;
pub mod prime_roots;
pub mod hilbert;
pub mod sept_diffusion;
pub mod sept_unet;
pub mod gf17_recurrent;
pub mod tmu_inference;
pub mod angel;
#[pymodule]
fn amni_kernels(m: &Bound<'_, PyModule>) -> PyResult<()> {
    nonce::register(m)?;
    tokenizer::register(m)?;
    tex_io::register(m)?;
    resonance::register(m)?;
    septidecimal::register(m)?;
    prime_roots::register(m)?;
    hilbert::register(m)?;
    sept_diffusion::register(m)?;
    sept_unet::register(m)?;
    gf17_recurrent::register(m)?;
    tmu_inference::register(m)?;
    angel::register(m)?;
    Ok(())
}
