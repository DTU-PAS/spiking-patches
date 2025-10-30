mod continuous_spiking_patch;
mod continuous_tokenizer;
mod discrete_spiking_patch;
mod discrete_tokenizer;
mod event;
mod token;
mod voxel_tokenizer;

use token::{PyTokens, Tokens};

use numpy::ndarray::{ArrayBase, Dim, ViewRepr};
use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::prelude::*;
use rayon::prelude::*;

type Location<'a> = ArrayBase<ViewRepr<&'a u16>, Dim<[usize; 1]>>;
type Time<'a> = ArrayBase<ViewRepr<&'a u64>, Dim<[usize; 1]>>;
type Polarity<'a> = ArrayBase<ViewRepr<&'a bool>, Dim<[usize; 1]>>;

#[pymodule]
fn spiking_patches(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyContinuousBatchTokenizer>()?;
    m.add_class::<PyContinuousStreamingTokenizer>()?;
    m.add_class::<PyDiscreteBatchTokenizer>()?;
    m.add_class::<PyDiscreteStreamingTokenizer>()?;
    m.add_class::<PyVoxelTokenizer>()?;
    Ok(())
}

#[pyclass(name = "ContinuousBatchTokenizer")]
struct PyContinuousBatchTokenizer {
    absolute_refractory_period: u64,
    decay: f64,
    height: usize,
    patch_size: usize,
    relative_refractory_period: u64,
    relative_refractory_scale: f64,
    spike_threshold: f64,
    width: usize,
}

#[pymethods]
impl PyContinuousBatchTokenizer {
    #[new]
    fn new(
        absolute_refractory_period: u64,
        decay: f64,
        height: usize,
        patch_size: usize,
        relative_refractory_period: u64,
        relative_refractory_scale: f64,
        spike_threshold: f64,
        width: usize,
    ) -> Self {
        PyContinuousBatchTokenizer {
            absolute_refractory_period,
            decay,
            height,
            patch_size,
            relative_refractory_period,
            relative_refractory_scale,
            spike_threshold,
            width,
        }
    }

    fn tokenize_batch<'py>(
        &self,
        py: Python<'py>,
        batch: Vec<(
            PyReadonlyArray1<u16>,
            PyReadonlyArray1<u16>,
            PyReadonlyArray1<u64>,
            PyReadonlyArray1<bool>,
        )>,
    ) -> Vec<PyTokens<'py>> {
        let batch: Vec<(Location, Location, Time, Polarity)> = batch
            .iter()
            .map(|(x, y, t, p)| {
                let x = x.as_array();
                let y = y.as_array();
                let t = t.as_array();
                let p = p.as_array();
                (x, y, t, p)
            })
            .collect();

        let batch: Vec<Tokens> = py.allow_threads(|| {
            batch
                .into_par_iter()
                .map(|(x, y, t, p)| {
                    let mut tokenizer = continuous_tokenizer::Tokenizer::new(
                        self.absolute_refractory_period,
                        self.decay,
                        self.height,
                        self.patch_size,
                        self.relative_refractory_period,
                        self.relative_refractory_scale,
                        self.spike_threshold,
                        self.width,
                    );
                    tokenizer.tokenize(x, y, t, p)
                })
                .collect()
        });

        batch
            .into_iter()
            .map(|tokenizer_output| tokens_to_python(py, tokenizer_output))
            .collect()
    }
}

#[pyclass(name = "ContinuousStreamingTokenizer")]
struct PyContinuousStreamingTokenizer {
    tokenizer: continuous_tokenizer::Tokenizer,
}

#[pymethods]
impl PyContinuousStreamingTokenizer {
    #[new]
    fn new(
        absolute_refractory_period: u64,
        decay: f64,
        height: usize,
        patch_size: usize,
        relative_refractory_period: u64,
        relative_refractory_scale: f64,
        spike_threshold: f64,
        width: usize,
    ) -> Self {
        let tokenizer = continuous_tokenizer::Tokenizer::new(
            absolute_refractory_period,
            decay,
            height,
            patch_size,
            relative_refractory_period,
            relative_refractory_scale,
            spike_threshold,
            width,
        );

        PyContinuousStreamingTokenizer { tokenizer }
    }

    fn reset(&mut self) {
        self.tokenizer.reset();
    }

    fn stream<'py>(
        &mut self,
        py: Python<'py>,
        x: PyReadonlyArray1<u16>,
        y: PyReadonlyArray1<u16>,
        t: PyReadonlyArray1<u64>,
        p: PyReadonlyArray1<bool>,
    ) -> PyTokens<'py> {
        let x = x.as_array();
        let y = y.as_array();
        let t = t.as_array();
        let p = p.as_array();

        let tokens = self.tokenizer.tokenize(x, y, t, p);
        tokens_to_python(py, tokens)
    }
}

#[pyclass(name = "DiscreteBatchTokenizer")]

struct PyDiscreteBatchTokenizer {
    decay: u64,
    delay: u64,
    height: usize,
    patch_size: usize,
    spike_threshold: usize,
    width: usize,
}

#[pymethods]
impl PyDiscreteBatchTokenizer {
    #[new]
    fn new(
        decay: u64,
        delay: u64,
        height: usize,
        patch_size: usize,
        spike_threshold: usize,
        width: usize,
    ) -> Self {
        PyDiscreteBatchTokenizer {
            decay,
            delay,
            height,
            patch_size,
            spike_threshold,
            width,
        }
    }

    fn tokenize_batch<'py>(
        &self,
        py: Python<'py>,
        batch: Vec<(
            PyReadonlyArray1<u16>,
            PyReadonlyArray1<u16>,
            PyReadonlyArray1<u64>,
            PyReadonlyArray1<bool>,
        )>,
    ) -> Vec<PyTokens<'py>> {
        let batch: Vec<(Location, Location, Time, Polarity)> = batch
            .iter()
            .map(|(x, y, t, p)| {
                let x = x.as_array();
                let y = y.as_array();
                let t = t.as_array();
                let p = p.as_array();
                (x, y, t, p)
            })
            .collect();

        let batch: Vec<Tokens> = py.allow_threads(|| {
            batch
                .into_par_iter()
                .map(|(x, y, t, p)| {
                    let mut tokenizer = discrete_tokenizer::Tokenizer::new(
                        self.decay,
                        self.delay,
                        self.height,
                        self.patch_size,
                        self.spike_threshold,
                        self.width,
                    );
                    tokenizer.tokenize(x, y, t, p)
                })
                .collect()
        });

        batch
            .into_iter()
            .map(|tokenizer_output| tokens_to_python(py, tokenizer_output))
            .collect()
    }
}

#[pyclass(name = "DiscreteStreamingTokenizer")]
struct PyDiscreteStreamingTokenizer {
    tokenizer: discrete_tokenizer::Tokenizer,
}

#[pymethods]
impl PyDiscreteStreamingTokenizer {
    #[new]
    fn new(
        decay: u64,
        delay: u64,
        height: usize,
        patch_size: usize,
        spike_threshold: usize,
        width: usize,
    ) -> Self {
        let tokenizer = discrete_tokenizer::Tokenizer::new(
            decay,
            delay,
            height,
            patch_size,
            spike_threshold,
            width,
        );

        PyDiscreteStreamingTokenizer { tokenizer }
    }

    fn reset(&mut self) {
        self.tokenizer.reset();
    }

    fn stream<'py>(
        &mut self,
        py: Python<'py>,
        x: PyReadonlyArray1<u16>,
        y: PyReadonlyArray1<u16>,
        t: PyReadonlyArray1<u64>,
        p: PyReadonlyArray1<bool>,
    ) -> PyTokens<'py> {
        let x = x.as_array();
        let y = y.as_array();
        let t = t.as_array();
        let p = p.as_array();

        let tokens = self.tokenizer.tokenize(x, y, t, p);
        tokens_to_python(py, tokens)
    }
}

#[pyclass(name = "VoxelTokenizer")]

struct PyVoxelTokenizer {
    tokenizer: voxel_tokenizer::Tokenizer,
}

#[pymethods]
impl PyVoxelTokenizer {
    #[new]
    fn new(
        duration_us: u64,
        height: usize,
        patch_size: usize,
        threshold: usize,
        width: usize,
    ) -> Self {
        let tokenizer =
            voxel_tokenizer::Tokenizer::new(duration_us, height, patch_size, threshold, width);

        PyVoxelTokenizer { tokenizer }
    }

    fn reset(&mut self) {
        // nothing to reset because the tokenizer is stateless
        // but we implement this method to match the interface
    }

    fn stream<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray1<u16>,
        y: PyReadonlyArray1<u16>,
        t: PyReadonlyArray1<u64>,
        p: PyReadonlyArray1<bool>,
    ) -> PyTokens<'py> {
        let x = x.as_array();
        let y = y.as_array();
        let t = t.as_array();
        let p = p.as_array();

        let tokens = self.tokenizer.tokenize(x, y, t, p);
        tokens_to_python(py, tokens)
    }

    fn tokenize_batch<'py>(
        &self,
        py: Python<'py>,
        batch: Vec<(
            PyReadonlyArray1<u16>,
            PyReadonlyArray1<u16>,
            PyReadonlyArray1<u64>,
            PyReadonlyArray1<bool>,
        )>,
    ) -> Vec<PyTokens<'py>> {
        let batch: Vec<(Location, Location, Time, Polarity)> = batch
            .iter()
            .map(|(x, y, t, p)| {
                let x = x.as_array();
                let y = y.as_array();
                let t = t.as_array();
                let p = p.as_array();
                (x, y, t, p)
            })
            .collect();

        let batch: Vec<Tokens> = py.allow_threads(|| {
            batch
                .into_par_iter()
                .map(|(x, y, t, p)| self.tokenizer.tokenize(x, y, t, p))
                .collect()
        });

        batch
            .into_iter()
            .map(|tokenizer_output| tokens_to_python(py, tokenizer_output))
            .collect()
    }
}

fn tokens_to_python<'py>(py: Python<'py>, tokens: Tokens) -> PyTokens<'py> {
    let x = tokens.x.into_pyarray_bound(py);
    let y = tokens.y.into_pyarray_bound(py);
    let t = tokens.t.into_pyarray_bound(py);

    let events_x = tokens
        .events_x
        .into_iter()
        .map(|arr| arr.into_pyarray_bound(py))
        .collect::<Vec<_>>();

    let events_y = tokens
        .events_y
        .into_iter()
        .map(|arr| arr.into_pyarray_bound(py))
        .collect::<Vec<_>>();

    let events_t = tokens
        .events_t
        .into_iter()
        .map(|arr| arr.into_pyarray_bound(py))
        .collect::<Vec<_>>();

    let events_p = tokens
        .events_p
        .into_iter()
        .map(|arr| arr.into_pyarray_bound(py))
        .collect::<Vec<_>>();

    (x, y, t, events_x, events_y, events_t, events_p)
}
