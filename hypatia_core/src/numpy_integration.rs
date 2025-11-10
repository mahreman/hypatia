use pyo3::prelude::*;
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyUntypedArrayMethods}; // Değişiklik yok
use crate::python_bindings::{PyMultiVector2D, PyMultiVector3D};

/// NumPy ile Hypatia entegrasyonu
pub fn register_numpy_integration(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // 2D NumPy fonksiyonları
    #[pyfunction]
    fn mv2d_from_array(array: &Bound<'_, PyArray1<f64>>) -> PyResult<PyMultiVector2D> {
        // DÜZELTME 1: unsafe blok eklendi
        let slice = unsafe { array.as_slice()? };
        if slice.len() != 4 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Array must have exactly 4 elements for MultiVector2D"
            ));
        }
        Ok(PyMultiVector2D::new(slice[0], slice[1], slice[2], slice[3]))
    }

    #[pyfunction]
    fn mv2d_to_array<'a>(mv: &PyMultiVector2D, py: Python<'a>) -> PyResult<Bound<'a, PyArray1<f64>>> {
        let array = PyArray1::from_slice_bound(py, &[
            mv.s(), 
            mv.e1(), 
            mv.e2(), 
            mv.e12()
        ]);
        Ok(array)
    }

    #[pyfunction]
    fn batch_rotate_2d<'a>(
        rotor: &PyMultiVector2D, 
        vectors: &Bound<'_, PyArray2<f64>>,
        py: Python<'a>
    ) -> PyResult<Bound<'a, PyArray2<f64>>> {
        // DÜZELTME 2: unsafe blok eklendi
        let vectors_slice = unsafe { vectors.as_slice()? };
        let shape = vectors.shape();
        
        if shape.len() != 2 || shape[1] != 2 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Input must be Nx2 array of 2D vectors"
            ));
        }

        let n = shape[0];
        let mut result = Vec::with_capacity(n * 2);
        
        for i in 0..n {
            let start = i * 2;
            let vector = PyMultiVector2D::vector(vectors_slice[start], vectors_slice[start + 1]);
            let rotated = rotor.rotate_vector(&vector);
            result.push(rotated.e1());
            result.push(rotated.e2());
        }

        // DÜZELTME 3: 1D Array oluşturup reshape() ile 2D'ye çevir
        let result_array = PyArray1::from_vec_bound(py, result).reshape((n, 2))?;
        Ok(result_array)
    }

    // 3D NumPy fonksiyonları
    #[pyfunction]
    fn mv3d_from_array(array: &Bound<'_, PyArray1<f64>>) -> PyResult<PyMultiVector3D> {
        // DÜZELTME 4: unsafe blok eklendi
        let slice = unsafe { array.as_slice()? };
        if slice.len() != 8 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Array must have exactly 8 elements for MultiVector3D"
            ));
        }
        Ok(PyMultiVector3D::new(
            slice[0], slice[1], slice[2], slice[3],
            slice[4], slice[5], slice[6], slice[7]
        ))
    }

    #[pyfunction]
    fn mv3d_to_array<'a>(mv: &PyMultiVector3D, py: Python<'a>) -> PyResult<Bound<'a, PyArray1<f64>>> {
        let array = PyArray1::from_slice_bound(py, &[
            mv.s(), 
            mv.e1(), 
            mv.e2(), 
            mv.e3(),
            mv.e12(),
            mv.e23(),
            mv.e31(),
            mv.e123()
        ]);
        Ok(array)
    }

    #[pyfunction]
    fn batch_rotate_3d<'a>(
        rotor: &PyMultiVector3D,
        vectors: &Bound<'_, PyArray2<f64>>,
        py: Python<'a>
    ) -> PyResult<Bound<'a, PyArray2<f64>>> {
        // DÜZELTME 5: unsafe blok eklendi
        let vectors_slice = unsafe { vectors.as_slice()? };
        let shape = vectors.shape();
        
        if shape.len() != 2 || shape[1] != 3 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Input must be Nx3 array of 3D vectors"
            ));
        }

        let n = shape[0];
        let mut result = Vec::with_capacity(n * 3);
        
        for i in 0..n {
            let start = i * 3;
            let vector = PyMultiVector3D::vector(
                vectors_slice[start],
                vectors_slice[start + 1],
                vectors_slice[start + 2]
            );
            
            let rotated = rotor.rotate_vector(&vector);
            result.push(rotated.e1());
            result.push(rotated.e2());
            result.push(rotated.e3());
        }

        // DÜZELTME 6: 1D Array oluşturup reshape() ile 3D'ye çevir
        let result_array = PyArray1::from_vec_bound(py, result).reshape((n, 3))?;
        Ok(result_array)
    }

    // Fonksiyonları modüle ekle
    m.add_function(wrap_pyfunction!(mv2d_from_array, m)?)?;
    m.add_function(wrap_pyfunction!(mv2d_to_array, m)?)?;
    m.add_function(wrap_pyfunction!(batch_rotate_2d, m)?)?;
    m.add_function(wrap_pyfunction!(mv3d_from_array, m)?)?;
    m.add_function(wrap_pyfunction!(mv3d_to_array, m)?)?;
    m.add_function(wrap_pyfunction!(batch_rotate_3d, m)?)?;

    Ok(())
}