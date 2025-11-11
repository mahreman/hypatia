use numpy::{PyArray2, PyReadonlyArray2};
use pyo3::{prelude::*};
// (GÜNCELLENDİ) PyValueError yerine kendi HypatiaError'umuzu import edelim
use crate::python_bindings::{HypatiaError, PyMultiVector2D};

/// (1x2) numpy -> PyMultiVector2D
#[pyfunction]
pub fn mv2d_from_array(arr: PyReadonlyArray2<f64>) -> PyResult<PyMultiVector2D> {
// ... (içerik değişmedi) ...
    let a = arr.as_array();
    if a.shape() != [1, 2] {
        // (GÜNCELLENDİ) PyValueError -> HypatiaError
        return Err(HypatiaError::new_err(
            "mv2d_from_array: shape (1,2) bekleniyordu",
        ));
    }
    Ok(PyMultiVector2D::vector(a[[0, 0]], a[[0, 1]]))
}

/// PyMultiVector2D -> (1x2) numpy
#[pyfunction]
pub fn mv2d_to_array<'py>(
// ... (içerik değişmedi) ...
    py: Python<'py>,
    mv: &PyMultiVector2D,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let out_rows = vec![vec![mv.inner.e1, mv.inner.e2]];
    PyArray2::from_vec2_bound(py, &out_rows)
        // (GÜNCELLENDİ) PyValueError -> HypatiaError
        .map_err(|e| HypatiaError::new_err(format!("mv2d_to_array: NumPy dizisi oluşturulamadı: {}", e)))
}

/// (N x 2) numpy vektörleri -> theta radian kadar 2D dönmüş (N x 2)
#[pyfunction]
pub fn batch_rotate_2d<'py>(
// ... (içerik değişmedi) ...
    py: Python<'py>,
    theta: f64,
    arr: PyReadonlyArray2<f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let a = arr.as_array();
    if a.shape().len() != 2 || a.shape()[1] != 2 {
        // (GÜNCELLENDİ) PyValueError -> HypatiaError
        return Err(HypatiaError::new_err(
            "batch_rotate_2d: input shape (N,2) bekleniyordu",
        ));
    }

    let r = PyMultiVector2D::rotor(theta);
    let mut out = Vec::with_capacity(a.shape()[0]);
    for i in 0..a.shape()[0] {
        let v = PyMultiVector2D::vector(a[[i, 0]], a[[i, 1]]);
        let w = r.rotate_vector(&v).grade(1);
        out.push(vec![w.e1(), w.e2()]);
    }

    PyArray2::from_vec2_bound(py, &out)
        // (GÜNCELLENDİ) PyValueError -> HypatiaError
        .map_err(|e| HypatiaError::new_err(format!("batch_rotate_2d: NumPy dizisi oluşturulamadı: {}", e)))
}

/// (N x 3) numpy vektörleri -> theta radian kadar Z ekseni etrafında dönmüş (N x 3)
/// Not: Şimdilik _ax/_ay/_az parametreleri kullanılmıyor (API stabil tutmak için korunuyor).
#[pyfunction]
pub fn batch_rotate_3d<'py>(
// ... (içerik değişmedi) ...
    py: Python<'py>,
    theta: f64,
    _ax: f64,
    _ay: f64,
    _az: f64,
    arr: PyReadonlyArray2<f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let a = arr.as_array();
    if a.shape().len() != 2 || a.shape()[1] != 3 {
        // (GÜNCELLENDİ) PyValueError -> HypatiaError
        return Err(HypatiaError::new_err(
            "batch_rotate_3d: input shape (N,3) bekleniyordu",
        ));
    }

    // Z ekseni etrafında: (x,y) 2D döner, z sabit kalır
    let rot2d = PyMultiVector2D::rotor(theta);
    let mut out = Vec::with_capacity(a.shape()[0]);
    for i in 0..a.shape()[0] {
        let v = PyMultiVector2D::vector(a[[i, 0]], a[[i, 1]]);
        let w = rot2d.rotate_vector(&v).grade(1);
        out.push(vec![w.e1(), w.e2(), a[[i, 2]]]);
    }

    PyArray2::from_vec2_bound(py, &out)
        // (GÜNCELLENDİ) PyValueError -> HypatiaError
        .map_err(|e| HypatiaError::new_err(format!("batch_rotate_3d: NumPy dizisi oluşturulamadı: {}", e)))
}


// (EKLENDİ) NumPy entegrasyon testleri
#[cfg(test)]
mod tests {
    use super::{batch_rotate_2d, batch_rotate_3d};
    use numpy::PyArray2;
    use pyo3::Python;

    #[test]
    fn test_batch_rotate_2d_identity() {
        Python::with_gil(|py| {
            let data = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
            let arr = PyArray2::from_vec2_bound(py, &data).unwrap();
            
            // theta = 0 (identity rotasyon)
            let result_arr = batch_rotate_2d(py, 0.0, arr.readonly()).unwrap();
            let result_data = result_arr.to_vec2().unwrap();

            assert_eq!(data, result_data);
        });
    }

    #[test]
    fn test_batch_rotate_3d_identity() {
        Python::with_gil(|py| {
            let data = vec![vec![1.0, 2.0, 5.0], vec![3.0, 4.0, 6.0]];
            let arr = PyArray2::from_vec2_bound(py, &data).unwrap();
            
            // theta = 0 (identity rotasyon)
            let result_arr = batch_rotate_3d(py, 0.0, 0.0, 0.0, 1.0, arr.readonly()).unwrap();
            let result_data = result_arr.to_vec2().unwrap();

            assert_eq!(data, result_data);
        });
    }
}