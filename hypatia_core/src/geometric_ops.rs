//! Geometric Algebra Neural Network Operations
//!
//! Bridges Hypatia's Geometric Algebra primitives (MultiVector2D/3D)
//! with the neural network computation framework.
//!
//! Key operations:
//! - GA-Linear: Linear layer operating on multivector coefficients
//! - GA-Rotate: Learnable rotation via rotor parameters
//! - GA-Norm: Multivector magnitude normalization
//! - Geometric Product Layer: Full geometric product as differentiable op
//!
//! These enable "geometric neural networks" that respect the algebraic
//! structure of the data (rotations, reflections, projections).

use crate::multivector2d::MultiVector2D;
use crate::multivector3d::MultiVector3D;
use crate::native_ops;

/// Apply a 2D rotor (rotation) to a batch of 2D vectors.
///
/// input: [batch, 2] f32 (e1, e2 components)
/// theta: rotation angle in radians
/// output: [batch, 2] f32 (rotated vectors)
pub fn batch_rotor_2d(input: &[f32], batch: usize, theta: f32) -> Vec<f32> {
    debug_assert_eq!(input.len(), batch * 2);

    let (sin_t, cos_t) = (theta / 2.0).sin_cos();
    // Rotor R = cos(θ/2) - sin(θ/2) * e12
    // Rotation: R * v * R†
    // For 2D vector (e1, e2):
    //   rotated_e1 = cos(θ)*e1 - sin(θ)*e2
    //   rotated_e2 = sin(θ)*e1 + cos(θ)*e2
    let cos_full = cos_t * cos_t - sin_t * sin_t; // cos(θ)
    let sin_full = 2.0 * sin_t * cos_t; // sin(θ)

    let mut output = vec![0.0f32; batch * 2];
    for b in 0..batch {
        let e1 = input[b * 2];
        let e2 = input[b * 2 + 1];
        output[b * 2] = cos_full * e1 - sin_full * e2;
        output[b * 2 + 1] = sin_full * e1 + cos_full * e2;
    }
    output
}

/// Apply a 3D rotor (rotation) to a batch of 3D vectors.
///
/// input: [batch, 3] f32 (e1, e2, e3 components)
/// axis: [3] normalized rotation axis (ax, ay, az)
/// theta: rotation angle in radians
/// output: [batch, 3] f32 (rotated vectors)
pub fn batch_rotor_3d(input: &[f32], batch: usize, axis: &[f32; 3], theta: f32) -> Vec<f32> {
    debug_assert_eq!(input.len(), batch * 3);

    let (sin_half, cos_half) = (theta / 2.0).sin_cos();
    // Rotor R = cos(θ/2) + sin(θ/2)(ax*e23 + ay*e31 + az*e12)
    // Rodrigues' rotation formula equivalent:
    // v' = v*cos(θ) + (axis × v)*sin(θ) + axis*(axis·v)*(1 - cos(θ))
    let cos_t = cos_half * cos_half - sin_half * sin_half;
    let sin_t = 2.0 * sin_half * cos_half;
    let ax = axis[0];
    let ay = axis[1];
    let az = axis[2];

    let mut output = vec![0.0f32; batch * 3];
    for b in 0..batch {
        let vx = input[b * 3];
        let vy = input[b * 3 + 1];
        let vz = input[b * 3 + 2];

        // axis × v
        let cx = ay * vz - az * vy;
        let cy = az * vx - ax * vz;
        let cz = ax * vy - ay * vx;

        // axis · v
        let dot = ax * vx + ay * vy + az * vz;

        output[b * 3] = vx * cos_t + cx * sin_t + ax * dot * (1.0 - cos_t);
        output[b * 3 + 1] = vy * cos_t + cy * sin_t + ay * dot * (1.0 - cos_t);
        output[b * 3 + 2] = vz * cos_t + cz * sin_t + az * dot * (1.0 - cos_t);
    }
    output
}

/// Geometric product layer for 2D multivectors.
///
/// Computes the geometric product of input multivectors with learned weight multivectors.
/// input: [batch, 4] (s, e1, e2, e12 coefficients per multivector)
/// weights: [out_features, 4] (learned multivector weights)
/// output: [batch, out_features * 4] (geometric products)
///
/// Each output multivector is the geometric product: weight_j * input_i
pub fn ga2d_geometric_product_layer(
    input: &[f32],
    weights: &[f32],
    batch: usize,
    out_features: usize,
) -> Vec<f32> {
    let in_components = 4; // 2D GA: s, e1, e2, e12
    debug_assert_eq!(input.len(), batch * in_components);
    debug_assert_eq!(weights.len(), out_features * in_components);

    let out_size = out_features * in_components;
    let mut output = vec![0.0f32; batch * out_size];

    for b in 0..batch {
        let a_s = input[b * 4];
        let a_e1 = input[b * 4 + 1];
        let a_e2 = input[b * 4 + 2];
        let a_e12 = input[b * 4 + 3];

        for j in 0..out_features {
            let w_s = weights[j * 4];
            let w_e1 = weights[j * 4 + 1];
            let w_e2 = weights[j * 4 + 2];
            let w_e12 = weights[j * 4 + 3];

            // Geometric product in Cl(2,0):
            // (a_s + a_e1*e1 + a_e2*e2 + a_e12*e12) * (w_s + w_e1*e1 + w_e2*e2 + w_e12*e12)
            // Using e1*e1=1, e2*e2=1, e1*e2=e12, e2*e1=-e12
            let out_s = w_s * a_s + w_e1 * a_e1 + w_e2 * a_e2 - w_e12 * a_e12;
            let out_e1 = w_s * a_e1 + w_e1 * a_s - w_e2 * a_e12 + w_e12 * a_e2;
            let out_e2 = w_s * a_e2 + w_e1 * a_e12 + w_e2 * a_s - w_e12 * a_e1;
            let out_e12 = w_s * a_e12 + w_e1 * a_e2 - w_e2 * a_e1 + w_e12 * a_s;

            let offset = b * out_size + j * 4;
            output[offset] = out_s;
            output[offset + 1] = out_e1;
            output[offset + 2] = out_e2;
            output[offset + 3] = out_e12;
        }
    }

    output
}

/// Multivector norm for 2D GA.
/// Uses reverse norm: ||x||^2 = <x * ~x>_0 = s^2 + e1^2 + e2^2 + e12^2
///
/// input: [batch, 4] (s, e1, e2, e12)
/// output: [batch, 4] normalized multivectors (each divided by its norm)
pub fn ga2d_normalize(input: &[f32], batch: usize) -> Vec<f32> {
    let mut output = vec![0.0f32; batch * 4];

    for b in 0..batch {
        let s = input[b * 4];
        let e1 = input[b * 4 + 1];
        let e2 = input[b * 4 + 2];
        let e12 = input[b * 4 + 3];

        // Reverse norm: all components squared (Euclidean norm of coefficients)
        let norm_sq = s * s + e1 * e1 + e2 * e2 + e12 * e12;
        let norm = if norm_sq > 1e-10 { norm_sq.sqrt() } else { 1.0 };
        let inv_norm = 1.0 / norm;

        output[b * 4] = s * inv_norm;
        output[b * 4 + 1] = e1 * inv_norm;
        output[b * 4 + 2] = e2 * inv_norm;
        output[b * 4 + 3] = e12 * inv_norm;
    }

    output
}

/// Geometric Product Layer for 3D multivectors.
///
/// input: [batch, 8] (s, e1, e2, e3, e12, e23, e31, e123)
/// weights: [out_features, 8]
/// output: [batch, out_features * 8]
pub fn ga3d_geometric_product_layer(
    input: &[f32],
    weights: &[f32],
    batch: usize,
    out_features: usize,
) -> Vec<f32> {
    let in_components = 8; // 3D GA: s, e1, e2, e3, e12, e23, e31, e123
    debug_assert_eq!(input.len(), batch * in_components);
    debug_assert_eq!(weights.len(), out_features * in_components);

    let out_size = out_features * in_components;
    let mut output = vec![0.0f32; batch * out_size];

    for b in 0..batch {
        let a = &input[b * 8..(b + 1) * 8];

        for j in 0..out_features {
            let w = &weights[j * 8..(j + 1) * 8];

            // Full geometric product in Cl(3,0):
            // Cayley table for {1, e1, e2, e3, e12, e23, e31, e123}
            let offset = b * out_size + j * 8;

            // scalar: 1*1 + e1*e1 + e2*e2 + e3*e3 - e12*e12 - e23*e23 - e31*e31 - e123*e123
            output[offset] = w[0]*a[0] + w[1]*a[1] + w[2]*a[2] + w[3]*a[3]
                           - w[4]*a[4] - w[5]*a[5] - w[6]*a[6] - w[7]*a[7];

            // e1: 1*e1 + e1*1 + e2*e12 - e3*e31 - e12*e2 - e23*e123 + e31*e3 + e123*e23
            output[offset + 1] = w[0]*a[1] + w[1]*a[0] - w[2]*a[4] + w[3]*a[6]
                                + w[4]*a[2] + w[5]*a[7] - w[6]*a[3] - w[7]*a[5];

            // e2: 1*e2 - e1*e12 + e2*1 + e3*e23 + e12*e1 - e23*e3 - e31*e123 + e123*e31
            output[offset + 2] = w[0]*a[2] + w[1]*a[4] + w[2]*a[0] - w[3]*a[5]
                                - w[4]*a[1] + w[5]*a[3] + w[6]*a[7] - w[7]*a[6];

            // e3: 1*e3 + e1*e31 - e2*e23 + e3*1 + e12*e123 + e23*e2 - e31*e1 - e123*e12
            output[offset + 3] = w[0]*a[3] - w[1]*a[6] + w[2]*a[5] + w[3]*a[0]
                                - w[4]*a[7] - w[5]*a[2] + w[6]*a[1] + w[7]*a[4];

            // e12: 1*e12 + e1*e2 - e2*e1 + e3*e123 + e12*1 - e23*e31 + e31*e23 - e123*e3
            output[offset + 4] = w[0]*a[4] + w[1]*a[2] - w[2]*a[1] + w[3]*a[7]
                                + w[4]*a[0] + w[5]*a[6] - w[6]*a[5] - w[7]*a[3];

            // e23: 1*e23 - e1*e123 + e2*e3 - e3*e2 + e12*e31 + e23*1 - e31*e12 + e123*e1
            output[offset + 5] = w[0]*a[5] - w[1]*a[7] + w[2]*a[3] - w[3]*a[2]
                                - w[4]*a[6] + w[5]*a[0] + w[6]*a[4] + w[7]*a[1];

            // e31: 1*e31 + e1*e3 + e2*e123 - e3*e1 - e12*e23 + e23*e12 + e31*1 - e123*e2
            output[offset + 6] = w[0]*a[6] + w[1]*a[3] + w[2]*a[7] - w[3]*a[1]
                                + w[4]*a[5] - w[5]*a[4] + w[6]*a[0] - w[7]*a[2];

            // e123: 1*e123 + e1*e23 + e2*e31 + e3*e12 + e12*e3 + e23*e1 + e31*e2 + e123*1
            output[offset + 7] = w[0]*a[7] + w[1]*a[5] + w[2]*a[6] + w[3]*a[4]
                                + w[4]*a[3] + w[5]*a[1] + w[6]*a[2] + w[7]*a[0];
        }
    }

    output
}

/// Multivector norm for 3D GA.
///
/// Uses the reverse norm: ||x||^2 = <x * ~x>_0
/// In Cl(3,0), the reverse flips signs of grade-2 and grade-3 elements.
/// The scalar part of x * ~x gives:
///   s^2 + e1^2 + e2^2 + e3^2 + e12^2 + e23^2 + e31^2 + e123^2
/// (all positive, because grade-k reverse sign (-1)^(k(k-1)/2) cancels with e_I^2 sign)
pub fn ga3d_normalize(input: &[f32], batch: usize) -> Vec<f32> {
    let mut output = vec![0.0f32; batch * 8];

    for b in 0..batch {
        let mv = &input[b * 8..(b + 1) * 8];

        // Reverse norm: all components squared (Euclidean norm of coefficients)
        let norm_sq = mv[0]*mv[0] + mv[1]*mv[1] + mv[2]*mv[2] + mv[3]*mv[3]
                    + mv[4]*mv[4] + mv[5]*mv[5] + mv[6]*mv[6] + mv[7]*mv[7];
        let norm = if norm_sq > 1e-10 { norm_sq.sqrt() } else { 1.0 };
        let inv_norm = 1.0 / norm;

        for i in 0..8 {
            output[b * 8 + i] = mv[i] * inv_norm;
        }
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_rotor_2d() {
        use std::f32::consts::PI;
        // Rotate (1, 0) by 90 degrees -> (0, 1)
        let input = vec![1.0f32, 0.0];
        let output = batch_rotor_2d(&input, 1, PI / 2.0);
        assert!((output[0] - 0.0).abs() < 1e-5, "e1={}", output[0]);
        assert!((output[1] - 1.0).abs() < 1e-5, "e2={}", output[1]);
    }

    #[test]
    fn test_batch_rotor_3d() {
        use std::f32::consts::PI;
        // Rotate (1,0,0) by 90 degrees around z-axis -> (0,1,0)
        let input = vec![1.0f32, 0.0, 0.0];
        let axis = [0.0, 0.0, 1.0];
        let output = batch_rotor_3d(&input, 1, &axis, PI / 2.0);
        assert!((output[0] - 0.0).abs() < 1e-5, "x={}", output[0]);
        assert!((output[1] - 1.0).abs() < 1e-5, "y={}", output[1]);
        assert!((output[2] - 0.0).abs() < 1e-5, "z={}", output[2]);
    }

    #[test]
    fn test_ga2d_geometric_product_identity() {
        // Geometric product with scalar 1 should be identity
        let input = vec![0.0f32, 1.0, 2.0, 0.5]; // s=0, e1=1, e2=2, e12=0.5
        let weight = vec![1.0f32, 0.0, 0.0, 0.0]; // scalar 1
        let output = ga2d_geometric_product_layer(&input, &weight, 1, 1);
        assert!((output[0] - 0.0).abs() < 1e-5);
        assert!((output[1] - 1.0).abs() < 1e-5);
        assert!((output[2] - 2.0).abs() < 1e-5);
        assert!((output[3] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_ga2d_normalize() {
        let input = vec![3.0f32, 4.0, 0.0, 0.0]; // norm = 5
        let output = ga2d_normalize(&input, 1);
        assert!((output[0] - 0.6).abs() < 1e-5);
        assert!((output[1] - 0.8).abs() < 1e-5);
    }
}
