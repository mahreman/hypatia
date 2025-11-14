use hypatia_core::MultiVector3D;
use std::f64::consts::FRAC_PI_2;

fn main() {
    // Hangi versiyonu istediğimizi belirtelim: f64
    let r = MultiVector3D::<f64>::rotor(FRAC_PI_2, 0.0, 0.0, 1.0);
    let x = MultiVector3D::<f64>::vector(1.0, 0.0, 0.0);

    let y = r.rotate_vector(&x).grade(1);
    println!(
        "(1,0,0) --90° around Z--> ({:.6},{:.6},{:.6})",
        y.e1, y.e2, y.e3
    );
}
