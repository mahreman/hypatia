use hypatia_core::MultiVector2D;
use std::f64::consts::FRAC_PI_2;

fn main() {
    // Hangi versiyonu istediğimizi belirtelim: f64
    let r = MultiVector2D::<f64>::rotor(FRAC_PI_2);
    let v = MultiVector2D::<f64>::vector(1.0, 0.0);

    let w = r.rotate_vector(&v).grade(1);
    println!("2D Rotation: (1,0) -> ({:.6},{:.6})", w.e1, w.e2);
}
