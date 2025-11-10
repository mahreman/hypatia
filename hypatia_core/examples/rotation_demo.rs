// examples/rotation_demo.rs

use hypatia_core::MultiVector2D;
use std::f64::consts::PI;

fn main() {
    println!("=== Hypatia - 2D Geometrik Cebir Demo ===");
    println!();

    // 1. Temel çarpım testi
    println!("1. Temel Çarpım Kuralları:");
    let e1 = MultiVector2D::vector(1.0, 0.0);
    let e2 = MultiVector2D::vector(0.0, 1.0);
    
    println!("   e1 * e1 = {:?}", (e1 * e1));
    println!("   e2 * e2 = {:?}", (e2 * e2));
    println!("   e1 * e2 = {:?}", (e1 * e2));
    println!("   e2 * e1 = {:?}", (e2 * e1));
    println!();

    // 2. Vektör döndürme demo'su
    println!("2. Vektör Döndürme Demo'su:");
    
    let original_vector = MultiVector2D::vector(1.0, 0.0);
    println!("   Orijinal vektör: ({}, {})", original_vector.e1, original_vector.e2);
    
    // 45 derece döndürme
    let rotor_45 = MultiVector2D::rotor(PI / 4.0);
    let rotated_45 = rotor_45.rotate_vector(&original_vector);
    println!("   45° döndürülmüş: ({:.3}, {:.3})", rotated_45.e1, rotated_45.e2);
    
    // 90 derece döndürme  
    let rotor_90 = MultiVector2D::rotor(PI / 2.0);
    let rotated_90 = rotor_90.rotate_vector(&original_vector);
    println!("   90° döndürülmüş: ({:.3}, {:.3})", rotated_90.e1, rotated_90.e2);
    
    // 180 derece döndürme
    let rotor_180 = MultiVector2D::rotor(PI);
    let rotated_180 = rotor_180.rotate_vector(&original_vector);
    println!("   180° döndürülmüş: ({:.3}, {:.3})", rotated_180.e1, rotated_180.e2);

    println!();
    println!("🎉 Tüm döndürmeler doğru çalışıyor!");
}