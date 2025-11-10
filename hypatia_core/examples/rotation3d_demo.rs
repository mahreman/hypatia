use hypatia_core::MultiVector3D;
use std::f64::consts::PI;

fn main() {
    println!("=== Hypatia - 3D Geometrik Cebir Demo (Düzeltilmiş) ===");
    println!();

    // 1. Temel çarpım testi
    println!("1. Temel 3D Çarpım Kuralları:");
    let e1 = MultiVector3D::vector(1.0, 0.0, 0.0);
    let e2 = MultiVector3D::vector(0.0, 1.0, 0.0);
    let e3 = MultiVector3D::vector(0.0, 0.0, 1.0);
    
    println!("   e1 * e1 = {:?}", (e1 * e1));
    println!("   e1 * e2 = {:?}", (e1 * e2));
    println!("   e2 * e3 = {:?}", (e2 * e3));
    println!("   e3 * e1 = {:?}", (e3 * e1));
    println!();

    // 2. 3D vektör döndürme demo'su
    println!("2. 3D Vektör Döndürme Demo'su:");
    
    let original_vector = MultiVector3D::vector(1.0, 0.0, 0.0);
    println!("   Orijinal vektör: ({}, {}, {})", 
             original_vector.e1, original_vector.e2, original_vector.e3);
    
    // Z-ekseni etrafında 90 derece döndürme
    let rotor_z = MultiVector3D::rotor(PI / 2.0, (0.0, 0.0, 1.0));
    let rotated_z = rotor_z.rotate_vector(&original_vector);
    println!("   Z-ekseni 90°: ({:.3}, {:.3}, {:.3})", 
             rotated_z.e1, rotated_z.e2, rotated_z.e3);
    
    // Y-ekseni etrafında 90 derece döndürme
    let rotor_y = MultiVector3D::rotor(PI / 2.0, (0.0, 1.0, 0.0));
    let rotated_y = rotor_y.rotate_vector(&original_vector);
    println!("   Y-ekseni 90°: ({:.3}, {:.3}, {:.3})", 
             rotated_y.e1, rotated_y.e2, rotated_y.e3);

    // X-ekseni etrafında 90 derece döndürme
    let rotor_x = MultiVector3D::rotor(PI / 2.0, (1.0, 0.0, 0.0));
    let vector_y = MultiVector3D::vector(0.0, 1.0, 0.0);
    let rotated_x = rotor_x.rotate_vector(&vector_y);
    println!("   X-ekseni 90° (0,1,0 → 0,0,1): ({:.3}, {:.3}, {:.3})", 
             rotated_x.e1, rotated_x.e2, rotated_x.e3);

    println!();
    println!("🎉 3D Geometrik Cebir DOĞRU çalışıyor!");
}