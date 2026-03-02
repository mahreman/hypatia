fn main() {
    // Link against system OpenBLAS for cblas_sgemm
    println!("cargo:rustc-link-lib=openblas");
    println!("cargo:rustc-link-search=/usr/lib/x86_64-linux-gnu");
}
