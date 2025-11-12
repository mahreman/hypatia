use egg::{
    define_language,
    rewrite, CostFunction, Extractor, Id, RecExpr, Rewrite, Runner, Symbol,
    Language, // (DÜZELTME) 'children' metodunu kapsama almak için eklendi
};
use ordered_float::NotNan;  // f64 Eq/Ord/Hash için wrapper

// Adım 1: Dilimizi (Symbol AST) egg için tanımla
define_language! {
    pub enum HypatiaLang {
        "add" = Add([Id; 2]),
        "mul" = Mul([Id; 2]),
        "neg" = Neg(Id),
        // FIX: NotNan<f64> ile Eq/Ord/Hash sağlanır
        Constant(NotNan<f64>),
        // Değişkenler (x, p_a, vb.)
        Var(Symbol),
    }
}

// Adım 2: FLOPs Maliyet Modeli
pub struct FLOPsCost;
impl CostFunction<HypatiaLang> for FLOPsCost {
    type Cost = usize;
    fn cost<C>(&mut self, enode: &HypatiaLang, mut costs: C) -> Self::Cost
    where
        C: FnMut(Id) -> Self::Cost,
    {
        // FIX: usize annotation ile sum() ambiguity gider
        // 'children' metodu artık 'Language' trait'i import edildiği için bulunacak
        let children_cost: usize = enode.children().iter().map(|&id| costs(id)).sum();
       
        let op_cost = match enode {
            // Matmul/Mul pahalıdır (örn. 10 FLOPs)
            HypatiaLang::Mul(_) => 10,
            // Add/Neg ucuzdur (örn. 1 FLOP)
            HypatiaLang::Add(_) => 1,
            HypatiaLang::Neg(_) => 1,
            // Sabitler ve Değişkenler maliyetsizdir (0 FLOPs)
            HypatiaLang::Constant(_) => 0,
            HypatiaLang::Var(_) => 0,
        };
       
        op_cost + children_cost
    }
}

// Adım 3: Optimizasyon Kuralları (Manifesto v3.0)
fn get_rules() -> Vec<Rewrite<HypatiaLang, ()>> {
    vec![
        // Dağılma Kuralı (Faktoring): A*B + A*C => A*(B+C)
        rewrite!("factor";
            "(add (mul ?a ?b) (mul ?a ?c))" =>
            "(mul ?a (add ?b ?c))"
        ),
       
        // Dağılma Kuralı (Tersi): A*(B+C) => A*B + A*C
        rewrite!("distribute";
            "(mul ?a (add ?b ?c))" =>
            "(add (mul ?a ?b) (mul ?a ?c))"
        ),
       
        // Diğer temel cebir kuralları (simplify.rs'dekine benzer)
        rewrite!("commute-add"; "(add ?a ?b)" => "(add ?b ?a)"),
        rewrite!("commute-mul"; "(mul ?a ?b)" => "(mul ?b ?a)"),
       
        rewrite!("add-0"; "(add ?a 0)" => "?a"),
        rewrite!("mul-1"; "(mul ?a 1)" => "?a"),
        rewrite!("mul-0"; "(mul ?a 0)" => "0"),
    ]
}

/// Python'dan çağrılacak ana optimizasyon fonksiyonu
pub fn optimize_ast(expr_str: &str) -> String {
    // 1. Gelen S-expression string'ini parse et
    let start_expr: RecExpr<HypatiaLang> = match expr_str.parse() {
        Ok(expr) => expr,
        Err(e) => return format!("(error: 'Parse Error: {}')", e),
    };
    // 2. Kuralları ve e-graph runner'ı hazırla
    let rules = get_rules();
    let runner = Runner::default()
        .with_expr(&start_expr)
        .run(&rules);
    // 3. En düşük maliyetli ifadeyi FLOPsCost'a göre çıkar
    let cost_function = FLOPsCost;
    let extractor = Extractor::new(&runner.egraph, cost_function);
   
    let (_best_cost, best_expr) = extractor.find_best(runner.roots[0]);
    // 4. Optimize edilmiş ifadeyi S-expression string olarak geri dön
    best_expr.to_string()
}