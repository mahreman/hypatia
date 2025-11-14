use egg::{
    define_language, rewrite, CostFunction, Extractor, Id, Language, RecExpr, Rewrite, Runner,
    Symbol as EggSymbol, Analysis, DidMerge, EGraph,
};
use ordered_float::NotNan;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::time::Duration;
use crate::symbolic::Symbol; 
use crate::python_bindings::ModuleInfo; // ModuleInfo'yu kullanmak için import
// `HashMap` kullanılmıyor, uyarıyı önlemek için kaldırıldı

// ============================================================================
// HYPATIA LANGUAGE DEFINITION
// ============================================================================

define_language! {
    pub enum HypatiaLang {
        // --- Temel Aritmetik ---
        "add" = Add([Id; 2]), 
        "mul" = Mul([Id; 2]), 
        "sub" = Sub([Id; 2]),
        "div" = Div([Id; 2]), 
        "neg" = Neg(Id), 
        "exp" = Exp(Id), 
        "log" = Log(Id),
        "sqrt" = Sqrt(Id), 
        "pow" = Pow([Id; 2]),
        
        // --- Temel Aktivasyonlar ---
        "relu" = ReLU(Id), 
        "relu_grad" = ReLUGrad(Id), 
        "sigmoid" = Sigmoid(Id), 
        "tanh" = Tanh(Id), 
        "softmax" = Softmax(Id),
        
        // --- İstatistiksel ---
        "mean" = Mean(Id), 
        "var" = Variance(Id),
        "max" = Max([Id; 2]), 
        "min" = Min([Id; 2]),

        // --- Şekil (Shape) Operatörleri ---
        "flatten" = Flatten(Id), // (flatten x)

        // --- PHASE 3: Gelişmiş AI Operatörleri ---
        "matmul" = MatMul([Id; 2]), // (matmul a b)
        "linear" = Linear([Id; 3]), // (linear w b x)
        
        // --- ResNet Operatörleri ---
        // (conv2d w b x stride padding dilation groups)
        "conv2d" = Conv2d([Id; 7]), 
        // (batchnorm w b mean var x eps)
        "batchnorm" = BatchNorm([Id; 6]), 
        // (maxpool2d x kernel_size stride padding)
        "maxpool2d" = MaxPool2d([Id; 4]),
        "avgpool2d" = AvgPool2d([Id; 4]),
        "adaptive_avg_pool2d" = AdaptiveAvgPool2d(Id),
        
        "attention" = Attention([Id; 3]), 
        "embedding" = Embedding([Id; 2]), 
        "transformer_encoder" = TransformerEncoder(Id),
        
        // --- Fusion Hedefleri ---
        "linear-relu" = LinearReLU([Id; 3]), // (linear-relu w b x)
        "fused-mlp" = FusedMLP([Id; 5]), // (fused-mlp w1 b1 w2 b2 x)
        "fused_conv_bn" = FusedConvBN([Id; 12]),

        // --- Temel ---
        Constant(NotNan<f64>), 
        Var(EggSymbol),
    }
}

// ============================================================================
// Constant Folding Analysis
// ============================================================================

#[derive(Default)]
pub struct ConstantFoldingAnalysis;

impl Analysis<HypatiaLang> for ConstantFoldingAnalysis {
    type Data = Option<f64>; 

    fn make( egraph: &EGraph<HypatiaLang, Self>, enode: &HypatiaLang) -> Self::Data { 
        let get_data = |id: Id| egraph[id].data;
        match enode {
            HypatiaLang::Constant(c) => Some(c.into_inner()),
            HypatiaLang::Neg(id) => get_data(*id).map(|c| -c),
            HypatiaLang::Add([a, b]) => get_data(*a).zip(get_data(*b)).map(|(a, b)| a + b),
            HypatiaLang::Sub([a, b]) => get_data(*a).zip(get_data(*b)).map(|(a, b)| a - b),
            HypatiaLang::Mul([a, b]) => get_data(*a).zip(get_data(*b)).map(|(a, b)| a * b),
            HypatiaLang::Div([a, b]) => get_data(*a).zip(get_data(*b)).and_then(|(a, b)| {
                if b != 0.0 { Some(a / b) } else { None }
            }),
            HypatiaLang::Pow([a, b]) => get_data(*a).zip(get_data(*b)).map(|(a, b)| a.powf(b)),
            HypatiaLang::Exp(id) => get_data(*id).map(|c| c.exp()),
            HypatiaLang::Log(id) => get_data(*id).and_then(|c| {
                if c > 0.0 { Some(c.ln()) } else { None }
            }),
            HypatiaLang::Sqrt(id) => get_data(*id).and_then(|c| {
                if c >= 0.0 { Some(c.sqrt()) } else { None }
            }),
            HypatiaLang::ReLU(id) => get_data(*id).map(|c| if c > 0.0 { c } else { 0.0 }),
            HypatiaLang::Sigmoid(id) => get_data(*id).map(|c| 1.0 / (1.0 + (-c).exp())),
            HypatiaLang::Tanh(id) => get_data(*id).map(|c| c.tanh()),
            HypatiaLang::Softmax(id) => get_data(*id).map(|_c| 1.0),
            HypatiaLang::Mean(id) => get_data(*id),
            HypatiaLang::Variance(id) => get_data(*id).map(|_c| 0.0),
            _ => None,
        }
    }

    fn merge(&mut self, to: &mut Self::Data, from: Self::Data) -> DidMerge { 
        egg::merge_option(to, from, |a, b| {
            if (*a - b).abs() < 1e-9 { DidMerge(false, false) } 
            else { DidMerge(false, false) }
        })
    }

    fn modify( egraph: &mut EGraph<HypatiaLang, Self>, id: Id) { 
        if let Some(c) = egraph[id].data {
            if let Ok(c_not_nan) = NotNan::new(c) {
                let const_node = HypatiaLang::Constant(c_not_nan);
                let const_id = egraph.add(const_node);
                egraph.union(id, const_id);
            }
        }
    }
}

// ============================================================================
// HardwareAwareCost (Gelişmiş Maliyet Modeli)
// ============================================================================
pub struct HardwareAwareCost {
    pub is_inference: bool,
}

impl CostFunction<HypatiaLang> for HardwareAwareCost { 
    type Cost = f64; 

    fn cost<C>(&mut self, enode: &HypatiaLang, mut costs: C) -> Self::Cost
    where C: FnMut(Id) -> Self::Cost,
    {
        let children_cost: f64 = enode.children().iter().map(|&id| costs(id)).sum();
        
        // Basitleştirilmiş Maliyet Hesaplama
        let (flops, mem_access, stability_penalty) = match enode {
            HypatiaLang::MatMul(_) => (300.0, 10.0, 0.0), 
            HypatiaLang::Conv2d(_) => (500.0, 50.0, 0.0),
            HypatiaLang::Linear(_) => (100.0, 5.0, 0.0),
            HypatiaLang::BatchNorm(_) => (50.0, 20.0, 0.0), 
            HypatiaLang::Add(_) | HypatiaLang::Sub(_) => (1.0, 0.0, 0.0),
            HypatiaLang::Mul(_) => (10.0, 0.0, 0.0),
            HypatiaLang::Div(_) => (40.0, 0.0, 100.0), 
            
            // Fusion Hedefleri
            HypatiaLang::FusedConvBN(_) => (500.0, 50.0, 0.0), 
            HypatiaLang::LinearReLU(_) => (100.0, 5.0, 0.0),
            HypatiaLang::FusedMLP(_) => (200.0, 10.0, 0.0),

            _ => (0.0, 0.0, 0.0),
        };

        // Cost = FLOPs * 1.0 + Memory_Access * 10.0 + Stability_Penalty * lambda
        let lambda = 1.0; 
        let cost = flops * 1.0 + mem_access * 10.0 + stability_penalty * lambda + children_cost;
        cost
    }
}

// ============================================================================
// Rewrite Kuralları
// ============================================================================
// fn is_inference_mode(_id: Id, is_inference_flag: bool) -> Option<()> {
//     if is_inference_flag { Some(()) } else { None }
// }

fn get_rules(is_inference_mode_flag: bool) -> Vec<Rewrite<HypatiaLang, ConstantFoldingAnalysis>> { 
    let mut rules = vec![
        // --- Aritmetik Kurallar (Her zaman güvenli) ---
        rewrite!("commute-add"; "(add ?a ?b)" => "(add ?b ?a)"),
        rewrite!("commute-mul"; "(mul ?a ?b)" => "(mul ?b ?a)"),
        rewrite!("assoc-add"; "(add (add ?a ?b) ?c)" => "(add ?a (add ?b ?c))"),
        rewrite!("assoc-mul"; "(mul (mul ?a ?b) ?c)" => "(mul ?a (mul ?b ?c))"),
        rewrite!("factor"; "(add (mul ?a ?b) (mul ?a ?c))" => "(mul ?a (add ?b ?c))"),
        rewrite!("add-0";  "(add ?a 0)" => "?a"),
        rewrite!("mul-0";  "(mul ?a 0)" => "0"),
        rewrite!("mul-1";  "(mul ?a 1)" => "?a"),
        rewrite!("sub-0";  "(sub ?a 0)" => "?a"),
        rewrite!("sub-self"; "(sub ?a ?a)" => "0"),
        rewrite!("add-self"; "(add ?a ?a)" => "(mul 2 ?a)"),
        rewrite!("neg-neg"; "(neg (neg ?x))" => "?x"),
        rewrite!("neg-sub"; "(neg (sub ?a ?b))" => "(sub ?b ?a)"),
        rewrite!("neg-mul-distribute"; "(neg (mul ?a ?b))" => "(mul (neg ?a) ?b)"),
        rewrite!("neg-mul-commute"; "(mul (neg ?a) ?b)" => "(mul ?a (neg ?b))"),
        rewrite!("neg-mul-neg"; "(mul (neg ?a) (neg ?b))" => "(mul ?a ?b)"),
        rewrite!("div-self"; "(div ?x ?x)" => "1"),
        rewrite!("div-1";    "(div ?x 1)" => "?x"),
        rewrite!("div-to-mul-pow"; "(div ?a ?b)" => "(mul ?a (pow ?b -1))"),
        rewrite!("mul-pow-to-div"; "(mul ?a (pow ?b -1))" => "(div ?a ?b)"),
        rewrite!("pow-pow"; "(pow (pow ?a ?b) ?c)" => "(pow ?a (mul ?b ?c))"),
        rewrite!("sqrt-to-pow"; "(sqrt ?x)" => "(pow ?x 0.5)"),
        rewrite!("pow-to-sqrt"; "(pow ?x 0.5)" => "(sqrt ?x)"),
        rewrite!("sqrt-pow2"; "(sqrt (pow ?x 2))" => "?x"), 
        rewrite!("sqrt-mul"; "(sqrt (mul ?a ?b))" => "(mul (sqrt ?a) (sqrt ?b))"),
        rewrite!("softmax-stability"; "(softmax (add ?x ?c))" => "(softmax ?x)"),
        rewrite!("relu-idempotent"; "(relu (relu ?x))" => "(relu ?x)"),
    ];
    
    // ✅ DÜZELTME: Tehlikeli optimizasyonlar (fusion)
    // sadece inference (çıkarım) modunda çalışır.
    if is_inference_mode_flag {
        rules.extend(vec![
            rewrite!("linear-relu-fusion";
                "(relu (linear ?w ?b ?x))"
                => 
                "(linear-relu ?w ?b ?x)"),
            
            rewrite!("mlp-fusion-from-fused";
                "(linear ?w2 ?b2 (linear-relu ?w1 ?b1 ?x))"
                =>
                "(fused-mlp ?w1 ?b1 ?w2 ?b2 ?x)"),
            
            // BU KURAL ZATEN BN FOLDING YAPIYOR (CONV İÇİN)
            rewrite!("conv-bn-fusion";
                "(batchnorm ?w_bn ?b_bn ?m ?v (conv2d ?w_c ?b_c ?x ?s ?p ?d ?g) ?eps)"
                =>
                "(fused_conv_bn ?w_c ?b_c ?w_bn ?b_bn ?m ?v ?x ?eps ?s ?p ?d ?g)"),
                
            rewrite!("linear-chain";
                "(linear ?w2 ?b2 (linear ?w1 ?b1 ?x))"
                =>
                "(linear (matmul ?w2 ?w1) (add (matmul ?w2 ?b1) ?b2) ?x)"),
            
            // ✅ DÜZELTME: PANIC ATAN 'batchnorm-fold' KURALI SİLİNDİ.
            // Bu kural, `?w_fold` ve `?b_fold` değişkenlerini tanımlamadığı için
            // geçersizdi ve paniğe neden oluyordu.
        ]);
    }
    
    rules
}

// ============================================================================
// Dış API
// ============================================================================

// ✅ DÜZELTME: Fonksiyonu pub olarak dışa aktar
pub fn rec_to_string(expr: &RecExpr<HypatiaLang>) -> String {
    format!("{}", expr)
}

// ✅ DÜZELTME: E0428 hatasını çözmek için bu fonksiyon `optimize_to_ast` olarak yeniden adlandırıldı.
pub fn optimize_to_ast(expr_str: &str) -> Result<RecExpr<HypatiaLang>, String> { 
    // Varsayılan olarak inference modu açık
    let info = ModuleInfo {
        module_type: "Unknown".to_string(),
        has_bias: false,
        is_inference: true // Manuel olarak ayarlandı
    };
    optimize_to_ast_internal(expr_str, &info)
}

// ✅ GÜNCEL: ModuleInfo'yu parametre olarak alan ana optimizasyon fonksiyonu
pub fn optimize_to_ast_with_info(expr_str: &str, info: &ModuleInfo) -> Result<RecExpr<HypatiaLang>, String> {
    optimize_to_ast_internal(expr_str, info)
}

fn optimize_to_ast_internal(expr_str: &str, info: &ModuleInfo) -> Result<RecExpr<HypatiaLang>, String> { 
    // ✅ DÜZELTME: info.is_inference zaten bool (manuel FromPyObject sayesinde)
    let is_inference_mode_flag = info.is_inference; 
    
    let res = catch_unwind(AssertUnwindSafe(|| {
        let start_expr: RecExpr<HypatiaLang> = match expr_str.parse() {
            Ok(expr) => expr,
            Err(e) => return Err(format!("(error \"Parse Error: {}\")", e)),
        };
        
        let rules = get_rules(is_inference_mode_flag);
        
        // Gelişmiş maliyet modelini kullan
        let cost_function = HardwareAwareCost { 
            is_inference: is_inference_mode_flag 
        };
        
        let runner = Runner::default()
            .with_egraph(egg::EGraph::new(ConstantFoldingAnalysis))
            .with_node_limit(20_000).with_iter_limit(30)
            .with_time_limit(Duration::from_millis(150))
            .with_expr(&start_expr).run(&rules);
            
        let extractor = Extractor::new(&runner.egraph, cost_function);
        let (_best_cost, best_expr) = extractor.find_best(runner.roots[0]);
        Ok(best_expr)
    }));
    match res {
        Ok(Ok(expr)) => Ok(expr),
        Ok(Err(e)) => Err(e),
        Err(_) => Err("(error \"optimizer panic caught\")".to_string()),
    }
}

// ============================================================================
// E-graph → Symbol Çeviricileri (Değişiklik yok)
// ============================================================================

/// `HypatiaLang` RecExpr'ini özyinelemeli olarak `Symbol`'e çevirir.
fn build_symbol(node_id: Id, rec_expr: &RecExpr<HypatiaLang>) -> Symbol { 
    let node = &rec_expr[node_id];
    match node {
        HypatiaLang::Constant(c) => Symbol::Const(c.into_inner()),
        HypatiaLang::Var(v) => Symbol::Variable(v.to_string()),
        HypatiaLang::Neg(id) => Symbol::Neg(Box::new(build_symbol(*id, rec_expr))),
        HypatiaLang::Exp(id) => Symbol::Exp(Box::new(build_symbol(*id, rec_expr))),
        HypatiaLang::Log(id) => Symbol::Log(Box::new(build_symbol(*id, rec_expr))),
        HypatiaLang::Sqrt(id) => Symbol::Sqrt(Box::new(build_symbol(*id, rec_expr))),
        HypatiaLang::ReLU(id) => Symbol::ReLU(Box::new(build_symbol(*id, rec_expr))),
        HypatiaLang::ReLUGrad(id) => Symbol::ReLUGrad(Box::new(build_symbol(*id, rec_expr))),
        HypatiaLang::Sigmoid(id) => Symbol::Sigmoid(Box::new(build_symbol(*id, rec_expr))),
        HypatiaLang::Tanh(id) => Symbol::Tanh(Box::new(build_symbol(*id, rec_expr))),
        HypatiaLang::Softmax(id) => Symbol::Softmax(Box::new(build_symbol(*id, rec_expr))),
        HypatiaLang::Mean(id) => Symbol::Mean(Box::new(build_symbol(*id, rec_expr))),
        HypatiaLang::Variance(id) => Symbol::Variance(Box::new(build_symbol(*id, rec_expr))),
        HypatiaLang::Add([a, b]) => Symbol::Add( Box::new(build_symbol(*a, rec_expr)), Box::new(build_symbol(*b, rec_expr)), ),
        HypatiaLang::Sub([a, b]) => Symbol::Sub( Box::new(build_symbol(*a, rec_expr)), Box::new(build_symbol(*b, rec_expr)), ),
        HypatiaLang::Mul([a, b]) => Symbol::Mul( Box::new(build_symbol(*a, rec_expr)), Box::new(build_symbol(*b, rec_expr)), ),
        HypatiaLang::Div([a, b]) => Symbol::Div( Box::new(build_symbol(*a, rec_expr)), Box::new(build_symbol(*b, rec_expr)), ),
        HypatiaLang::Pow([a, b]) => Symbol::Pow( Box::new(build_symbol(*a, rec_expr)), Box::new(build_symbol(*b, rec_expr)), ),
        HypatiaLang::Max([a, b]) => Symbol::Max( Box::new(build_symbol(*a, rec_expr)), Box::new(build_symbol(*b, rec_expr)), ),
        HypatiaLang::Min([a, b]) => Symbol::Min( Box::new(build_symbol(*a, rec_expr)), Box::new(build_symbol(*b, rec_expr)), ),
        
        HypatiaLang::Flatten(id) => build_symbol(*id, rec_expr), 
        
        HypatiaLang::Embedding([w, x]) => Symbol::Embedding(Box::new(build_symbol(*w, rec_expr)), Box::new(build_symbol(*x, rec_expr))),
        
        HypatiaLang::TransformerEncoder(id) => build_symbol(*id, rec_expr), 

        HypatiaLang::MatMul([a, _b]) => build_symbol(*a, rec_expr), 
        HypatiaLang::Linear([_w, _b, x]) => build_symbol(*x, rec_expr), 
        
        HypatiaLang::Conv2d([_w, _b, x, ..]) => build_symbol(*x, rec_expr),
        
        HypatiaLang::BatchNorm([w, b, mean, var, x, eps]) => {
            Symbol::BatchNorm {
                weight: Box::new(build_symbol(*w, rec_expr)),
                bias: Box::new(build_symbol(*b, rec_expr)), 
                running_mean: Box::new(build_symbol(*mean, rec_expr)),
                running_var: Box::new(build_symbol(*var, rec_expr)),
                input: Box::new(build_symbol(*x, rec_expr)),
                eps: Box::new(build_symbol(*eps, rec_expr)),
            }
        },
        
        HypatiaLang::MaxPool2d([x, k, s, p]) => {
            Symbol::MaxPool2d {
                input: Box::new(build_symbol(*x, rec_expr)),
                kernel_size: Box::new(build_symbol(*k, rec_expr)),
                stride: Box::new(build_symbol(*s, rec_expr)),
                padding: Box::new(build_symbol(*p, rec_expr)),
            }
        },

        HypatiaLang::AvgPool2d([x, ..]) => build_symbol(*x, rec_expr),
        HypatiaLang::AdaptiveAvgPool2d(id) => build_symbol(*id, rec_expr), 
        
        HypatiaLang::Attention([q, _k, _v]) => build_symbol(*q, rec_expr), 
        HypatiaLang::LinearReLU([_w, _b, x]) => Symbol::ReLU(Box::new(build_symbol(*x, rec_expr))), 
        HypatiaLang::FusedMLP([_w1, _b1, _w2, _b2, x]) => build_symbol(*x, rec_expr), 

        HypatiaLang::FusedConvBN([_w_c, _b_c, _w_bn, _b_bn, _m, _v, x, ..]) => build_symbol(*x, rec_expr),
    }
}

pub fn parse_expr_to_symbol(expr_str: &str) -> Result<Symbol, String> { 
    let rec_expr: RecExpr<HypatiaLang> = expr_str.parse()
        .map_err(|e| format!("Parse Error: {}", e))?;
    let root_id = Id::from(rec_expr.as_ref().len() - 1);
    let symbol = build_symbol(root_id, &rec_expr);
    Ok(symbol)
}
pub fn is_equivalent(expr1_str: &str, expr2_str: &str) -> Result<bool, String> { 
    let expr1: RecExpr<HypatiaLang> = expr1_str.parse()
        .map_err(|e| format!("Parse Error (expr1): {}", e))?;
    let expr2: RecExpr<HypatiaLang> = expr2_str.parse()
        .map_err(|e| format!("Parse Error (expr2): {}", e))?;
    let rules = get_rules(true); 
    let mut egraph = EGraph::new(ConstantFoldingAnalysis);
    let root1 = egraph.add_expr(&expr1);
    let root2 = egraph.add_expr(&expr2);
    let runner = Runner::default()
        .with_egraph(egraph)
        .with_node_limit(20_000)
        .with_iter_limit(30)
        .with_time_limit(Duration::from_millis(500))
        .run(&rules);
    let id1 = runner.egraph.find(root1);
    let id2 = runner.egraph.find(root2);
    Ok(id1 == id2)
}


// ============================================================================
// Tests
// ============================================================================
#[cfg(test)]
mod tests {
    use super::*;

    // optimize_ast (String döndüren) için bir yardımcı
    fn optimize_ast(expr_str: &str) -> String {
        let info = ModuleInfo {
            module_type: "Unknown".to_string(),
            has_bias: false,
            is_inference: true
        };
        match optimize_to_ast_internal(expr_str, &info) {
            Ok(best_expr) => rec_to_string(&best_expr),
            Err(e) => e,
        }
    }
    
    #[test]
    fn t_zero_mul() { assert_eq!(optimize_ast("(mul x 0)"), "0"); }
    #[test]
    fn t_neg_mul() { assert_eq!(optimize_ast("(neg (mul a b))"), "(mul (neg a) b)"); }
    #[test]
    fn t_sigmoid_zero_analysis() { assert_eq!(optimize_ast("(sigmoid 0)"), "0.5"); }
    #[test]
    fn t_constant_folding_add() { assert_eq!(optimize_ast("(add 1 2)"), "3"); }
    #[test]
    fn t_constant_folding_pow() { assert_eq!(optimize_ast("(pow 2 -1)"), "0.5"); }
    #[test]
    fn t_div_to_mul_pow_analysis() {
        assert_eq!(optimize_ast("(div x 2)"), "(mul x 0.5)");
    }
    #[test]
    fn t_parse_expr() { 
        let expr_str = "(mul x (add 2 3))";
        let symbol = parse_expr_to_symbol(expr_str).unwrap();
        let expected = Symbol::Mul(
            Box::new(Symbol::Variable("x".to_string())),
            Box::new(Symbol::Add( Box::new(Symbol::Const(2.0)), Box::new(Symbol::Const(3.0)) ))
        );
        assert_eq!(symbol, expected);
    }
    #[test]
    fn t_parse_and_derivative() { 
        let expr_str = "(mul a (add b c))";
        let symbol = parse_expr_to_symbol(expr_str).unwrap();
        let grad_a = symbol.derivative("a").simplify(); 
        let expected = Symbol::Add(
            Box::new(Symbol::Add( 
                Box::new(Symbol::Variable("b".to_string())), 
                Box::new(Symbol::Variable("c".to_string())) 
            )),
            Box::new(Symbol::Const(0.0))
        );
        assert_eq!(grad_a, expected);
    }
    
    #[test]
    fn t_ai_rule_sqrt_to_pow() {
        assert_eq!(optimize_ast("(sqrt x)"), "(sqrt x)");
        assert_eq!(optimize_ast("(pow x 0.5)"), "(sqrt x)");
    }
    
    #[test]
    fn t_ai_rule_layernorm_fusion() {
        let start = "(div (sub X mean) (sqrt var))";
        let expected = "(mul (sub X mean) (pow var -0.5))";
        assert_eq!(optimize_ast(start), expected);
        assert_eq!(optimize_ast(expected), expected);
    }
    #[test]
    fn t_is_equivalent() {
        assert!(is_equivalent("(add 1 2)", "3").unwrap());
        assert!(is_equivalent(
            "(div (sub X mean) (sqrt var))", 
            "(mul (sub X mean) (pow var -0.5))"
        ).unwrap());
        assert!(is_equivalent(
            "(softmax (mul k_t (mul q (pow d_sqrt -1))))",
            "(softmax (mul q (mul k_t (pow d_sqrt -1))))"
        ).unwrap());
        assert!(is_equivalent(
            "(softmax (div (mul q k_t) d_sqrt))",
            "(softmax (mul q (mul k_t (pow d_sqrt -1))))"
        ).unwrap());
        assert!(!is_equivalent("(add 1 2)", "4").unwrap());
    }
    
    #[test]
    fn t_fusion_linear_relu() {
        assert_eq!(optimize_ast("(relu (linear w b x))"), "(linear-relu w b x)");
    }
    #[test]
    fn t_fusion_mlp_direct() {
        assert_eq!(optimize_ast("(linear w2 b2 (relu (linear w1 b1 x)))"), "(fused-mlp w1 b1 w2 b2 x)");
    }
    #[test]
    fn t_fusion_conv_bn() {
        let start = "(batchnorm w_bn b_bn m v (conv2d w_c b_c x 1_1 0_0 1_1 1) 1e-05)";
        let expected = "(fused_conv_bn w_c b_c w_bn b_bn m v x 1e-05 1_1 0_0 1_1 1)";
        assert_eq!(optimize_ast(start), expected);
    }
    
    #[test]
    fn t_fusion_linear_chain() {
        let start = "(linear w2 b2 (linear w1 b1 x))";
        let expected = "(linear (matmul w2 w1) (add (matmul w2 b1) b2) x)";
        assert_eq!(optimize_ast(start), expected);
    }
}