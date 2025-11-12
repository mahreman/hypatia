use egg::{
    define_language, rewrite, CostFunction, Extractor, Id, Language, RecExpr, Rewrite, Runner,
    Symbol as EggSymbol, Analysis, DidMerge, EGraph,
};
use ordered_float::NotNan;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::time::Duration;

// ============================================================================
// HYPATIA LANGUAGE DEFINITION
// ============================================================================

define_language! {
    pub enum HypatiaLang {
        // Aritmetik
        "add" = Add([Id; 2]),
        "mul" = Mul([Id; 2]),
        "sub" = Sub([Id; 2]),
        "div" = Div([Id; 2]),
        "neg" = Neg(Id),

        // Matematiksel fonksiyonlar
        "exp" = Exp(Id),
        "log" = Log(Id),
        "sqrt" = Sqrt(Id),
        "pow" = Pow([Id; 2]),

        // Aktivasyonlar
        "relu" = ReLU(Id),
        "relu_grad" = ReLUGrad(Id),
        "sigmoid" = Sigmoid(Id),
        "tanh" = Tanh(Id),

        // Yardımcılar
        "max" = Max([Id; 2]),
        "min" = Min([Id; 2]),

        // Taban
        Constant(NotNan<f64>),
        Var(EggSymbol),
    }
}

// ============================================================================
// Constant Folding Analysis (FEZ 4.2 - DÜZELTİLDİ)
// ============================================================================

#[derive(Default)]
pub struct ConstantFoldingAnalysis;

impl Analysis<HypatiaLang> for ConstantFoldingAnalysis {
    type Data = Option<f64>; // NotNan yerine f64

    fn make(
        egraph: &EGraph<HypatiaLang, Self>,
        enode: &HypatiaLang,
    ) -> Self::Data {
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
            _ => None,
        }
    }

    fn merge(&mut self, to: &mut Self::Data, from: Self::Data) -> DidMerge {
        egg::merge_option(to, from, |a, b| {
            // DÜZELTME: (a - b) -> (*a - b)
            // 'a' &mut f64 tipindedir, 'b' ise f64 tipindedir.
            if (*a - b).abs() < 1e-9 {
                DidMerge(false, false)
            } else {
                DidMerge(false, false) 
            }
        })
    }

    fn modify(
        egraph: &mut EGraph<HypatiaLang, Self>,
        id: Id,
    ) {
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
// Basit FLOPs Cost Model (Değişiklik yok)
// ============================================================================

pub struct FLOPsCost;

impl CostFunction<HypatiaLang> for FLOPsCost {
    type Cost = usize;
    fn cost<C>(&mut self, enode: &HypatiaLang, mut costs: C) -> Self::Cost
    where
        C: FnMut(Id) -> Self::Cost,
    {
        let children_cost: usize = enode.children().iter().map(|&id| costs(id)).sum();
        let op_cost = match enode {
            HypatiaLang::Add(_) => 1,
            HypatiaLang::Sub(_) => 1,
            HypatiaLang::Neg(_) => 1,
            HypatiaLang::Mul(_) => 10,
            HypatiaLang::Div(_) => 15,
            HypatiaLang::Exp(_) => 20,
            HypatiaLang::Log(_) => 20,
            HypatiaLang::Sqrt(_) => 15,
            HypatiaLang::Pow(_) => 25,
            HypatiaLang::ReLU(_) => 1,
            HypatiaLang::ReLUGrad(_) => 2,
            HypatiaLang::Sigmoid(_) => 25,
            HypatiaLang::Tanh(_) => 25,
            HypatiaLang::Max(_) => 2,
            HypatiaLang::Min(_) => 1,
            HypatiaLang::Constant(_) => 0,
            HypatiaLang::Var(_) => 0,
        };
        op_cost + children_cost
    }
}

// ============================================================================
// Rewrite Kuralları (Değişiklik yok)
// ============================================================================

fn get_rules() -> Vec<Rewrite<HypatiaLang, ConstantFoldingAnalysis>> {
    vec![
        rewrite!("commute-add"; "(add ?a ?b)" => "(add ?b ?a)"),
        rewrite!("commute-mul"; "(mul ?a ?b)" => "(mul ?b ?a)"),
        rewrite!("commute-max"; "(max ?a ?b)" => "(max ?b ?a)"),
        rewrite!("commute-min"; "(min ?a ?b)" => "(min ?b ?a)"),
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
        rewrite!("pow-0";    "(pow ?x 0)" => "1"),
        rewrite!("pow-1";    "(pow ?x 1)" => "?x"),
        rewrite!("exp-log"; "(exp (log ?x))" => "?x"),
        rewrite!("log-exp"; "(log (exp ?x))" => "?x"),
        rewrite!("sqrt-pow2"; "(sqrt (pow ?x 2))" => "?x"),
        rewrite!("relu-idem";    "(relu (relu ?x))" => "(relu ?x)"),
        rewrite!("max0-to-relu"; "(max ?x 0)"      => "(relu ?x)"),
        rewrite!("max-idem"; "(max ?x ?x)" => "?x"),
        rewrite!("min-idem"; "(min ?x ?x)" => "?x"),
    ]
}

// ============================================================================
// Yardımcı
// ============================================================================

fn rec_to_string(expr: &RecExpr<HypatiaLang>) -> String {
    format!("{}", expr)
}

// ============================================================================
// Dış API: Güvenli optimize (Artık Analysis kullanıyor)
// ============================================================================

pub fn optimize_ast(expr_str: &str) -> String {
    let res = catch_unwind(AssertUnwindSafe(|| {
        let start_expr: RecExpr<HypatiaLang> = match expr_str.parse() {
            Ok(expr) => expr,
            Err(e) => {
                return format!("(error \"Parse Error: {}\")", e);
            }
        };

        let rules = get_rules();
        let runner = Runner::default()
            .with_egraph(egg::EGraph::new(ConstantFoldingAnalysis))
            .with_node_limit(20_000)
            .with_iter_limit(30)
            .with_time_limit(Duration::from_millis(150))
            .with_expr(&start_expr)
            .run(&rules);

        let extractor = Extractor::new(&runner.egraph, FLOPsCost);
        let (_best_cost, best_expr) = extractor.find_best(runner.roots[0]);

        rec_to_string(&best_expr)
    }));

    match res {
        Ok(s) => s,
        Err(_) => "(error \"optimizer panic caught\")".to_string(),
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn t_zero_mul() {
        assert_eq!(optimize_ast("(mul x 0)"), "0");
    }

    #[test]
    fn t_neg_mul() {
        assert_eq!(optimize_ast("(neg (mul a b))"), "(mul (neg a) b)");
    }

    #[test]
    fn t_sigmoid_zero_analysis() {
        assert_eq!(optimize_ast("(sigmoid 0)"), "0.5");
    }
    
    #[test]
    fn t_constant_folding_add() {
        assert_eq!(optimize_ast("(add 1 2)"), "3");
    }

    #[test]
    fn t_constant_folding_pow() {
        assert_eq!(optimize_ast("(pow 2 -1)"), "0.5");
    }

    #[test]
    fn t_div_to_mul_pow_analysis() {
        // 1. E-graph (div x 2)'yi görür
        // 2. Kural: (mul x (pow 2 -1)) oluşturur
        // 3. Analiz: (pow 2 -1)'i 0.5'e çevirir
        // 4. E-graph'te iki seçenek olur:
        //    a) (div x 2)        -> Maliyet: div(15) + x(0) + 2(0) = 15
        //    b) (mul x 0.5)      -> Maliyet: mul(10) + x(0) + 0.5(0) = 10
        // 5. Extractor, maliyeti 10 olan (b) seçeneğini seçer.
        assert_eq!(optimize_ast("(div x 2)"), "(mul x 0.5)");
    }
}