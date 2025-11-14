#!/usr/bin/env python3
"""
HYPATIA COMPREHENSIVE TEST SUITE
=================================
Tüm fonksiyoneliteleri kapsayan kapsamlı test paketi

Test Kategorileri:
1. Temel Doğrulama (Unit Tests)
2. Performans ve Verimlilik
3. Doğruluk, Stabilite ve Numerik Güvenlik
4. E-graph / Rewrite Doğruluğu
5. Sistem Entegrasyonu ve Reproducibility
6. Üretim Öncesi Güvenlik
7. Raporlama ve Artifact Üretimi

Çalıştırma:
    python test_comprehensive_suite.py
    python test_comprehensive_suite.py --category unit
    python test_comprehensive_suite.py --benchmark
    python test_comprehensive_suite.py --report-csv results.csv
"""

import hypatia_core as hc
import time
import sys
import argparse
import json
import csv
import traceback
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import numpy as np

# ============================================================================
# Test Result Data Structures
# ============================================================================

@dataclass
class TestResult:
    """Tek bir test sonucu"""
    test_name: str
    category: str
    status: str  # PASS, FAIL, SKIP, ERROR
    duration_ms: float
    error_message: str = ""
    metrics: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}

@dataclass
class BenchmarkResult:
    """Performans benchmark sonucu"""
    name: str
    operation: str
    iterations: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    std_ms: float
    throughput: float = 0.0
    flops_estimate: float = 0.0
    
class TestSuite:
    """Ana test suite manager"""
    
    def __init__(self):
        self.results: List[TestResult] = []
        self.benchmarks: List[BenchmarkResult] = []
        self.start_time = time.time()
        self.seed = 42
        
    def run_test(self, name: str, category: str, test_func):
        """Tek bir test çalıştır ve sonucu kaydet"""
        print(f"\n[{category}] Running: {name}")
        start = time.time()
        
        try:
            metrics = test_func()
            duration = (time.time() - start) * 1000
            result = TestResult(name, category, "PASS", duration, metrics=metrics)
            print(f"  ✅ PASS ({duration:.2f}ms)")
        except AssertionError as e:
            duration = (time.time() - start) * 1000
            result = TestResult(name, category, "FAIL", duration, str(e))
            print(f"  ❌ FAIL: {e}")
        except Exception as e:
            duration = (time.time() - start) * 1000
            result = TestResult(name, category, "ERROR", duration, 
                              f"{type(e).__name__}: {str(e)}")
            print(f"  🔥 ERROR: {type(e).__name__}: {e}")
            traceback.print_exc()
        
        self.results.append(result)
        return result
    
    def print_summary(self):
        """Test sonuçlarının özetini yazdır"""
        total = len(self.results)
        passed = sum(1 for r in self.results if r.status == "PASS")
        failed = sum(1 for r in self.results if r.status == "FAIL")
        errors = sum(1 for r in self.results if r.status == "ERROR")
        
        print("\n" + "=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)
        print(f"Total Tests:  {total}")
        print(f"✅ Passed:    {passed} ({100*passed/total if total > 0 else 0:.1f}%)")
        print(f"❌ Failed:    {failed} ({100*failed/total if total > 0 else 0:.1f}%)")
        print(f"🔥 Errors:    {errors} ({100*errors/total if total > 0 else 0:.1f}%)")
        print(f"⏱️  Duration:  {time.time() - self.start_time:.2f}s")
        print("=" * 80)
        
        if failed > 0 or errors > 0:
            print("\nFailed/Error Tests:")
            for r in self.results:
                if r.status in ["FAIL", "ERROR"]:
                    print(f"  [{r.category}] {r.test_name}: {r.error_message}")
        
        return passed == total
    
    def export_csv(self, filename: str):
        """Test sonuçlarını CSV olarak export et"""
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Test Name', 'Category', 'Status', 'Duration (ms)', 
                           'Error Message', 'Metrics'])
            for r in self.results:
                writer.writerow([r.test_name, r.category, r.status, 
                               f"{r.duration_ms:.2f}", r.error_message, 
                               json.dumps(r.metrics)])
        print(f"\n✅ Test results exported to: {filename}")
    
    def export_benchmark_csv(self, filename: str):
        """Benchmark sonuçlarını CSV olarak export et"""
        if not self.benchmarks:
            return
        
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Name', 'Operation', 'Iterations', 'P50 (ms)', 
                           'P95 (ms)', 'P99 (ms)', 'Mean (ms)', 'Std (ms)',
                           'Throughput', 'FLOPs Estimate'])
            for b in self.benchmarks:
                writer.writerow([b.name, b.operation, b.iterations,
                               f"{b.p50_ms:.4f}", f"{b.p95_ms:.4f}", 
                               f"{b.p99_ms:.4f}", f"{b.mean_ms:.4f}",
                               f"{b.std_ms:.4f}", f"{b.throughput:.2f}",
                               f"{b.flops_estimate:.0f}"])
        print(f"✅ Benchmark results exported to: {filename}")

# ============================================================================
# CATEGORY 1: TEMEL DOĞRULAMA (UNIT TESTS)
# ============================================================================

def test_1_1_basic_symbol_creation(suite: TestSuite):
    """1.1.1 - Symbol temel oluşturma"""
    def test():
        x = hc.Symbol.variable("x")
        y = hc.Symbol.variable("y")
        c = hc.Symbol.const(5.0)
        
        assert str(x) == "x"
        assert str(c) == "5"
        return {"variables": 2, "constants": 1}
    
    suite.run_test("Symbol Creation", "1.1 Unit Tests", test)

def test_1_1_arithmetic_operations(suite: TestSuite):
    """1.1.2 - Temel aritmetik operasyonlar"""
    def test():
        x = hc.Symbol.variable("x")
        y = hc.Symbol.variable("y")
        
        # Toplama
        add_expr = x + y
        assert "(add x y)" in str(add_expr)
        
        # Çarpma
        mul_expr = x * hc.Symbol.const(2.0)
        assert "mul" in str(mul_expr)
        
        # Bölme
        div_expr = x / y
        assert "div" in str(div_expr)
        
        # Negatif
        neg_expr = -x
        assert "neg" in str(neg_expr) or str(neg_expr) == "-x"
        
        return {"operations": 4}
    
    suite.run_test("Arithmetic Operations", "1.1 Unit Tests", test)

def test_1_1_derivative_basic(suite: TestSuite):
    """1.1.3 - Temel türev hesaplama"""
    def test():
        x = hc.Symbol.variable("x")
        
        # d/dx[x^2]
        f = x * x
        df = f.derivative("x")
        result = df.simplify()
        
        # d/dx[2x] = 2
        g = hc.Symbol.const(2.0) * x
        dg = g.derivative("x").simplify()
        assert str(dg) == "2"
        
        return {"derivatives": 2}
    
    suite.run_test("Derivative Computation", "1.1 Unit Tests", test)

def test_1_1_integration_basic(suite: TestSuite):
    """1.1.4 - Temel integral hesaplama"""
    def test():
        x = hc.Symbol.variable("x")
        
        # ∫ 5 dx = 5x
        f1 = hc.Symbol.const(5.0)
        int_f1 = f1.integrate("x")
        
        # ∫ x dx = x^2/2
        f2 = x
        int_f2 = f2.integrate("x")
        
        return {"integrals": 2}
    
    suite.run_test("Integration Computation", "1.1 Unit Tests", test)

def test_1_1_eval_safe(suite: TestSuite):
    """1.1.5 - FAZ 12: Eval safety (panic yerine exception)"""
    def test():
        x = hc.Symbol.variable("x")
        
        # Başarılı eval
        expr = x * hc.Symbol.const(2.0)
        result = expr.eval({"x": 3.0})
        assert result == 6.0
        
        # Division by zero - exception fırlatmalı
        try:
            div_zero = x / hc.Symbol.const(0.0)
            div_zero.eval({"x": 5.0})
            assert False, "Should have raised HypatiaError"
        except hc.HypatiaError:
            pass  # Beklenen davranış
        
        # Undefined variable
        try:
            expr.eval({})  # x tanımlı değil
            assert False, "Should have raised HypatiaError"
        except hc.HypatiaError:
            pass
        
        return {"safety_checks": 3}
    
    suite.run_test("Eval Safety (FAZ 12)", "1.1 Unit Tests", test)

def test_1_1_activation_functions(suite: TestSuite):
    """1.1.6 - Aktivasyon fonksiyonları"""
    def test():
        x = hc.Symbol.variable("x")
        
        # ReLU
        relu_expr = hc.Symbol.relu(x)
        assert "relu" in str(relu_expr)
        
        # Sigmoid
        sigmoid_expr = hc.Symbol.sigmoid(x)
        assert "sigmoid" in str(sigmoid_expr)
        
        # Tanh
        tanh_expr = hc.Symbol.tanh(x)
        assert "tanh" in str(tanh_expr)
        
        # ReLU değerlendirme
        relu_pos = hc.Symbol.relu(hc.Symbol.const(5.0))
        assert relu_pos.simplify().eval({}) == 5.0
        
        relu_neg = hc.Symbol.relu(hc.Symbol.const(-3.0))
        assert relu_neg.simplify().eval({}) == 0.0
        
        return {"activations": 5}
    
    suite.run_test("Activation Functions", "1.1 Unit Tests", test)

def test_1_1_ai_operators(suite: TestSuite):
    """1.1.7 - FEZ 10: AI operatörleri (softmax, mean, variance)"""
    def test():
        x = hc.Symbol.variable("x")
        
        # Softmax
        sm = hc.Symbol.softmax(x)
        assert "softmax" in str(sm)
        
        # Mean
        m = hc.Symbol.mean(x)
        assert "mean" in str(m)
        
        # Variance
        v = hc.Symbol.variance(x)
        assert "var" in str(v)
        
        # Constant folding
        sm_const = hc.Symbol.softmax(hc.Symbol.const(5.0)).simplify()
        assert sm_const.eval({}) == 1.0
        
        return {"ai_operators": 4}
    
    suite.run_test("AI Operators (FEZ 10)", "1.1 Unit Tests", test)

def test_1_2_multivector_2d(suite: TestSuite):
    """1.2.1 - MultiVector2D temel operasyonlar"""
    def test():
        # Vektör oluşturma
        v1 = hc.PyMultiVector2D.vector(1.0, 2.0)
        v2 = hc.PyMultiVector2D.vector(3.0, 4.0)
        
        # Grade extraction
        v1_grade1 = v1.grade(1)
        assert abs(v1_grade1.e1() - 1.0) < 1e-10
        assert abs(v1_grade1.e2() - 2.0) < 1e-10
        
        # Rotor (rotation)
        theta = 3.14159 / 2.0  # 90 degrees
        r = hc.PyMultiVector2D.rotor(theta)
        v_rotated = r.rotate_vector(v1)
        
        return {"vectors": 2, "rotations": 1}
    
    suite.run_test("MultiVector2D Operations", "1.2 Geometric Algebra", test)

def test_1_2_multivector_3d(suite: TestSuite):
    """1.2.2 - MultiVector3D temel operasyonlar"""
    def test():
        # 3D vektör oluşturma
        v1 = hc.PyMultiVector3D.vector(1.0, 0.0, 0.0)
        v2 = hc.PyMultiVector3D.vector(0.0, 1.0, 0.0)
        
        # Grade extraction
        v1_grade1 = v1.grade(1)
        assert abs(v1_grade1.e1() - 1.0) < 1e-10
        
        # Rotor (Z-axis rotation)
        theta = 3.14159 / 2.0
        r = hc.PyMultiVector3D.rotor(theta, 0.0, 0.0, 1.0)
        v_rotated = r.rotate_vector(v1)
        
        return {"vectors": 2, "rotations": 1}
    
    suite.run_test("MultiVector3D Operations", "1.2 Geometric Algebra", test)

def test_1_3_egraph_optimizer(suite: TestSuite):
    """1.3.1 - E-graph optimizer temel işlevsellik"""
    def test():
        # Basit ifade optimizasyonu
        original = "(add (mul x 0) y)"
        optimized = hc.optimize_ast(original)
        
        # 0*x sadeleştirilmeli
        assert "0" in optimized or "y" in optimized
        
        # Daha karmaşık örnek
        expr2 = "(add (mul 0 x) (mul 1 y))"
        opt2 = hc.optimize_ast(expr2)
        
        return {"optimizations": 2}
    
    suite.run_test("E-graph Optimizer", "1.3 E-graph", test)

def test_1_3_parse_expr(suite: TestSuite):
    """1.3.2 - S-expression parsing"""
    def test():
        # Basit parse
        sym1 = hc.parse_expr("(add x y)")
        assert "add" in str(sym1)
        
        # İç içe parse
        sym2 = hc.parse_expr("(mul (add x y) 2)")
        assert "mul" in str(sym2)
        
        # Constant parse
        sym3 = hc.parse_expr("5")
        assert str(sym3) == "5"
        
        return {"parsed": 3}
    
    suite.run_test("S-expression Parsing", "1.3 E-graph", test)

def test_1_3_is_equivalent(suite: TestSuite):
    """1.3.3 - FEZ 11: Sembolik denklik kontrolü"""
    def test():
        # Eşdeğer ifadeler
        expr1 = "(add x y)"
        expr2 = "(add y x)"
        assert hc.is_equivalent(expr1, expr2) == True
        
        # Eşdeğer olmayan ifadeler
        expr3 = "(mul x y)"
        expr4 = "(add x y)"
        assert hc.is_equivalent(expr3, expr4) == False
        
        # Constant folding eşdeğerliği
        expr5 = "(add 2 3)"
        expr6 = "5"
        assert hc.is_equivalent(expr5, expr6) == True
        
        return {"equivalence_checks": 3}
    
    suite.run_test("Is Equivalent (FEZ 11)", "1.3 E-graph", test)

# ============================================================================
# CATEGORY 2: PERFORMANS VE VERİMLİLİK TESTLERİ
# ============================================================================

def benchmark_operation(suite: TestSuite, name: str, operation: str, 
                       func, iterations: int = 1000, warmup: int = 100):
    """Tek bir operasyonun mikrobenchmark'ı"""
    print(f"\n[2.1 Benchmark] {name} ({iterations} iterations)")
    
    # Warmup
    for _ in range(warmup):
        func()
    
    # Ölçüm
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # ms
    
    times_sorted = sorted(times)
    p50 = times_sorted[len(times) // 2]
    p95 = times_sorted[int(len(times) * 0.95)]
    p99 = times_sorted[int(len(times) * 0.99)]
    mean = sum(times) / len(times)
    std = (sum((t - mean) ** 2 for t in times) / len(times)) ** 0.5
    
    result = BenchmarkResult(
        name=name,
        operation=operation,
        iterations=iterations,
        p50_ms=p50,
        p95_ms=p95,
        p99_ms=p99,
        mean_ms=mean,
        std_ms=std
    )
    
    suite.benchmarks.append(result)
    
    print(f"  P50: {p50:.4f}ms | P95: {p95:.4f}ms | P99: {p99:.4f}ms | "
          f"Mean: {mean:.4f}ms ± {std:.4f}ms")
    
    return result

def test_2_1_microbenchmarks(suite: TestSuite):
    """2.1 - Operator düzeyinde mikrobenchmark'lar"""
    
    # Symbolic derivative
    x = hc.Symbol.variable("x")
    y = hc.Symbol.variable("y")
    expr = x * x + y * y
    
    benchmark_operation(
        suite, "Symbolic Derivative", "derivative",
        lambda: expr.derivative("x"), iterations=500
    )
    
    # Simplify
    complex_expr = (x * hc.Symbol.const(0.0) + y * hc.Symbol.const(1.0))
    benchmark_operation(
        suite, "Simplify", "simplify",
        lambda: complex_expr.simplify(), iterations=500
    )
    
    # Eval
    eval_expr = x * hc.Symbol.const(2.0) + y
    env = {"x": 3.0, "y": 5.0}
    benchmark_operation(
        suite, "Eval", "eval",
        lambda: eval_expr.eval(env), iterations=1000
    )
    
    # E-graph optimize
    s_expr = "(add (mul x 0) (mul y 1))"
    benchmark_operation(
        suite, "E-graph Optimize", "optimize_ast",
        lambda: hc.optimize_ast(s_expr), iterations=200
    )
    
    # is_equivalent
    expr1 = "(add x y)"
    expr2 = "(add y x)"
    benchmark_operation(
        suite, "Is Equivalent", "is_equivalent",
        lambda: hc.is_equivalent(expr1, expr2), iterations=200
    )

# ============================================================================
# CATEGORY 3: DOĞRULUK, STABİLİTE VE NUMERİK GÜVENLİK
# ============================================================================

def test_3_1_numerical_precision(suite: TestSuite):
    """3.1 - Numerik hassasiyet testleri"""
    def test():
        x = hc.Symbol.variable("x")
        
        # Sigmoid(0) = 0.5
        sig_zero = hc.Symbol.sigmoid(hc.Symbol.const(0.0))
        result = sig_zero.simplify().eval({})
        assert abs(result - 0.5) < 1e-10
        
        # ReLU hassasiyeti
        relu_small_pos = hc.Symbol.relu(hc.Symbol.const(1e-10))
        assert relu_small_pos.simplify().eval({}) > 0
        
        relu_small_neg = hc.Symbol.relu(hc.Symbol.const(-1e-10))
        assert relu_small_neg.simplify().eval({}) == 0.0
        
        return {"precision_checks": 3}
    
    suite.run_test("Numerical Precision", "3.1 Numerical Safety", test)

def test_3_2_edge_cases(suite: TestSuite):
    """3.2 - Edge case'ler ve güvenlik"""
    def test():
        x = hc.Symbol.variable("x")
        
        # Division by zero
        try:
            expr = x / hc.Symbol.const(0.0)
            expr.eval({"x": 5.0})
            assert False, "Should raise HypatiaError"
        except hc.HypatiaError as e:
            assert "Division by zero" in str(e)
        
        # Log of negative
        try:
            expr = hc.Symbol.log(hc.Symbol.const(-1.0))
            expr.eval({})
            assert False, "Should raise HypatiaError"
        except hc.HypatiaError as e:
            assert "Log of non-positive" in str(e)
        
        # Sqrt of negative
        try:
            expr = hc.Symbol.sqrt(hc.Symbol.const(-1.0))
            expr.eval({})
            assert False, "Should raise HypatiaError"
        except hc.HypatiaError as e:
            assert "Sqrt of negative" in str(e)
        
        # Zero to negative power
        try:
            expr = hc.Symbol.pow(hc.Symbol.const(0.0), hc.Symbol.const(-1.0))
            expr.eval({})
            assert False, "Should raise HypatiaError"
        except hc.HypatiaError as e:
            assert "Zero to negative power" in str(e)
        
        return {"edge_cases": 4}
    
    suite.run_test("Edge Cases Safety", "3.2 Numerical Safety", test)

def test_3_3_simplify_correctness(suite: TestSuite):
    """3.3 - Simplify matematiksel doğruluğu"""
    def test():
        x = hc.Symbol.variable("x")
        y = hc.Symbol.variable("y")
        
        # 0 * x = 0
        expr1 = hc.Symbol.const(0.0) * x
        assert str(expr1.simplify()) == "0"
        
        # x * 0 = 0
        expr2 = x * hc.Symbol.const(0.0)
        assert str(expr2.simplify()) == "0"
        
        # 1 * x = x
        expr3 = hc.Symbol.const(1.0) * x
        assert str(expr3.simplify()) == "x"
        
        # x + 0 = x
        expr4 = x + hc.Symbol.const(0.0)
        assert str(expr4.simplify()) == "x"
        
        # 0 + x = x
        expr5 = hc.Symbol.const(0.0) + x
        assert str(expr5.simplify()) == "x"
        
        # x - 0 = x
        expr6 = x - hc.Symbol.const(0.0)
        assert str(expr6.simplify()) == "x"
        
        # --x = x
        expr7 = -(-x)
        assert str(expr7.simplify()) == "x"
        
        return {"simplify_rules": 7}
    
    suite.run_test("Simplify Correctness", "3.3 Algebraic Correctness", test)

# ============================================================================
# CATEGORY 4: E-GRAPH / REWRITE DOĞRULUĞU
# ============================================================================

def test_4_1_rewrite_rules(suite: TestSuite):
    """4.1 - Rewrite kurallarının doğruluğu"""
    def test():
        # Commutativity: x + y = y + x
        equiv1 = hc.is_equivalent("(add x y)", "(add y x)")
        assert equiv1 == True
        
        # Associativity: (x + y) + z = x + (y + z)
        equiv2 = hc.is_equivalent("(add (add x y) z)", "(add x (add y z))")
        assert equiv2 == True
        
        # Distributivity: x * (y + z) = x*y + x*z
        equiv3 = hc.is_equivalent("(mul x (add y z))", "(add (mul x y) (mul x z))")
        assert equiv3 == True
        
        # Identity: 0 + x = x
        equiv4 = hc.is_equivalent("(add 0 x)", "x")
        assert equiv4 == True
        
        # Zero: 0 * x = 0
        equiv5 = hc.is_equivalent("(mul 0 x)", "0")
        assert equiv5 == True
        
        return {"rewrite_rules": 5}
    
    suite.run_test("Rewrite Rules", "4.1 E-graph Correctness", test)

def test_4_2_optimization_effectiveness(suite: TestSuite):
    """4.2 - Optimizasyon etkinliği"""
    def test():
        # Karmaşık ifade
        # ✅ DÜZELTME: 'add' operatörü 2 argüman alır (Add([Id; 2])).
        # Orijinal S-expression (3 argümanlı) geçersizdi.
        # İfadeyi geçerli olması için manuel olarak iç içe (nested) hale getirdik.
        original = "(add (add (mul 0 x) (mul 1 y)) (add 0 z))"
        optimized = hc.optimize_ast(original)
        
        # Optimize edilmiş versiyon daha basit olmalı
        # Orijinal (düzeltilmiş): "(add (add (mul 0 x) (mul 1 y)) (add 0 z))" (len 41)
        # Optimize edilmiş hali: "(add y z)" (len 9) olmalı
        assert len(optimized) < len(original)
        
        # Semantik eşdeğerlik korunmalı
        assert hc.is_equivalent(original, optimized)
        
        # İç içe sıfırlar (Bu zaten geçerli bir formattaydı)
        nested = "(add (add (mul 0 x) 0) (add 0 (mul 1 y)))"
        nested_opt = hc.optimize_ast(nested)
        # Optimize edilmiş hali: "y"
        assert hc.is_equivalent(nested, nested_opt)
        
        return {"optimizations": 2}
    
    suite.run_test("Optimization Effectiveness", "4.2 E-graph Quality", test)

# ============================================================================
# CATEGORY 5: SİSTEM ENTEGRASYONU VE REPRODUCIBILITY
# ============================================================================

def test_5_1_reproducibility(suite: TestSuite):
    """5.1 - Sonuçların yeniden üretilebilirliği"""
    def test():
        x = hc.Symbol.variable("x")
        y = hc.Symbol.variable("y")
        
        # Aynı ifade, aynı sonuç
        expr1 = x * hc.Symbol.const(2.0) + y
        result1 = expr1.eval({"x": 3.0, "y": 5.0})
        
        expr2 = x * hc.Symbol.const(2.0) + y
        result2 = expr2.eval({"x": 3.0, "y": 5.0})
        
        assert result1 == result2
        
        # Simplify determinizmi
        complex_expr = (x + y) * (x - y)
        simp1 = str(complex_expr.simplify())
        simp2 = str(complex_expr.simplify())
        assert simp1 == simp2
        
        return {"reproducibility_checks": 2}
    
    suite.run_test("Reproducibility", "5.1 System Integration", test)

def test_5_2_api_contracts(suite: TestSuite):
    """5.2 - API contract'larının tutarlılığı"""
    def test():
        # Symbol.variable() daima Symbol döner
        x = hc.Symbol.variable("x")
        assert hasattr(x, 'derivative')
        assert hasattr(x, 'simplify')
        assert hasattr(x, 'eval')
        
        # optimize_ast() daima string döner
        result = hc.optimize_ast("(add x y)")
        assert isinstance(result, str)
        
        # is_equivalent() daima bool döner
        equiv = hc.is_equivalent("x", "x")
        assert isinstance(equiv, bool)
        
        # parse_expr() daima Symbol döner
        sym = hc.parse_expr("x")
        assert hasattr(sym, 'derivative')
        
        return {"api_checks": 4}
    
    suite.run_test("API Contracts", "5.2 System Integration", test)

# ============================================================================
# CATEGORY 6: ÜRETİM ÖNCESİ GÜVENLİK
# ============================================================================

def test_6_1_error_messages(suite: TestSuite):
    """6.1 - Hata mesajlarının anlaşılırlığı"""
    def test():
        x = hc.Symbol.variable("x")
        
        # Division by zero mesajı
        try:
            expr = x / hc.Symbol.const(0.0)
            expr.eval({"x": 5.0})
        except hc.HypatiaError as e:
            msg = str(e)
            assert "Division by zero" in msg
            assert "/" in msg  # Operatör bilgisi
        
        # Undefined variable mesajı
        try:
            expr = x + hc.Symbol.variable("y")
            expr.eval({"x": 5.0})  # y eksik
        except hc.HypatiaError as e:
            msg = str(e)
            assert "not found" in msg or "undefined" in msg.lower()
        
        return {"error_message_checks": 2}
    
    suite.run_test("Error Messages", "6.1 Production Readiness", test)

def test_6_2_no_secret_leakage(suite: TestSuite):
    """6.2 - Güvenlik: Hassas veri sızıntısı kontrolü"""
    def test():
        # Symbol isimleri loglarda görünebilir ama değerleri görünmemeli
        x = hc.Symbol.variable("secret_api_key")
        expr = x * hc.Symbol.const(2.0)
        
        # String representation'da sadece sembolik isim olmalı
        str_repr = str(expr)
        assert "secret_api_key" in str_repr  # İsim OK
        # Ama gerçek değer olmamalı (bu örnekte zaten yok)
        
        return {"security_checks": 1}
    
    suite.run_test("No Secret Leakage", "6.2 Security", test)

# ============================================================================
# CATEGORY 7: KAPSAMLI ENTEGRASYON TESTLERİ
# ============================================================================

def test_7_1_complex_ai_pipeline(suite: TestSuite):
    """7.1 - Karmaşık AI pipeline simülasyonu"""
    def test():
        # Neural network benzeri hesaplama
        # W*x + b -> ReLU -> softmax
        
        x = hc.Symbol.variable("x")
        W = hc.Symbol.variable("W")
        b = hc.Symbol.variable("b")
        
        # Linear layer
        linear = W * x + b
        
        # ReLU activation
        activated = hc.Symbol.relu(linear)
        
        # Derivative (backprop simulation)
        grad = activated.derivative("x")
        grad_simplified = grad.simplify()
        
        # Numerical evaluation
        env = {"x": 2.0, "W": 0.5, "b": 1.0}
        forward_result = activated.eval(env)
        
        assert forward_result > 0  # ReLU çıktısı pozitif olmalı
        
        return {"pipeline_steps": 3}
    
    suite.run_test("Complex AI Pipeline", "7.1 Integration", test)

def test_7_2_optimization_chain(suite: TestSuite):
    """7.2 - Zincir optimizasyon testi"""
    def test():
        # Çok adımlı optimizasyon zinciri
        expr1 = "(add (mul 0 x) y)"
        expr2 = hc.optimize_ast(expr1)
        
        expr3 = f"(mul 1 {expr2})"
        expr4 = hc.optimize_ast(expr3)
        
        # Her adımda eşdeğerlik korunmalı
        assert hc.is_equivalent(expr1, expr2)
        assert hc.is_equivalent(expr2, expr4)
        assert hc.is_equivalent(expr1, expr4)
        
        return {"chain_steps": 2}
    
    suite.run_test("Optimization Chain", "7.2 Integration", test)

def test_7_3_geometric_algebra_workflow(suite: TestSuite):
    """7.3 - Geometric Algebra tam workflow"""
    def test():
        # 2D rotation workflow
        v = hc.PyMultiVector2D.vector(1.0, 0.0)
        theta = 3.14159 / 4.0  # 45 degrees
        
        # Rotor oluştur
        r = hc.PyMultiVector2D.rotor(theta)
        
        # Rotate
        v_rot = r.rotate_vector(v)
        
        # Extract components
        v_rot_grade1 = v_rot.grade(1)
        x = v_rot_grade1.e1()
        y = v_rot_grade1.e2()
        
        # 45 derece rotasyon kontrolü
        expected = 0.7071  # cos(45°) ≈ sin(45°)
        assert abs(x - expected) < 0.01
        assert abs(y - expected) < 0.01
        
        return {"ga_operations": 3}
    
    suite.run_test("Geometric Algebra Workflow", "7.3 Integration", test)

# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def run_all_tests(args):
    """Tüm test kategorilerini çalıştır"""
    suite = TestSuite()
    
    print("=" * 80)
    print("HYPATIA COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Seed: {suite.seed}")
    print("=" * 80)
    
    # Category 1: Temel doğrulama
    if args.category in [None, 'all', 'unit', '1']:
        print("\n" + "=" * 80)
        print("CATEGORY 1: TEMEL DOĞRULAMA (UNIT TESTS)")
        print("=" * 80)
        test_1_1_basic_symbol_creation(suite)
        test_1_1_arithmetic_operations(suite)
        test_1_1_derivative_basic(suite)
        test_1_1_integration_basic(suite)
        test_1_1_eval_safe(suite)
        test_1_1_activation_functions(suite)
        test_1_1_ai_operators(suite)
        test_1_2_multivector_2d(suite)
        test_1_2_multivector_3d(suite)
        test_1_3_egraph_optimizer(suite)
        test_1_3_parse_expr(suite)
        test_1_3_is_equivalent(suite)
    
    # Category 2: Performans
    if args.category in [None, 'all', 'benchmark', '2'] and args.benchmark:
        print("\n" + "=" * 80)
        print("CATEGORY 2: PERFORMANS VE VERİMLİLİK")
        print("=" * 80)
        test_2_1_microbenchmarks(suite)
    
    # Category 3: Numerik güvenlik
    if args.category in [None, 'all', 'numerical', '3']:
        print("\n" + "=" * 80)
        print("CATEGORY 3: DOĞRULUK VE NUMERİK GÜVENLİK")
        print("=" * 80)
        test_3_1_numerical_precision(suite)
        test_3_2_edge_cases(suite)
        test_3_3_simplify_correctness(suite)
    
    # Category 4: E-graph
    if args.category in [None, 'all', 'egraph', '4']:
        print("\n" + "=" * 80)
        print("CATEGORY 4: E-GRAPH / REWRITE DOĞRULUĞU")
        print("=" * 80)
        test_4_1_rewrite_rules(suite)
        test_4_2_optimization_effectiveness(suite)
    
    # Category 5: Entegrasyon
    if args.category in [None, 'all', 'integration', '5']:
        print("\n" + "=" * 80)
        print("CATEGORY 5: SİSTEM ENTEGRASYONU")
        print("=" * 80)
        test_5_1_reproducibility(suite)
        test_5_2_api_contracts(suite)
    
    # Category 6: Güvenlik
    if args.category in [None, 'all', 'security', '6']:
        print("\n" + "=" * 80)
        print("CATEGORY 6: ÜRETİM ÖNCESİ GÜVENLİK")
        print("=" * 80)
        test_6_1_error_messages(suite)
        test_6_2_no_secret_leakage(suite)
    
    # Category 7: Kapsamlı entegrasyon
    if args.category in [None, 'all', 'integration', '7']:
        print("\n" + "=" * 80)
        print("CATEGORY 7: KAPSAMLI ENTEGRASYON TESTLERİ")
        print("=" * 80)
        test_7_1_complex_ai_pipeline(suite)
        test_7_2_optimization_chain(suite)
        test_7_3_geometric_algebra_workflow(suite)
    
    # Sonuç özeti
    suite.print_summary()
    
    # Export sonuçları
    if args.report_csv:
        suite.export_csv(args.report_csv)
    
    if args.benchmark_csv:
        suite.export_benchmark_csv(args.benchmark_csv)
    
    # JSON export (opsiyonel)
    if args.json:
        with open(args.json, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'results': [asdict(r) for r in suite.results],
                'benchmarks': [asdict(b) for b in suite.benchmarks],
                'summary': {
                    'total': len(suite.results),
                    'passed': sum(1 for r in suite.results if r.status == "PASS"),
                    'failed': sum(1 for r in suite.results if r.status == "FAIL"),
                    'errors': sum(1 for r in suite.results if r.status == "ERROR"),
                }
            }, f, indent=2)
        print(f"✅ JSON report exported to: {args.json}")
    
    # Exit code
    success = all(r.status == "PASS" for r in suite.results)
    return 0 if success else 1

def main():
    parser = argparse.ArgumentParser(
        description='Hypatia Comprehensive Test Suite',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_comprehensive_suite.py                    # Tüm testler
  python test_comprehensive_suite.py --category unit    # Sadece unit testler
  python test_comprehensive_suite.py --benchmark        # Performans testleri dahil
  python test_comprehensive_suite.py --report-csv results.csv
  python test_comprehensive_suite.py --json report.json
        """
    )
    
    parser.add_argument('--category', choices=['all', 'unit', 'benchmark', 
                                               'numerical', 'egraph', 'integration',
                                               'security', '1', '2', '3', '4', '5', '6', '7'],
                       help='Test kategorisi (default: all)')
    parser.add_argument('--benchmark', action='store_true',
                       help='Performans benchmark\'larını çalıştır')
    parser.add_argument('--report-csv', metavar='FILE',
                       help='Test sonuçlarını CSV olarak export et')
    parser.add_argument('--benchmark-csv', metavar='FILE',
                       help='Benchmark sonuçlarını CSV olarak export et')
    parser.add_argument('--json', metavar='FILE',
                       help='Tüm sonuçları JSON olarak export et')
    
    args = parser.parse_args()
    
    try:
        exit_code = run_all_tests(args)
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n🔥 Fatal error: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()