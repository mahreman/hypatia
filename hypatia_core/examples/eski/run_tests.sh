#!/bin/bash
#
# HYPATIA TEST SUITE QUICK RUN SCRIPT
# ====================================
# Tüm testleri çalıştırır ve sonuçları kaydeder
#

set -e  # Exit on error

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║         HYPATIA COMPREHENSIVE TEST SUITE - QUICK RUN              ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""

# Renk tanımları
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_DIR="test_results_${TIMESTAMP}"

echo -e "${BLUE}[INFO]${NC} Creating results directory: ${RESULTS_DIR}"
mkdir -p "${RESULTS_DIR}"

# 1. Unit Tests
echo ""
echo -e "${BLUE}[STEP 1/5]${NC} Running Unit Tests..."
python test_comprehensive_suite.py \
    --category unit \
    --report-csv "${RESULTS_DIR}/unit_tests.csv" \
    --json "${RESULTS_DIR}/unit_tests.json" \
    | tee "${RESULTS_DIR}/unit_tests.log"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Unit tests PASSED${NC}"
else
    echo -e "${RED}❌ Unit tests FAILED${NC}"
    exit 1
fi

# 2. Numerical Safety Tests
echo ""
echo -e "${BLUE}[STEP 2/5]${NC} Running Numerical Safety Tests..."
python test_comprehensive_suite.py \
    --category numerical \
    --report-csv "${RESULTS_DIR}/numerical_tests.csv" \
    | tee "${RESULTS_DIR}/numerical_tests.log"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Numerical tests PASSED${NC}"
else
    echo -e "${RED}❌ Numerical tests FAILED${NC}"
    exit 1
fi

# 3. E-graph Tests
echo ""
echo -e "${BLUE}[STEP 3/5]${NC} Running E-graph Tests..."
python test_comprehensive_suite.py \
    --category egraph \
    --report-csv "${RESULTS_DIR}/egraph_tests.csv" \
    | tee "${RESULTS_DIR}/egraph_tests.log"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ E-graph tests PASSED${NC}"
else
    echo -e "${RED}❌ E-graph tests FAILED${NC}"
    exit 1
fi

# 4. Integration Tests
echo ""
echo -e "${BLUE}[STEP 4/5]${NC} Running Integration Tests..."
python test_comprehensive_suite.py \
    --category integration \
    --report-csv "${RESULTS_DIR}/integration_tests.csv" \
    | tee "${RESULTS_DIR}/integration_tests.log"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Integration tests PASSED${NC}"
else
    echo -e "${RED}❌ Integration tests FAILED${NC}"
    exit 1
fi

# 5. Performance Benchmarks (optional but recommended)
echo ""
echo -e "${BLUE}[STEP 5/5]${NC} Running Performance Benchmarks..."
python test_comprehensive_suite.py \
    --benchmark \
    --benchmark-csv "${RESULTS_DIR}/benchmarks.csv" \
    --json "${RESULTS_DIR}/benchmarks.json" \
    | tee "${RESULTS_DIR}/benchmarks.log"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Benchmarks completed${NC}"
else
    echo -e "${YELLOW}⚠️  Benchmarks failed (non-critical)${NC}"
fi

# Summary
echo ""
echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║                         TEST SUMMARY                               ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""
echo -e "${GREEN}✅ ALL CRITICAL TESTS PASSED!${NC}"
echo ""
echo "Results saved to: ${RESULTS_DIR}/"
echo ""
echo "Files:"
echo "  📄 unit_tests.csv         - Unit test results"
echo "  📄 numerical_tests.csv    - Numerical safety results"
echo "  📄 egraph_tests.csv       - E-graph test results"
echo "  📄 integration_tests.csv  - Integration test results"
echo "  📊 benchmarks.csv         - Performance metrics"
echo "  📋 *.log                  - Full console logs"
echo "  📦 *.json                 - Detailed JSON reports"
echo ""

# Generate summary HTML (optional)
if command -v python3 &> /dev/null; then
    echo -e "${BLUE}[BONUS]${NC} Generating HTML summary..."
    
    cat > "${RESULTS_DIR}/generate_html.py" << 'EOFPYTHON'
import json
import csv
from datetime import datetime

def generate_html(results_dir):
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Hypatia Test Results</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
            .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }
            h1 { color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }
            .summary { display: flex; gap: 20px; margin: 20px 0; }
            .card { flex: 1; padding: 20px; border-radius: 8px; text-align: center; }
            .pass { background: #4CAF50; color: white; }
            .fail { background: #f44336; color: white; }
            .info { background: #2196F3; color: white; }
            .card h2 { margin: 0; font-size: 48px; }
            .card p { margin: 10px 0 0 0; }
            table { width: 100%; border-collapse: collapse; margin: 20px 0; }
            th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
            th { background: #4CAF50; color: white; }
            tr:hover { background: #f5f5f5; }
            .status-pass { color: #4CAF50; font-weight: bold; }
            .status-fail { color: #f44336; font-weight: bold; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🧪 Hypatia Test Results</h1>
            <p>Generated: {timestamp}</p>
    """.format(timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    # Read unit test JSON
    try:
        with open(f'{results_dir}/unit_tests.json') as f:
            data = json.load(f)
            summary = data.get('summary', {})
            
            html += f"""
            <div class="summary">
                <div class="card pass">
                    <h2>{summary.get('passed', 0)}</h2>
                    <p>Tests Passed</p>
                </div>
                <div class="card fail">
                    <h2>{summary.get('failed', 0)}</h2>
                    <p>Tests Failed</p>
                </div>
                <div class="card info">
                    <h2>{summary.get('total', 0)}</h2>
                    <p>Total Tests</p>
                </div>
            </div>
            """
            
            # Test results table
            html += "<h2>Unit Test Results</h2><table><tr><th>Test Name</th><th>Category</th><th>Status</th><th>Duration (ms)</th></tr>"
            for result in data.get('results', []):
                status_class = 'status-pass' if result['status'] == 'PASS' else 'status-fail'
                html += f"""
                <tr>
                    <td>{result['test_name']}</td>
                    <td>{result['category']}</td>
                    <td class="{status_class}">{result['status']}</td>
                    <td>{result['duration_ms']:.2f}</td>
                </tr>
                """
            html += "</table>"
    except:
        pass
    
    # Benchmarks
    try:
        with open(f'{results_dir}/benchmarks.csv') as f:
            reader = csv.DictReader(f)
            html += "<h2>Performance Benchmarks</h2><table><tr><th>Name</th><th>P50 (ms)</th><th>P95 (ms)</th><th>P99 (ms)</th></tr>"
            for row in reader:
                html += f"""
                <tr>
                    <td>{row['Name']}</td>
                    <td>{row['P50 (ms)']}</td>
                    <td>{row['P95 (ms)']}</td>
                    <td>{row['P99 (ms)']}</td>
                </tr>
                """
            html += "</table>"
    except:
        pass
    
    html += """
        </div>
    </body>
    </html>
    """
    
    with open(f'{results_dir}/summary.html', 'w') as f:
        f.write(html)
    
    print(f"✅ HTML summary generated: {results_dir}/summary.html")

if __name__ == '__main__':
    import sys
    generate_html(sys.argv[1])
EOFPYTHON

    python3 "${RESULTS_DIR}/generate_html.py" "${RESULTS_DIR}"
    echo -e "  🌐 summary.html          - Visual test report"
    echo ""
    echo -e "${GREEN}Open in browser:${NC} file://$(pwd)/${RESULTS_DIR}/summary.html"
fi

echo ""
echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║                    🎉 TEST SUITE COMPLETED! 🎉                     ║"
echo "╚════════════════════════════════════════════════════════════════════╝"

exit 0