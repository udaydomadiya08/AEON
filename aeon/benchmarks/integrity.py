"""
AEON Phase 9 — Benchmark Integrity Scanner.

Performs static AST analysis and regex auditing across production and benchmark
source trees to detect hardcoded outputs, shortcut evaluations, or benchmark-only
production branching.
"""

import ast
import os
import re
from typing import Dict, List, Any


FORBIDDEN_PATTERNS = [
    r"(?<!\d)10\.33(?!\d)",
    r"(?<!\d)10\.30(?!\d)",
    r"(?<!\d)10\.34(?!\d)",
    r"(?<!\d)3\.91(?!\d)",
    r"(?<!\d)8\.65(?!\d)",
    r"(?<!\d)6\.19(?!\d)",
    r"7\.23e-6",
    r"(?<!\d)106\.07(?!\d)",
    r"(?<!\d)95\.8(?!\d)"
]


class BenchmarkIntegrityScanner:
    """Scans Python files for forbidden evaluation shortcuts and hardcoded artifacts."""

    def __init__(self, target_dir: str = "aeon"):
        self.target_dir = target_dir

    def scan_for_hardcoded_constants(self) -> List[Dict[str, Any]]:
        """Search for historical benchmark constants embedded in production or benchmark code."""
        findings = []
        for root, _, files in os.walk(self.target_dir):
            for file in files:
                if not file.endswith(".py"):
                    continue
                path = os.path.join(root, file)
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    for line_no, line in enumerate(f, start=1):
                        for pat in FORBIDDEN_PATTERNS:
                            if re.search(pat, line):
                                findings.append({
                                    "file": path,
                                    "line": line_no,
                                    "pattern": pat,
                                    "content": line.strip()
                                })
        return findings

    def scan_for_ast_shortcuts(self) -> List[Dict[str, Any]]:
        """
        Scan AST for blatant evaluation shortcuts outside test/evaluator scopes.
        Detects statements like 'correct = True' in non-evaluator files.
        """
        findings = []
        for root, _, files in os.walk(self.target_dir):
            for file in files:
                if not file.endswith(".py"):
                    continue
                path = os.path.join(root, file)
                # Allow ground truth evaluators to assign boolean correctness based on checks
                if "evaluator" in file or "ground_truth" in file or "workloads" in file:
                    continue
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        tree = ast.parse(f.read(), filename=path)
                except Exception:
                    continue

                for node in ast.walk(tree):
                    if isinstance(node, ast.Assign):
                        for target in node.targets:
                            if isinstance(target, ast.Name) and target.id == "correct":
                                if isinstance(node.value, ast.Constant) and node.value.value is True:
                                    findings.append({
                                        "file": path,
                                        "line": getattr(node, "lineno", 0),
                                        "type": "hardcoded_correct_assignment",
                                        "description": "Blatant 'correct = True' assignment found in production code!"
                                    })
        return findings

    def run_full_scan(self) -> Dict[str, Any]:
        """Execute full integrity scan and return audit report."""
        const_findings = self.scan_for_hardcoded_constants()
        ast_findings = self.scan_for_ast_shortcuts()
        passed = (len(const_findings) == 0) and (len(ast_findings) == 0)
        return {
            "passed": passed,
            "hardcoded_constants_count": len(const_findings),
            "ast_shortcuts_count": len(ast_findings),
            "findings": const_findings + ast_findings
        }
