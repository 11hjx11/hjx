"""
多智能体系统评测脚本

用法:
    cd loudi_food_recommend/backend
    python eval/run_eval.py                    # 跑全部用例
    python eval/run_eval.py --category recommendation_cuisine  # 按类别跑
    python eval/run_eval.py --limit 5          # 只跑前5条
    python eval/run_eval.py --output result.json

指标:
    - intent_accuracy:    意图识别准确率
    - tool_coverage:      期望工具命中率
    - keyword_coverage:   期望关键词命中率
    - avg_latency_ms:     平均延迟
    - p95_latency_ms:     P95 延迟
    - total_tokens:       token 用量估算（从 trace 推算）
"""
import argparse
import json
import os
import sys
import time
import statistics
from typing import Dict, Any, List

# 把 backend 目录加入 sys.path，让 eval 脚本能 import agents
_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BACKEND_DIR)

from agents import MultiAgentSystem


def load_eval_set(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_single_case(agent: MultiAgentSystem, case: Dict[str, Any]) -> Dict[str, Any]:
    """跑单条评测用例，返回评测结果"""
    user_input = case["input"]
    expected_intent = case.get("expected_intent", "")
    expected_tools = set(case.get("expected_tools_any", []))
    expected_keywords = case.get("expected_keywords_any", [])

    t0 = time.perf_counter()
    error = None
    result = None
    try:
        result = agent.run(user_input)
    except Exception as e:
        error = str(e)
    latency_ms = (time.perf_counter() - t0) * 1000

    if error:
        return {
            "id": case["id"],
            "input": user_input,
            "category": case.get("category", ""),
            "pass": False,
            "error": error,
            "latency_ms": round(latency_ms, 1),
        }

    actual_intent = result.get("intent", "")
    actual_tools = set(result.get("tools_used", []))
    response = result.get("response", "")

    # 意图准确率
    intent_match = (actual_intent == expected_intent) if expected_intent else True

    # 工具命中率: expected_tools_any 是 "命中任一即可"
    if expected_tools:
        tool_hit = bool(actual_tools & expected_tools)
    else:
        tool_hit = True  # 无期望工具要求

    # 关键词命中率: expected_keywords_any 是 "命中任一即可"
    if expected_keywords:
        keyword_hit = any(kw in response for kw in expected_keywords)
    else:
        keyword_hit = True

    passed = intent_match and tool_hit and keyword_hit

    return {
        "id": case["id"],
        "input": user_input,
        "category": case.get("category", ""),
        "pass": passed,
        "intent_match": intent_match,
        "tool_hit": tool_hit,
        "keyword_hit": keyword_hit,
        "expected_intent": expected_intent,
        "actual_intent": actual_intent,
        "expected_tools": list(expected_tools),
        "actual_tools": list(actual_tools),
        "expected_keywords": expected_keywords,
        "response_preview": response[:150],
        "iterations": result.get("iterations", 0),
        "latency_ms": round(latency_ms, 1),
    }


def aggregate_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """聚合评测指标"""
    total = len(results)
    if total == 0:
        return {"total": 0}

    passed = sum(1 for r in results if r.get("pass"))
    intent_match_count = sum(1 for r in results if r.get("intent_match"))
    tool_hit_count = sum(1 for r in results if r.get("tool_hit"))
    keyword_hit_count = sum(1 for r in results if r.get("keyword_hit"))

    latencies = [r["latency_ms"] for r in results if "latency_ms" in r]
    latencies_sorted = sorted(latencies)
    p95_idx = int(len(latencies_sorted) * 0.95)

    # 按类别聚合
    by_category: Dict[str, Dict[str, Any]] = {}
    for r in results:
        cat = r.get("category", "unknown")
        if cat not in by_category:
            by_category[cat] = {"total": 0, "passed": 0}
        by_category[cat]["total"] += 1
        if r.get("pass"):
            by_category[cat]["passed"] += 1

    return {
        "total": total,
        "passed": passed,
        "overall_accuracy": round(passed / total, 4),
        "intent_accuracy": round(intent_match_count / total, 4),
        "tool_coverage": round(tool_hit_count / total, 4),
        "keyword_coverage": round(keyword_hit_count / total, 4),
        "avg_latency_ms": round(statistics.mean(latencies), 1) if latencies else 0,
        "p50_latency_ms": round(statistics.median(latencies), 1) if latencies else 0,
        "p95_latency_ms": round(latencies_sorted[p95_idx], 1) if latencies_sorted else 0,
        "by_category": {
            cat: {
                "total": v["total"],
                "passed": v["passed"],
                "accuracy": round(v["passed"] / v["total"], 4) if v["total"] else 0,
            }
            for cat, v in by_category.items()
        },
    }


def print_report(results: List[Dict[str, Any]], summary: Dict[str, Any]):
    """打印评测报告"""
    print("\n" + "=" * 70)
    print("  多智能体系统评测报告")
    print("=" * 70)

    print(f"\n总用例数: {summary['total']}")
    print(f"通过数:   {summary['passed']}")
    print(f"整体准确率: {summary['overall_accuracy']:.1%}")
    print(f"意图准确率: {summary['intent_accuracy']:.1%}")
    print(f"工具覆盖率: {summary['tool_coverage']:.1%}")
    print(f"关键词覆盖: {summary['keyword_coverage']:.1%}")
    print(f"\n延迟统计:")
    print(f"  平均: {summary['avg_latency_ms']:.0f} ms")
    print(f"  P50:  {summary['p50_latency_ms']:.0f} ms")
    print(f"  P95:  {summary['p95_latency_ms']:.0f} ms")

    print(f"\n按类别:")
    for cat, v in summary["by_category"].items():
        status = "PASS" if v["accuracy"] == 1.0 else ("PARTIAL" if v["accuracy"] > 0 else "FAIL")
        print(f"  [{status}] {cat}: {v['passed']}/{v['total']} ({v['accuracy']:.0%})")

    # 失败用例明细
    failures = [r for r in results if not r.get("pass")]
    if failures:
        print(f"\n失败用例 ({len(failures)}):")
        for r in failures:
            print(f"  [{r['id']}] {r['input'][:40]}")
            if not r.get("intent_match"):
                print(f"    intent: expected={r.get('expected_intent')} actual={r.get('actual_intent')}")
            if not r.get("tool_hit"):
                print(f"    tools:  expected_any={r.get('expected_tools')} actual={r.get('actual_tools')}")
            if not r.get("keyword_hit"):
                print(f"    keywords: expected_any={r.get('expected_keywords')}")
                print(f"    response: {r.get('response_preview', '')[:80]}")
    else:
        print("\n全部通过!")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="多智能体系统评测")
    parser.add_argument("--eval-set", default=os.path.join(os.path.dirname(__file__), "eval_set.json"),
                        help="评测集路径")
    parser.add_argument("--category", default=None, help="只跑指定类别")
    parser.add_argument("--limit", type=int, default=None, help="限制用例数")
    parser.add_argument("--output", default=None, help="结果输出到 JSON 文件")
    parser.add_argument("--verbose", "-v", action="store_true", help="显示每条用例详情")
    args = parser.parse_args()

    eval_data = load_eval_set(args.eval_set)
    cases = eval_data["cases"]

    if args.category:
        cases = [c for c in cases if c.get("category") == args.category]
    if args.limit:
        cases = cases[: args.limit]

    print(f"加载评测集: {len(cases)} 条用例")

    agent = MultiAgentSystem()
    results = []
    for i, case in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] {case['id']}: {case['input'][:40]}...", end=" ", flush=True)
        r = run_single_case(agent, case)
        results.append(r)
        status = "PASS" if r.get("pass") else "FAIL"
        print(f"{status} ({r.get('latency_ms', 0):.0f}ms)")
        if args.verbose and not r.get("pass"):
            print(f"    intent: {r.get('actual_intent')}, tools: {r.get('actual_tools')}")

    summary = aggregate_results(results)
    print_report(results, summary)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "results": results}, f, ensure_ascii=False, indent=2)
        print(f"\n详细结果已写入: {args.output}")


if __name__ == "__main__":
    main()
