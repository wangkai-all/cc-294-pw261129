"""命令行演示:依次跑四个样例 + 基准对比 + 并发检查 + 中断恢复。

用法: python -m cachequery.demo
"""
from __future__ import annotations

import json
import tempfile

from . import samples


def _show_check(name: str, checks: dict) -> bool:
    ok_all = True
    print(f"\n== {name} ==")
    for k, v in checks.items():
        print(f"  [{'OK' if v else 'FAIL'}] {k}")
        ok_all &= bool(v)
    return ok_all


def main() -> int:
    ok = True
    eng = samples.new_engine()

    ok &= _show_check(*_tc(samples.sample_same_shape(eng)))
    ok &= _show_check(*_tc(samples.sample_nulls(eng)))
    ok &= _show_check(*_tc(samples.sample_cross_version(eng)))
    ok &= _show_check(*_tc(samples.sample_containment(eng)))

    bench = samples.benchmark(eng)
    s = bench["summary"]
    print(f"\n== {bench['title']} ==")
    print(f"  结果全部一致: {s['全部一致']}")
    print(f"  冷遍缓存 {s['冷遍缓存_ms']} ms / 热遍缓存 {s['热遍缓存_ms']} ms / 基线 {s['基线总_ms']} ms")
    print(f"  冷遍扫描 {s['冷遍扫描行']} 行 / 热遍扫描 {s['热遍扫描行']} 行 / 基线 {s['基线扫描行']} 行")
    for lv, st in s["层次统计"].items() if isinstance(s["层次统计"], dict) else []:
        if isinstance(st, dict):
            print(f"  {lv}: {st['entries']} 条, {st['bytes']} B, 命中 {st['hits']}, 构建 {st['builds']}")
    ok &= s["全部一致"]

    ok &= _show_check(*_tc(samples.concurrency_check(samples.new_engine())))
    with tempfile.TemporaryDirectory() as tmp:
        ok &= _show_check(*_tc(samples.recovery_check(tmp)))

    print("\n总体:", "全部通过" if ok else "存在失败")
    return 0 if ok else 1


def _tc(sample_out: dict):
    return sample_out["title"], sample_out["check"]


if __name__ == "__main__":
    raise SystemExit(main())
