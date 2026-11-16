"""样例场景与基准对比。

四个必含样例:
  1. 同形不同含义 —— 同一条 SQL,口径 v1/v2 含义不同,缓存必须隔离
  2. 空值         —— NULL country/amount/user_id 的聚合语义
  3. 跨版本       —— 数据更正后部分分区失效,健康分区继续复用
  4. 范围包含误用 —— 查询范围被缓存结果包含,但聚合不可复用,只能走 L2 重组
另含:基准对比(缓存开/关逐查询对照)与并发失效压力测试。
"""
from __future__ import annotations

import threading

from .cache import CacheStore
from .data import Dataset
from .executor import Engine


def new_engine(persistent_path=None, **data_kw) -> Engine:
    data = Dataset(**data_kw) if data_kw else Dataset()
    store = CacheStore(persistent_path)
    return Engine(data, store)


# ---------------------------------------------------------------- 样例 1:同形不同含义
def sample_same_shape(engine: Engine) -> dict:
    q = ("SELECT sum(amount), count_distinct(user_id) FROM events "
         "WHERE date >= '2026-01-01' AND date < '2026-01-04' GROUP BY country")
    steps = []
    r1 = engine.run(q, user="alice")
    steps.append({"action": "口径 v1 下首次执行", "result": r1})
    up = engine.upgrade_semantics(2)
    steps.append({"action": f"口径升级 → v2({up['desc']}),失效 {len(up['invalidated'])} 条",
                  "invalidated": up["invalidated"]})
    r2 = engine.run(q, user="alice")
    steps.append({"action": "同一条 SQL 在口径 v2 下再执行", "result": r2})
    same = r1["rows"] == r2["rows"]
    return {"title": "同形不同含义:同一 SQL,口径 v1/v2 结果不同且缓存隔离",
            "steps": steps,
            "check": {"v1_v2结果应不同": not same,
                      "v2未复用v1条目": all(p["source"] != "L3" for p in r2["provenance"])}}


# ---------------------------------------------------------------- 样例 2:空值
def sample_nulls(engine: Engine) -> dict:
    qs = {
        "avg 忽略 NULL 金额":
            "SELECT avg(amount) FROM events WHERE date >= '2026-01-01' AND date < '2026-01-03'",
        "去重忽略 NULL 用户":
            "SELECT count_distinct(user_id) FROM events WHERE date >= '2026-01-01' AND date < '2026-01-03'",
        "NULL 国家自成一组":
            "SELECT sum(amount) FROM events WHERE date >= '2026-01-01' AND date < '2026-01-03' GROUP BY country",
        "IS NULL 筛选":
            "SELECT count_distinct(user_id) FROM events WHERE country IS NULL "
            "AND date >= '2026-01-01' AND date < '2026-01-03'",
    }
    steps = []
    checks = {}
    for name, q in qs.items():
        cached = engine.run(q, user="alice")
        base = engine.run(q, user="alice", use_cache=False)
        ok = cached["rows"] == base["rows"]
        checks[name] = ok
        steps.append({"action": name, "query": q, "result": cached,
                      "baseline_rows": base["rows"], "一致": ok})
    return {"title": "空值语义:avg/去重忽略 NULL,GROUP BY 保留 NULL 组,IS NULL 可筛选",
            "steps": steps, "check": checks}


# ---------------------------------------------------------------- 样例 3:跨版本(数据更正 → 部分失效)
def sample_cross_version(engine: Engine) -> dict:
    q = ("SELECT sum(amount) FROM events "
         "WHERE date >= '2026-01-01' AND date < '2026-01-06' GROUP BY channel")
    steps = []
    r1 = engine.run(q, user="alice")          # 建立 5 个分区的 L2 + L3
    steps.append({"action": "首次执行(建立缓存)", "result": r1})
    corr = engine.correct_data("2026-01-03", "restated_amount")
    steps.append({"action": f"更正 2026-01-03(web 金额 +10%),"
                            f"失效 {len(corr['invalidated'])} 条缓存",
                  "invalidated": corr["invalidated"]})
    r2 = engine.run(q, user="alice")
    reused = sorted({p["day"] for p in r2["provenance"] if p["source"] == "L2"})
    raw = sorted({p["day"] for p in r2["provenance"] if p["source"] == "RAW"})
    steps.append({"action": "更正后重跑同一查询", "result": r2,
                  "复用的分区": reused, "重算的分区": raw})
    base = engine.run(q, user="alice", use_cache=False)
    return {"title": "跨版本:单日数据更正 → 该日 L1/L2 与依赖它的 L3 失效,其余分区继续复用",
            "steps": steps,
            "check": {"更正日重算": raw == ["2026-01-03"],
                      "其余分区复用": set(reused) == {"2026-01-01", "2026-01-02",
                                                    "2026-01-04", "2026-01-05"},
                      "结果与基线一致": r2["rows"] == base["rows"]}}


# ---------------------------------------------------------------- 样例 4:范围包含 ≠ 可复用
def sample_containment(engine: Engine) -> dict:
    q_full = ("SELECT sum(amount), count_distinct(user_id) FROM events "
              "WHERE date >= '2026-01-01' AND date < '2026-02-01'")
    q_sub = ("SELECT sum(amount), count_distinct(user_id) FROM events "
             "WHERE date >= '2026-01-05' AND date < '2026-01-12'")
    steps = []
    engine.run(q_full, user="alice")                       # 缓存整月 L3
    steps.append({"action": "缓存整月查询的 L3 结果", "query": q_full})
    r = engine.run(q_sub, user="alice")                    # 子范围:不能用整月 L3
    steps.append({"action": "执行子范围查询(范围被整月包含)", "query": q_sub,
                  "result": r})
    base = engine.run(q_sub, user="alice", use_cache=False)
    has_reject = any("范围包含" in x["reason"] for x in r["rejected"])
    used_l2 = any(p["source"] == "L2" for p in r["provenance"])
    used_l3 = any(p["source"] == "L3" for p in r["provenance"])
    return {"title": "范围包含误用防护:子范围查询不套用超集结果,改由每日 L2 状态重组",
            "steps": steps,
            "check": {"明确拒绝L3复用": has_reject,
                      "未使用L3": not used_l3,
                      "改走L2重组": used_l2,
                      "结果与基线一致": r["rows"] == base["rows"]}}


# ---------------------------------------------------------------- 基准对比
def benchmark(engine: Engine, users=("alice", "bob")) -> dict:
    """典型分析负载:同一批查询反复改一个筛选/时间范围。缓存开/关对照。"""
    workload = []
    for i in range(1, 8):
        workload.append(
            ("alice",
             f"SELECT sum(amount), avg(amount) FROM events "
             f"WHERE date >= '2026-01-0{i}' AND date < '2026-01-0{i + 3}' "
             f"GROUP BY country"))
        workload.append(
            ("alice",
             f"SELECT count_distinct(user_id) FROM events "
             f"WHERE channel = 'web' AND date >= '2026-01-0{i}' "
             f"AND date < '2026-01-0{i + 2}'"))
        workload.append(
            ("bob",
             f"SELECT sum(amount) FROM events "
             f"WHERE date >= '2026-01-0{i}' AND date < '2026-01-0{i + 2}' "
             f"GROUP BY channel"))
    rows = []
    passes = {"cold": {"ms": 0.0, "scan": 0}, "warm": {"ms": 0.0, "scan": 0}}
    tot_b = 0.0
    scan_b = 0
    all_equal = True
    for pass_name in ("cold", "warm"):          # 冷/热两遍:热遍应大量命中
        for user, q in workload:
            rc = engine.run(q, user=user)                   # 缓存
            rb = engine.run(q, user=user, use_cache=False)  # 无缓存基线
            eq = rc["rows"] == rb["rows"]
            all_equal &= eq
            passes[pass_name]["ms"] += rc["elapsed_ms"]
            passes[pass_name]["scan"] += rc["rows_scanned"]
            if pass_name == "cold":
                tot_b += rb["elapsed_ms"]
                scan_b += rb["rows_scanned"]
                rows.append({"user": user, "query": q, "一致": eq,
                             "缓存_ms": rc["elapsed_ms"], "基线_ms": rb["elapsed_ms"],
                             "缓存扫描行": rc["rows_scanned"],
                             "基线扫描行": rb["rows_scanned"]})
    return {"title": "基准对比:缓存 vs 无缓存基线",
            "rows": rows,
            "summary": {"全部一致": all_equal,
                        "冷遍缓存_ms": round(passes["cold"]["ms"], 2),
                        "热遍缓存_ms": round(passes["warm"]["ms"], 2),
                        "基线总_ms": round(tot_b, 2),
                        "冷遍扫描行": passes["cold"]["scan"],
                        "热遍扫描行": passes["warm"]["scan"],
                        "基线扫描行": scan_b,
                        "层次统计": engine.store.level_stats()}}


# ---------------------------------------------------------------- 并发失效压力
def concurrency_check(engine: Engine, n_threads: int = 6, n_corrections: int = 40) -> dict:
    """多线程查询与数据更正并发。

    一致性断言(epoch 夹逼):若一次「带缓存执行 + 基线执行」窗口内
    失效纪元没有变化,则两次结果必须逐行相等;窗口内发生过失效则该次
    不计(无法钉住版本)。统计有效断言次数,确保检查不是空转。
    """
    q = ("SELECT sum(amount) FROM events "
         "WHERE date >= '2026-01-02' AND date < '2026-01-05' GROUP BY country")
    errors = []
    checked = [0]
    stop = threading.Event()

    def worker():
        while not stop.is_set():
            try:
                e0 = engine.store.snapshot_epoch()
                r = engine.run(q, user="alice")
                b = engine.run(q, user="alice", use_cache=False)
                e1 = engine.store.snapshot_epoch()
                if e0 == e1:
                    checked[0] += 1
                    if r["rows"] != b["rows"]:
                        errors.append("epoch 未变但缓存结果 ≠ 基线")
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{type(exc).__name__}: {exc}")

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for _ in range(n_corrections):
        engine.correct_data("2026-01-03", "restated_amount")
    stop.set()
    for t in threads:
        t.join()
    # 静默后:缓存结果必须等于基线,且仓库中无失效条目残留
    r = engine.run(q, user="alice")
    b = engine.run(q, user="alice", use_cache=False)
    stale = [e["id"] for e in engine.store.list_entries() if not e["valid"]]
    return {"title": "并发查询 + 并发失效:不漏失效、结果始终一致",
            "check": {"并发期间无错误": not errors,
                      "有效断言次数>0": checked[0] > 0,
                      "静默后结果一致": r["rows"] == b["rows"],
                      "无失效条目残留": not stale},
            "errors": errors[:5], "stale": stale[:5],
            "assertions": checked[0]}


# ---------------------------------------------------------------- 中断恢复
def recovery_check(path: str) -> dict:
    """缓存生成中断(building 半成品落盘)→ 重启恢复 → 结果仍正确。"""
    eng = new_engine(persistent_path=path)
    q = ("SELECT sum(amount) FROM events "
         "WHERE date >= '2026-01-01' AND date < '2026-01-04' GROUP BY country")
    eng.store.crash_after_building = True
    crashed = False
    try:
        eng.run(q, user="alice")
    except Exception as exc:  # CrashSimulated
        crashed = type(exc).__name__ == "CrashSimulated"
    # 模拟重启:同一目录新建 Store/Engine
    eng2 = new_engine(persistent_path=path)
    recovered = eng2.store.recover()
    r = eng2.run(q, user="alice")
    b = eng2.run(q, user="alice", use_cache=False)
    building_left = [e for e in eng2.store.list_entries() if e["status"] == "building"]
    return {"title": "缓存生成中断后的恢复",
            "check": {"确实发生了中断": crashed,
                      "恢复时丢弃半成品": len(recovered) >= 1,
                      "恢复后结果与基线一致": r["rows"] == b["rows"],
                      "无 building 残留": not building_left},
            "recovered": recovered}
