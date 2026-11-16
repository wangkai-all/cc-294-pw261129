"""cachequery 测试套件:解析、蕴含、执行正确性、复用规则、失效传播、并发、恢复。"""
import tempfile
import unittest

from cachequery.querylang import Cond, implies, match_cond, parse, ParseError
from cachequery.cache import CacheStore, CrashSimulated
from cachequery.data import Dataset
from cachequery.executor import Engine
from cachequery import samples


def fresh(**kw):
    return samples.new_engine(**kw)


Q_RANGE = ("SELECT sum(amount), avg(amount), count_distinct(user_id) FROM events "
           "WHERE date >= '2026-01-01' AND date < '2026-01-05' GROUP BY country")


# ---------------------------------------------------------------- 解析
class TestParser(unittest.TestCase):
    def test_basic(self):
        q = parse(Q_RANGE)
        self.assertEqual(q.table, "events")
        self.assertEqual([m.kind for m in q.measures], ["avg", "count_distinct", "sum"])
        self.assertEqual(q.group_by, ("country",))

    def test_normalization_commutative(self):
        a = parse("SELECT sum(amount) FROM events WHERE country='CN' AND channel='web'")
        b = parse("SELECT sum(amount) FROM events WHERE channel='web' AND country='CN'")
        self.assertEqual(a.norm(), b.norm())

    def test_literal_left_flipped(self):
        a = parse("SELECT sum(amount) FROM events WHERE 'CN' = country")
        b = parse("SELECT sum(amount) FROM events WHERE country = 'CN'")
        self.assertEqual(a.norm(), b.norm())

    def test_in_list_sorted(self):
        a = parse("SELECT sum(amount) FROM events WHERE channel IN ('web','app')")
        b = parse("SELECT sum(amount) FROM events WHERE channel IN ('app','web')")
        self.assertEqual(a.norm(), b.norm())

    def test_is_null(self):
        q = parse("SELECT avg(amount) FROM events WHERE country IS NULL")
        self.assertIn(Cond("country", "IS NULL"), q.conds)

    def test_errors(self):
        for bad in ["SELECT * FROM events",
                    "SELECT sum(amount) FROM events WHERE country = 'CN' OR channel='web'",
                    "SELECT sum(amount) FROM events WHERE amount = NULL",
                    "DELETE FROM events"]:
            with self.assertRaises(ParseError, msg=bad):
                parse(bad)


# ---------------------------------------------------------------- 蕴含证明
class TestImplication(unittest.TestCase):
    def test_range_subset(self):
        f2 = parse("SELECT sum(amount) FROM events WHERE date >= '2026-01-05' "
                   "AND date < '2026-01-10'").conds
        f1 = parse("SELECT sum(amount) FROM events WHERE date >= '2026-01-01'").conds
        ok, proofs = implies(f2, f1)
        self.assertTrue(ok)
        self.assertTrue(proofs)

    def test_range_not_subset(self):
        f2 = parse("SELECT sum(amount) FROM events WHERE date >= '2026-01-01'").conds
        f1 = parse("SELECT sum(amount) FROM events WHERE date >= '2026-01-05'").conds
        ok, _ = implies(f2, f1)
        self.assertFalse(ok)

    def test_in_subset(self):
        f2 = [Cond("channel", "IN", ("web",))]
        f1 = [Cond("channel", "IN", ("web", "app"))]
        self.assertTrue(implies(f2, f1)[0])
        self.assertFalse(implies(f1, f2)[0])

    def test_is_null(self):
        self.assertTrue(implies([Cond("country", "IS NULL")],
                                [Cond("country", "IS NULL")])[0])
        self.assertFalse(implies([Cond("country", "=", "CN")],
                                 [Cond("country", "IS NULL")])[0])

    def test_unprovable_is_false(self):
        # 无法证明时一律 False(调用方真实重算),绝不猜测
        ok, _ = implies([Cond("amount", "!=", 5)], [Cond("amount", "<", 100)])
        self.assertFalse(ok)


# ---------------------------------------------------------------- 执行正确性
class TestExecutorVsBaseline(unittest.TestCase):
    QUERIES = [
        "SELECT sum(amount) FROM events",
        "SELECT avg(amount) FROM events WHERE country = 'CN'",
        "SELECT count_distinct(user_id) FROM events WHERE channel IN ('web','app')",
        "SELECT sum(amount), avg(amount) FROM events "
        "WHERE date >= '2026-01-03' AND date < '2026-01-08' GROUP BY country, channel",
        "SELECT count_distinct(user_id) FROM events WHERE country IS NULL",
        "SELECT sum(amount) FROM events WHERE amount IS NOT NULL GROUP BY channel",
        "SELECT avg(amount) FROM events WHERE country != 'JP' AND date < '2026-01-04'",
    ]

    def test_cached_equals_baseline(self):
        eng = fresh()
        for q in self.QUERIES:
            for user in ("alice", "bob"):
                eng.run(q, user=user)                       # 冷
                r = eng.run(q, user=user)                   # 热
                b = eng.run(q, user=user, use_cache=False)
                self.assertEqual(r["rows"], b["rows"], msg=f"{user}: {q}")

    def test_rollup_from_finer_groupby(self):
        eng = fresh()
        eng.run("SELECT sum(amount) FROM events WHERE date >= '2026-01-01' "
                "AND date < '2026-01-04' GROUP BY country, channel")
        r = eng.run("SELECT sum(amount) FROM events WHERE date >= '2026-01-01' "
                    "AND date < '2026-01-04' GROUP BY country")
        self.assertTrue(any("上卷" in p for p in r["provenance"]
                            for p in p["proof"]))
        b = eng.run("SELECT sum(amount) FROM events WHERE date >= '2026-01-01' "
                    "AND date < '2026-01-04' GROUP BY country", use_cache=False)
        self.assertEqual(r["rows"], b["rows"])

    def test_l3_containment_rejected(self):
        eng = fresh()
        eng.run("SELECT sum(amount) FROM events WHERE date >= '2026-01-01' "
                "AND date < '2026-02-01'")
        r = eng.run("SELECT sum(amount) FROM events WHERE date >= '2026-01-05' "
                    "AND date < '2026-01-10'")
        self.assertFalse(any(p["source"] == "L3" for p in r["provenance"]))
        self.assertTrue(any("范围包含" in x["reason"] for x in r["rejected"]))
        self.assertTrue(all(p["source"] == "L2" for p in r["provenance"]))

    def test_null_semantics(self):
        eng = fresh()
        # avg 忽略 NULL;NULL 国家自成一组
        r = eng.run("SELECT avg(amount) FROM events GROUP BY country")
        b = eng.run("SELECT avg(amount) FROM events GROUP BY country", use_cache=False)
        self.assertEqual(r["rows"], b["rows"])
        countries = [row[0] for row in b["rows"]]
        self.assertIn(None, countries)          # 数据含 NULL country


# ---------------------------------------------------------------- 失效传播
class TestInvalidation(unittest.TestCase):
    def test_data_correction_partial(self):
        eng = fresh()
        q = ("SELECT sum(amount) FROM events WHERE date >= '2026-01-01' "
             "AND date < '2026-01-05' GROUP BY country")
        eng.run(q)
        out = eng.correct_data("2026-01-02", "restated_amount")
        self.assertGreater(len(out["invalidated"]), 0)
        r = eng.run(q)
        raw_days = {p["day"] for p in r["provenance"] if p["source"] == "RAW"}
        l2_days = {p["day"] for p in r["provenance"] if p["source"] == "L2"}
        self.assertEqual(raw_days, {"2026-01-02"})
        self.assertEqual(l2_days, {"2026-01-01", "2026-01-03", "2026-01-04"})
        b = eng.run(q, use_cache=False)
        self.assertEqual(r["rows"], b["rows"])

    def test_l3_cascade(self):
        eng = fresh()
        eng.run(Q_RANGE)
        n_l3_before = len(eng.store.find("L3"))
        self.assertGreater(n_l3_before, 0)
        eng.correct_data("2026-01-01", "fix_null_amount")
        # 覆盖 2026-01-01 的 L3 必须被级联失效
        for e in eng.store.list_entries():
            if e["level"] == "L3":
                self.assertNotEqual(e["resources"].get("data:2026-01-01"), 1)

    def test_permission_narrow_no_aggregate_reuse(self):
        eng = fresh()
        q = "SELECT sum(amount) FROM events WHERE date < '2026-01-03' GROUP BY country"
        eng.run(q, user="alice")                # scope ALL → 建立 L2/L3
        eng.narrow_permission("alice", ["CN"])
        r = eng.run(q, user="alice")
        # 聚合结果(L2/L3)不得从更宽范围复用
        self.assertFalse(any(p["source"] in ("L2", "L3") for p in r["provenance"]))
        b = eng.run(q, user="alice", use_cache=False)
        self.assertEqual(r["rows"], b["rows"])
        countries = {row[0] for row in r["rows"]}
        self.assertLessEqual(countries, {"CN", None})

    def test_l1_shard_reusable_under_narrowing(self):
        eng = fresh()
        q = "SELECT sum(amount) FROM events WHERE date < '2026-01-03'"
        eng.run(q, user="alice")                # ALL 范围 → L1 分片(无 country 条件)
        eng.narrow_permission("alice", ["CN", "JP"])
        r = eng.run(q, user="alice")
        # 原始分片可在包含证明下复用(再过滤),聚合不可
        self.assertTrue(any(p["source"] == "L1" for p in r["provenance"]))
        b = eng.run(q, user="alice", use_cache=False)
        self.assertEqual(r["rows"], b["rows"])

    def test_restrict_country_evicts(self):
        eng = fresh()
        eng.run("SELECT sum(amount) FROM events WHERE date < '2026-01-03'")
        out = eng.restrict_country("JP")
        self.assertGreater(len(out["invalidated"]), 0)
        # bob 的可见范围 (CN,JP) 含被限制的 JP → 拒绝
        with self.assertRaises(PermissionError):
            eng.run("SELECT sum(amount) FROM events", user="bob")
        # alice(ALL)有效范围自动排除 JP,且结果与基线一致
        r = eng.run("SELECT sum(amount) FROM events GROUP BY country", user="alice")
        b = eng.run("SELECT sum(amount) FROM events GROUP BY country",
                    user="alice", use_cache=False)
        self.assertEqual(r["rows"], b["rows"])
        self.assertNotIn("JP", {row[0] for row in r["rows"]})

    def test_semantics_upgrade_keeps_l1(self):
        eng = fresh()
        eng.run("SELECT sum(amount) FROM events WHERE date < '2026-01-03'")
        eng.upgrade_semantics(2)
        levels = {e["level"] for e in eng.store.list_entries()}
        self.assertIn("L1", levels)             # 原始分片与口径无关 → 保留
        self.assertNotIn("L2", levels)
        self.assertNotIn("L3", levels)

    def test_same_shape_different_meaning(self):
        eng = fresh()
        q = "SELECT sum(amount) FROM events WHERE date < '2026-01-03'"
        r1 = eng.run(q)
        eng.upgrade_semantics(2)
        r2 = eng.run(q)
        self.assertNotEqual(r1["rows"], r2["rows"])   # 同形不同含义
        self.assertFalse(any(p["source"] == "L3" for p in r2["provenance"]))
        b2 = eng.run(q, use_cache=False)
        self.assertEqual(r2["rows"], b2["rows"])


# ---------------------------------------------------------------- 并发
class TestConcurrency(unittest.TestCase):
    def test_concurrent_queries_and_invalidation(self):
        out = samples.concurrency_check(fresh())
        self.assertEqual(out["errors"], [])
        self.assertGreater(out["assertions"], 0)
        for k, v in out["check"].items():
            self.assertTrue(v, msg=k)


# ---------------------------------------------------------------- 中断恢复
class TestRecovery(unittest.TestCase):
    def test_crash_then_recover(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = samples.recovery_check(tmp)
            for k, v in out["check"].items():
                self.assertTrue(v, msg=k)

    def test_building_entry_invisible_before_commit(self):
        eng = fresh()
        store = eng.store
        store.crash_after_building = True
        with self.assertRaises(CrashSimulated):
            eng.run("SELECT sum(amount) FROM events WHERE date < '2026-01-02'")
        # building 条目对查询不可见,recover 后被清除
        self.assertTrue(all(e["status"] != "building" or not e["valid"]
                            for e in store.list_entries()))
        recovered = store.recover()
        self.assertGreaterEqual(len(recovered), 1)
        r = eng.run("SELECT sum(amount) FROM events WHERE date < '2026-01-02'")
        b = eng.run("SELECT sum(amount) FROM events WHERE date < '2026-01-02'",
                    use_cache=False)
        self.assertEqual(r["rows"], b["rows"])


# ---------------------------------------------------------------- 基准
class TestBenchmark(unittest.TestCase):
    def test_benchmark_all_equal(self):
        out = samples.benchmark(fresh())
        self.assertTrue(out["summary"]["全部一致"])
        self.assertGreater(out["summary"]["基线扫描行"],
                           out["summary"]["热遍扫描行"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
