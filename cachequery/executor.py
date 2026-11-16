"""执行器:三层缓存的构建与复用、复用证明、并发一致性、无缓存基线。

复用规则(核心设计,也是本项目的「证明条件」):
  L3 最终结果   :仅当规范化查询完全相同 + 权限范围相同 + 口径/数据版本一致。
                 范围被包含 ≠ 可复用 —— sum/avg/count_distinct 都无法从超集结果中扣除。
  L2 可组合状态 :同一分区、筛选完全一致、权限范围一致、版本一致;
                 分组维度更细的缓存可上卷(投影)到更粗分组;
                 多个分区的状态可合并(sum 相加、avg 的 (和,计数) 相加、去重集合求并)。
  L1 原始分片   :同一分区、数据版本一致,且能证明「查询行集 ⊆ 分片行集」
                 (筛选蕴含 + 权限范围蕴含)时,取分片重新过滤后使用。
                 原始分片可以在包含关系下复用 —— 聚合结果不可以。

并发一致性:计算前读 epoch,组装后在同一把锁上校验所有用到的条目与资源版本,
有任何失效发生在中途则丢弃重来(有界重试,最终在全局锁下完成,保证终止)。
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from .querylang import (Cond, Query, implies, match_all, parse, select_days)
from .cache import CacheStore, Entry, CrashSimulated

# ---------------------------------------------------------------- 口径(语义)版本
# 同形不同含义:同一条 SQL 在不同口径版本下含义不同,缓存必须隔离。
SEMANTICS = {
    1: {"desc": "v1 毛额口径:amount 取原始值;去重统计全部用户"},
    2: {"desc": "v2 净额口径:退款行不计入金额;去重仅统计活跃用户"},
}


def _sem_amount(ver: int, row: dict):
    v = row.get("amount")
    if ver >= 2 and row.get("refunded"):
        return None          # v2:退款行不计入金额
    return v


def _sem_uid_ok(ver: int, row: dict) -> bool:
    if ver >= 2 and not row.get("active"):
        return False         # v2:去重仅统计活跃用户
    return True


# ---------------------------------------------------------------- 上下文

@dataclass(frozen=True)
class Ctx:
    user: str
    scope: tuple | None        # None = 全部国家可见

    def scope_fp(self) -> str:
        return "ALL" if self.scope is None else ",".join(sorted(self.scope))

    def scope_conds(self) -> list:
        if self.scope is None:
            return []
        return [Cond("country", "IN", tuple(sorted(self.scope)))]


# ---------------------------------------------------------------- 聚合状态
# 金额一律以「分」(整数)累计,保证分片合并与一次性全量计算结果逐位相等。

def _empty_state(kind: str):
    if kind == "sum":
        return 0                      # 分
    if kind == "avg":
        return [0, 0]                 # [分和, 行数]
    if kind == "count_distinct":
        return set()
    raise ValueError(kind)


def _state_add(kind: str, state, row: dict, sem_ver: int):
    if kind in ("sum", "avg"):
        v = _sem_amount(sem_ver, row)
        if v is None:
            return state
        cents = int(round(v * 100))
        if kind == "sum":
            return state + cents
        state[0] += cents
        state[1] += 1
        return state
    if kind == "count_distinct":
        uid = row.get("user_id")
        if uid is not None and _sem_uid_ok(sem_ver, row):
            state.add(uid)
        return state
    raise ValueError(kind)


def _state_merge(kind: str, a, b):
    if kind == "sum":
        return a + b
    if kind == "avg":
        return [a[0] + b[0], a[1] + b[1]]
    if kind == "count_distinct":
        return a | b
    raise ValueError(kind)


def _state_finalize(kind: str, state):
    if kind == "sum":
        return round(state / 100, 2)
    if kind == "avg":
        return None if state[1] == 0 else round(state[0] / 100 / state[1], 4)
    if kind == "count_distinct":
        return len(state)
    raise ValueError(kind)


def _state_to_json(kind: str, state):
    if kind == "count_distinct":
        return sorted(state)
    return list(state) if kind == "avg" else state


def _state_from_json(kind: str, data):
    if kind == "count_distinct":
        return set(data)
    if kind == "avg":
        return [data[0], data[1]]
    return data


# ---------------------------------------------------------------- 执行器

class Engine:
    def __init__(self, data, store: CacheStore, semantics_version: int = 1):
        self.data = data
        self.store = store
        self.semantics_version = semantics_version
        self.permissions = {"alice": None, "bob": ("CN", "JP")}
        self.restricted: set[str] = set()
        self.perm_epoch = 0
        # 对齐资源版本注册表(并清掉重启期间已失效的条目)
        self.store.invalidate_resource("semantics", semantics_version, "引擎启动:口径版本对齐")
        self.store.invalidate_resource("perm", self.perm_epoch, "引擎启动:权限纪元对齐")
        for d in self.data.days():
            self.store.invalidate_resource(f"data:{d}", self.data.version(d),
                                           "引擎启动:数据版本对齐")

    # ---------------- 上下文 ----------------
    def _ctx(self, user: str) -> Ctx:
        scope = self.permissions.get(user)
        if scope is None:
            # ALL 范围用户同样受全局限制约束:有效范围 = 全部国家 - 被限制国家
            if self.restricted:
                from .data import COUNTRIES
                scope = tuple(c for c in COUNTRIES if c not in self.restricted)
        else:
            scope = tuple(scope)
            bad = set(scope) & self.restricted
            if bad:
                raise PermissionError(f"用户 {user} 的可见范围包含已被限制的国家: {sorted(bad)}")
        return Ctx(user=user, scope=scope)

    # ---------------- 对外:失效源 ----------------
    def correct_data(self, day: str, mode: str) -> dict:
        from .data import preset_correction
        ver = preset_correction(self.data, day, mode)
        removed = self.store.invalidate_resource(f"data:{day}", ver,
                                                 f"数据更正({mode})")
        return {"day": day, "new_version": ver, "invalidated": removed}

    def narrow_permission(self, user: str, countries) -> dict:
        """收窄某用户可见范围。新查询的 scope 指纹变化 → L2/L3 自然失配重算;
        L1 若满足包含证明仍可复用(再过滤),这正是分层的意义。"""
        self.permissions[user] = tuple(sorted(countries))
        return {"user": user, "scope": sorted(countries)}

    def restrict_country(self, country: str) -> dict:
        """全局权限策略收紧:某国家被限制。含该国家的缓存条目必须主动失效。"""
        self.restricted.add(country)
        self.perm_epoch += 1
        self.store.sync_resource("perm", self.perm_epoch)
        removed = self.store.invalidate_where(
            lambda e: e.key.get("scope") in ("ALL",) or
                      country in str(e.key.get("scope", "")).split(",") or
                      country in str(e.key.get("filter", "")),
            f"权限策略收紧:限制 {country}")
        return {"restricted": country, "perm_epoch": self.perm_epoch,
                "invalidated": removed}

    def upgrade_semantics(self, version: int) -> dict:
        """口径升级:L2/L3 记录口径版本 → 全部失效;L1 原始分片与口径无关 → 保留。"""
        if version not in SEMANTICS:
            raise ValueError(f"未知口径版本 {version}")
        self.semantics_version = version
        removed = self.store.invalidate_resource("semantics", version, "口径升级")
        return {"semantics_version": version, "invalidated": removed,
                "desc": SEMANTICS[version]["desc"]}

    # ---------------- 聚合原语 ----------------
    def _group_key(self, q: Query, row: dict) -> tuple:
        return tuple(row.get(c) for c in q.group_by)

    def _add_row(self, states: dict, gkey: tuple, measures, row: dict):
        gs = states.get(gkey)
        if gs is None:
            gs = states[gkey] = {}
        for m in measures:
            k = m.norm()
            st = gs.get(k)
            if st is None:
                st = _empty_state(m.kind)
            gs[k] = _state_add(m.kind, st, row, self.semantics_version)

    def _merge_group_states(self, into: dict, src: dict):
        for gkey, mstates in src.items():
            gs = into.get(gkey)
            if gs is None:
                into[gkey] = {k: v for k, v in mstates.items()}
                continue
            for k, v in mstates.items():
                kind = k.split("(")[0]
                gs[k] = _state_merge(kind, gs[k], v) if k in gs else v

    def _finalize(self, q: Query, states: dict):
        columns = list(q.group_by) + [m.norm() for m in q.measures]
        rows = []
        for gkey in sorted(states, key=lambda k: tuple((v is None, str(v)) for v in k)):
            row = list(gkey)
            for m in q.measures:
                st = states[gkey].get(m.norm())
                row.append(None if st is None else _state_finalize(m.kind, st))
            rows.append(row)
        return {"columns": columns, "rows": rows}

    # ---------------- 无缓存基线 ----------------
    def baseline(self, q: Query, ctx: Ctx) -> dict:
        t0 = time.perf_counter()
        needed = list(q.conds) + ctx.scope_conds()
        states: dict = {}
        scanned = 0
        for d in select_days(q.conds, self.data.days()):
            rows, _ = self.data.snapshot(d)
            scanned += len(rows)
            for r in rows:
                if match_all(r, needed):
                    self._add_row(states, self._group_key(q, r), q.measures, r)
        out = self._finalize(q, states)
        out["rows_scanned"] = scanned
        out["elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 3)
        return out

    # ---------------- 带缓存执行 ----------------
    def run(self, query_text: str, user: str = "alice", use_cache: bool = True) -> dict:
        q = parse(query_text)
        if q.table != "events":
            raise ValueError(f"未知表 {q.table!r}(本系统只有 events)")
        ctx = self._ctx(user)
        if not use_cache:
            out = self.baseline(q, ctx)
            out["provenance"] = [{"source": "RAW", "day": "*",
                                  "proof": ["缓存关闭:全量真实重算(无缓存基线)"]}]
            out["rejected"] = []
            return out
        last = None
        for _attempt in range(4):
            res = self._try_cached(q, ctx)
            if res is not None:
                return res
        # 兜底:在全局锁内一次性完成,保证终止
        with self.store.lock():
            res = self._try_cached(q, ctx)
            if res is not None:
                return res
            out = self.baseline(q, ctx)
            out["provenance"] = [{"source": "RAW", "day": "*",
                                  "proof": ["并发冲突重试耗尽,锁内真实重算"]}]
            out["rejected"] = []
            return out

    def _try_cached(self, q: Query, ctx: Ctx) -> dict | None:
        store = self.store
        t0 = time.perf_counter()
        epoch0 = store.snapshot_epoch()
        prov, rejected = [], []
        used_entries: list[Entry] = []
        new_entries: list[Entry] = []
        used_resources: dict[str, int] = {}
        scanned = 0
        scope_fp = ctx.scope_fp()
        # date 条件由分区裁剪(select_days)保证,不进入残差筛选;
        # 这样「只改时间范围」的查询可以复用同一批按日 L2/L1 条目。
        residual_conds = [c for c in q.conds if c.column != "date"]
        needed_conds = residual_conds + ctx.scope_conds()
        residual_norm = (" AND ".join(c.norm() for c in residual_conds)
                         if residual_conds else "TRUE")
        needed_norm = " AND ".join(c.norm() for c in
                                   sorted(needed_conds, key=lambda c: c.norm())) or "TRUE"
        sem_ver = self.semantics_version
        days = select_days(q.conds, self.data.days())

        # ---------- L3:最终结果,仅完全相同的查询可复用 ----------
        l3_key = {"q": q.norm(), "scope": scope_fp}
        for e in store.find("L3", **l3_key):
            if store.valid(e):
                store.stats["hits_L3"] += 1
                prov.append({"source": "L3", "entry": e.id, "day": "*",
                             "proof": ["规范化查询完全一致",
                                       f"权限范围一致: {scope_fp}",
                                       f"口径版本一致: semantics=v{sem_ver}",
                                       f"数据版本一致: 涉及 {len(e.resources) - 2} 个分区"]})
                out = {"columns": e.payload["columns"], "rows": e.payload["rows"],
                       "provenance": prov, "rejected": rejected,
                       "rows_scanned": 0,
                       "elapsed_ms": round((time.perf_counter() - t0) * 1000, 3)}
                return out
            rejected.append({"level": "L3", "entry": e.id,
                             "reason": "相同查询的缓存已失效(资源版本变化)→ 真实重算"})
        # 范围包含但不可复用的明确说明
        for e in store.find("L3", scope=scope_fp):
            if e.key.get("q") == q.norm():
                continue
            cached_conds = [Cond(*c) for c in e.key.get("conds", [])]
            if cached_conds and cached_conds != list(q.conds):
                ok, _ = implies(list(q.conds), cached_conds)
                if ok:
                    rejected.append({
                        "level": "L3", "entry": e.id,
                        "reason": "查询行集 ⊆ 该缓存结果的行集,但 sum/avg/count_distinct "
                                  "均无法从超集结果中扣除 → 范围包含不等于可复用,改走 L2 重组"})
                    break

        # ---------- 逐分区:L2 可组合状态 ----------
        day_states: dict[str, dict] = {}     # day -> 该日(已投影到请求分组)的状态
        missing: list[str] = []
        for day in days:
            hit = None
            cands = [e for e in store.find("L2", day=day, filter=residual_norm,
                                           scope=scope_fp) if store.valid(e)]
            req_gb, req_ms = set(q.group_by), {m.norm() for m in q.measures}
            usable = [e for e in cands
                      if req_gb <= set(e.key["group_by"])
                      and req_ms <= set(e.key["measures"])]
            if usable:
                usable.sort(key=lambda e: (len(e.key["group_by"]), len(e.key["measures"])))
                hit = usable[0]
            if hit is None:
                missing.append(day)
                continue
            store.stats["hits_L2"] += 1
            used_entries.append(hit)
            used_resources.update(hit.resources)
            cached_gb = tuple(hit.key["group_by"])
            idx = [cached_gb.index(c) for c in q.group_by]
            states: dict = {}
            for key_list, mstates in hit.payload["groups"]:
                gkey = tuple(key_list[i] for i in idx)
                gs = states.setdefault(gkey, {})
                for mk, sv in mstates.items():
                    kind = mk.split("(")[0]
                    st = _state_from_json(kind, sv)
                    gs[mk] = _state_merge(kind, gs[mk], st) if mk in gs else st
            day_states[day] = states
            proof = [f"筛选完全一致(不含日期,日期由分区保证): {residual_norm}",
                     f"权限范围一致: {scope_fp}",
                     f"数据版本一致: data:{day}=v{hit.resources.get(f'data:{day}')}",
                     f"口径版本一致: semantics=v{sem_ver}"]
            proof.append("分组完全一致" if cached_gb == q.group_by else
                         f"分组上卷: ({','.join(cached_gb)}) → ({','.join(q.group_by)})")
            prov.append({"source": "L2", "entry": hit.id, "day": day, "proof": proof})

        # ---------- 缺失分区:L1 原始分片或真实扫描 ----------
        for day in missing:
            rows = None
            for e in store.find("L1", day=day):
                if not store.valid(e):
                    continue
                cached_conds = [Cond(c[0], c[1], tuple(c[2]) if c[1] == "IN" else c[2])
                                for c in e.key["conds"]]
                ok, subproof = implies(needed_conds, cached_conds)
                if not ok:
                    continue
                store.stats["hits_L1"] += 1
                used_entries.append(e)
                used_resources.update(e.resources)
                rows = [r for r in e.payload["rows"] if match_all(r, needed_conds)]
                scanned += len(e.payload["rows"])
                prov.append({"source": "L1", "entry": e.id, "day": day,
                             "proof": [f"数据版本一致: data:{day}=v{e.resources.get(f'data:{day}')}",
                                       "行集包含证明: 查询筛选+权限 ⇒ 分片筛选",
                                       *subproof,
                                       "复用方式: 对分片重新过滤后聚合"
                                       "(原始分片可在包含关系下复用,聚合结果不可)"]})
                break
            dep_ids = []
            if rows is None:
                raw, ver = self.data.snapshot(day)
                scanned += len(raw)
                rows = [r for r in raw if match_all(r, needed_conds)]
                used_resources[f"data:{day}"] = ver
                prov.append({"source": "RAW", "day": day,
                             "proof": [f"无可用缓存,真实扫描分区 data:{day}=v{ver}",
                                       f"扫描 {len(raw)} 行,命中 {len(rows)} 行"]})
                # 构建 L1 分片(键内含权限过滤条件)
                l1 = Entry("L1",
                           {"day": day, "filter": needed_norm,
                            "conds": [[c.column, c.op,
                                       list(c.value) if c.op == "IN" else c.value]
                                      for c in needed_conds]},
                           {f"data:{day}": ver, "perm": self.perm_epoch},
                           [], {"rows": rows},
                           note=f"分区 {day} 原始分片")
                store.begin_build(l1)          # 可能抛 CrashSimulated
                new_entries.append(l1)
                dep_ids = [l1.id]
            else:
                dep_ids = [used_entries[-1].id]
            # 由(再过滤后的)行构建 L2 状态
            ver = used_resources.get(f"data:{day}", self.data.version(day))
            states: dict = {}
            for r in rows:
                self._add_row(states, self._group_key(q, r), q.measures, r)
            payload = {"groups": [[list(gk), {mk: _state_to_json(mk.split("(")[0], st)
                                              for mk, st in ms.items()}]
                                  for gk, ms in states.items()]}
            l2 = Entry("L2",
                       {"day": day, "filter": residual_norm, "scope": scope_fp,
                        "group_by": list(q.group_by),
                        "measures": [m.norm() for m in q.measures]},
                       {f"data:{day}": ver, "semantics": sem_ver,
                        "perm": self.perm_epoch},
                       dep_ids, payload,
                       note=f"分区 {day} 可组合状态")
            store.begin_build(l2)
            new_entries.append(l2)
            day_states[day] = states

        # ---------- 合并各分区状态 → 最终结果 ----------
        merged: dict = {}
        for day in days:
            self._merge_group_states(merged, day_states.get(day, {}))
        out = self._finalize(q, merged)

        # ---------- 一致性校验 + 原子提交 ----------
        with store.lock():
            ok = True
            if store.epoch != epoch0:
                for e in used_entries:
                    if not store.valid(e):
                        ok = False
                        break
                if ok:
                    for k, v in used_resources.items():
                        if store.resource_version(k) != v:
                            ok = False
                            break
            if ok:
                for e in new_entries:
                    if any(store.resource_version(k) != v
                           for k, v in e.resources.items()):
                        ok = False
                        break
            if not ok:
                for e in new_entries:        # 重试前清理半成品,避免 building 泄漏
                    store.discard(e.id)
                return None
            for e in new_entries:
                store.commit(e.id)
            # L3 登记:依赖所有用到的 L2
            l3 = Entry("L3", {**l3_key,
                              "conds": [[c.column, c.op,
                                         list(c.value) if c.op == "IN" else c.value]
                                        for c in q.conds]},
                       dict(used_resources, semantics=sem_ver, perm=self.perm_epoch),
                       [e.id for e in used_entries if e.level == "L2"] +
                       [e.id for e in new_entries if e.level == "L2"],
                       {"columns": out["columns"], "rows": out["rows"]},
                       note="最终结果")
            store.begin_build(l3)
            if any(store.resource_version(k) != v for k, v in l3.resources.items()):
                store.discard(l3.id)
                return None
            store.commit(l3.id)

        out["provenance"] = prov
        out["rejected"] = rejected
        out["rows_scanned"] = scanned
        out["elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 3)
        return out
