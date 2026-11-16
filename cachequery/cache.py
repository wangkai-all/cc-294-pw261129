"""缓存存储:条目、依赖登记、失效传播、持久化与中断恢复。

三个缓存层次(在 executor 中实现构建/复用逻辑,这里只负责存储与一致性):
  L1 原始分片   —— 某分区某筛选下的明细行,空间最大,可在「包含」证明下再过滤复用
  L2 可组合状态 —— 某分区某筛选某分组下的 (sum, count, 去重集合),可跨分区合并、可上卷
  L3 最终结果   —— 整个查询的最终结果,仅完全相同查询可复用

一致性模型:
  - 每个条目记录其依赖的资源版本: data:<day> / semantics / perm
  - 每个条目登记其依赖的上游条目 id(deps),失效沿依赖边级联
  - 所有读写与失效都在同一把锁下完成;epoch 单调递增,
    执行器在计算前后比对 epoch / 逐条 revalidate,保证并发下不漏失效
  - 持久化采用「先写 building,提交时原子替换为 complete」的日志式流程,
    崩溃后重启时 recover() 丢弃所有 building 条目
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid


class CrashSimulated(Exception):
    """模拟「缓存生成中断」:条目已登记为 building 后进程崩溃。"""


class Entry:
    __slots__ = ("id", "level", "key", "resources", "deps", "payload",
                 "status", "size_bytes", "created", "note")

    def __init__(self, level, key, resources, deps, payload, note=""):
        self.id = uuid.uuid4().hex[:12]
        self.level = level                # 'L1' | 'L2' | 'L3'
        self.key = key                    # dict,规范化键组件
        self.resources = dict(resources)  # {资源名: 构建时版本}
        self.deps = list(deps)            # 上游条目 id
        self.payload = payload            # 已是 JSON 可序列化结构
        self.status = "building"
        self.size_bytes = 0
        self.created = time.time()
        self.note = note

    def to_json(self):
        return {"id": self.id, "level": self.level, "key": self.key,
                "resources": self.resources, "deps": self.deps,
                "payload": self.payload, "status": self.status,
                "size_bytes": self.size_bytes, "created": self.created,
                "note": self.note}

    @staticmethod
    def from_json(d):
        e = Entry.__new__(Entry)
        e.id, e.level, e.key = d["id"], d["level"], d["key"]
        e.resources, e.deps, e.payload = d["resources"], d["deps"], d["payload"]
        e.status, e.size_bytes, e.created = d["status"], d["size_bytes"], d["created"]
        e.note = d.get("note", "")
        return e


class CacheStore:
    def __init__(self, path: str | None = None):
        self._lock = threading.RLock()
        self.entries: dict[str, Entry] = {}
        self.resources: dict[str, int] = {}   # 当前资源版本注册表
        self.epoch = 0                        # 每次失效 +1
        self.events: list[dict] = []          # 失效/恢复事件日志
        self.stats = {"hits_L1": 0, "hits_L2": 0, "hits_L3": 0,
                      "builds_L1": 0, "builds_L2": 0, "builds_L3": 0,
                      "invalidated": 0, "recovered": 0}
        self.path = path
        self.crash_after_building = False     # 演示用:模拟中断
        self._last_persist = 0.0
        if path:
            os.makedirs(path, exist_ok=True)
            self._load()

    # ---------------- 资源版本 ----------------
    def resource_version(self, name: str) -> int:
        with self._lock:
            return self.resources.get(name, 0)

    def sync_resource(self, name: str, version: int):
        """登记/对齐资源当前版本(不触发失效)。"""
        with self._lock:
            self.resources[name] = version

    # ---------------- 查询 ----------------
    def find(self, level: str, **key_fields) -> list[Entry]:
        """按键组件精确匹配候选条目(调用方再自行 valid() 校验)。"""
        with self._lock:
            out = []
            for e in self.entries.values():
                if e.level != level or e.status != "complete":
                    continue
                if all(e.key.get(k) == v for k, v in key_fields.items()):
                    out.append(e)
            return out

    def get(self, entry_id: str) -> Entry | None:
        with self._lock:
            e = self.entries.get(entry_id)
            return e if e and e.status == "complete" else None

    def valid(self, e: Entry) -> bool:
        """条目当前是否可用:状态 complete 且所有记录的资源版本与注册表一致。"""
        with self._lock:
            if e.id not in self.entries or e.status != "complete":
                return False
            return all(self.resources.get(k) == v for k, v in e.resources.items())

    # ---------------- 构建(两阶段:building → complete) ----------------
    def begin_build(self, e: Entry) -> Entry:
        with self._lock:
            e.size_bytes = len(json.dumps(e.payload, ensure_ascii=False, default=str))
            self.entries[e.id] = e
            self.stats[f"builds_{e.level}"] += 1
            self._persist(force=self.crash_after_building)
            if self.crash_after_building:
                # 模拟进程在「已登记 building、尚未提交」时崩溃
                self.crash_after_building = False
                raise CrashSimulated(f"构建 {e.level} 条目 {e.id} 时进程中断")
            return e

    def commit(self, entry_id: str) -> bool:
        """原子提交。若构建期间资源已失效,调用方应放弃提交并改用 discard。"""
        with self._lock:
            e = self.entries.get(entry_id)
            if not e or e.status != "building":
                return False
            e.status = "complete"
            self._persist()
            return True

    def discard(self, entry_id: str):
        with self._lock:
            self.entries.pop(entry_id, None)
            self._persist()

    # ---------------- 失效传播 ----------------
    def invalidate_resource(self, name: str, new_version: int, reason: str) -> list[str]:
        """资源版本升级:使所有记录了该资源旧版本的条目失效,并沿依赖边级联。"""
        with self._lock:
            self.resources[name] = new_version
            victims = [e for e in self.entries.values()
                       if name in e.resources and e.resources[name] != new_version]
            removed = self._cascade(victims)
            self.epoch += 1
            self.events.append({"ts": time.time(), "kind": "invalidate",
                                "resource": name, "reason": reason,
                                "removed": sorted(removed)})
            self.stats["invalidated"] += len(removed)
            self._persist(force=True)
            return sorted(removed)

    def invalidate_where(self, pred, reason: str) -> list[str]:
        """按谓词失效(如:权限范围包含被限制国家的条目)。"""
        with self._lock:
            victims = [e for e in self.entries.values() if pred(e)]
            removed = self._cascade(victims)
            self.epoch += 1
            self.events.append({"ts": time.time(), "kind": "invalidate",
                                "resource": None, "reason": reason,
                                "removed": sorted(removed)})
            self.stats["invalidated"] += len(removed)
            self._persist(force=True)
            return sorted(removed)

    def _cascade(self, victims) -> set[str]:
        """删除 victims 及所有(传递)依赖它们的条目。"""
        removed = set()
        queue = list(victims)
        while queue:
            e = queue.pop()
            if e.id in removed:
                continue
            removed.add(e.id)
            # 找出所有依赖 e 的下游条目
            for other in self.entries.values():
                if other.id not in removed and e.id in other.deps:
                    queue.append(other)
        for eid in removed:
            self.entries.pop(eid, None)
        return removed

    # ---------------- 并发支持 ----------------
    def snapshot_epoch(self) -> int:
        with self._lock:
            return self.epoch

    def lock(self):
        return self._lock

    # ---------------- 持久化与恢复 ----------------
    def _persist(self, force: bool = False):
        """落盘(节流 100ms)。失效/恢复/崩溃演示等关键路径用 force=True 同步写。

        正确性论证:丢失的 commit 落盘 → 崩溃后该条目在盘上仍是 building
        → recover() 丢弃 → 只是少一条缓存,结果永远正确。
        """
        if not self.path:
            return
        now = time.time()
        if not force and now - self._last_persist < 0.1:
            return
        self._last_persist = now
        tmp = os.path.join(self.path, "entries.json.tmp")
        dst = os.path.join(self.path, "entries.json")
        data = {"resources": self.resources, "epoch": self.epoch,
                "entries": [e.to_json() for e in self.entries.values()]}
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, dst)   # 原子替换

    def _load(self):
        dst = os.path.join(self.path, "entries.json")
        if not os.path.exists(dst):
            return
        with open(dst, encoding="utf-8") as f:
            data = json.load(f)
        self.resources = data.get("resources", {})
        self.epoch = data.get("epoch", 0)
        for d in data.get("entries", []):
            e = Entry.from_json(d)
            self.entries[e.id] = e

    def recover(self) -> list[str]:
        """启动恢复:丢弃所有 building 状态的半成品条目(稍后按需重建)。"""
        with self._lock:
            incomplete = [e.id for e in self.entries.values() if e.status == "building"]
            for eid in incomplete:
                del self.entries[eid]
            if incomplete:
                self.epoch += 1
                self.events.append({"ts": time.time(), "kind": "recover",
                                    "resource": None,
                                    "reason": "丢弃中断时未完成的 building 条目",
                                    "removed": sorted(incomplete)})
                self.stats["recovered"] += len(incomplete)
                self._persist(force=True)
            return sorted(incomplete)

    # ---------------- 统计 ----------------
    def level_stats(self) -> dict:
        with self._lock:
            out = {}
            for lv in ("L1", "L2", "L3"):
                es = [e for e in self.entries.values() if e.level == lv]
                out[lv] = {"entries": len(es),
                           "bytes": sum(e.size_bytes for e in es),
                           "hits": self.stats[f"hits_{lv}"],
                           "builds": self.stats[f"builds_{lv}"]}
            out["invalidated"] = self.stats["invalidated"]
            out["recovered"] = self.stats["recovered"]
            return out

    def list_entries(self) -> list[dict]:
        with self._lock:
            return [{"id": e.id, "level": e.level, "key": e.key,
                     "resources": e.resources, "deps": e.deps,
                     "status": e.status, "size_bytes": e.size_bytes,
                     "valid": self.valid(e), "note": e.note}
                    for e in sorted(self.entries.values(),
                                    key=lambda x: (x.level, x.created))]
