"""数据集:按日分区的明细表,带数据版本与更正操作。

表结构 events(user_id, country, channel, amount, active, refunded, date)
  - 故意包含空值:country / amount / user_id 均可能为 NULL
  - refunded / active 供口径(semantics)版本使用
"""
from __future__ import annotations

import random
import threading
from datetime import date, timedelta

COUNTRIES = ["CN", "JP", "US"]
CHANNELS = ["web", "app", "store"]


def _day_str(d: date) -> str:
    return d.isoformat()


class Dataset:
    """线程安全的分区数据集。每个分区(日)有独立 data_version。"""

    def __init__(self, seed: int = 20260101, days: int = 12, per_day: int = 150,
                 start: str = "2026-01-01"):
        self._lock = threading.RLock()
        self.partitions: dict[str, list[dict]] = {}
        self.data_version: dict[str, int] = {}
        self.corrections: list[dict] = []   # 更正历史(页面展示用)
        self._generate(seed, days, per_day, start)

    # ---------------- 生成 ----------------
    def _generate(self, seed, days, per_day, start):
        rng = random.Random(seed)
        d0 = date.fromisoformat(start)
        user_pool = [f"u{n:04d}" for n in range(1, 501)]   # 500 个可重复用户
        for i in range(days):
            day = _day_str(d0 + timedelta(days=i))
            rows = []
            for _ in range(per_day):
                r = rng.random()
                country = None if r < 0.05 else COUNTRIES[int(rng.random() * len(COUNTRIES))]
                amount = None if rng.random() < 0.05 else round(rng.random() * 200, 2)
                rows.append({
                    "user_id": None if rng.random() < 0.03 else user_pool[int(rng.random() * len(user_pool))],
                    "country": country,
                    "channel": CHANNELS[int(rng.random() * len(CHANNELS))],
                    "amount": amount,
                    "active": rng.random() < 0.7,
                    "refunded": rng.random() < 0.1,
                    "date": day,
                })
            with self._lock:
                self.partitions[day] = rows
                self.data_version[day] = 1

    # ---------------- 读 ----------------
    def days(self) -> list[str]:
        with self._lock:
            return sorted(self.partitions)

    def version(self, day: str) -> int:
        with self._lock:
            return self.data_version.get(day, 0)

    def snapshot(self, day: str) -> tuple[list[dict], int]:
        """原子地取 (行副本, 数据版本) —— 执行器据此构建可校验的缓存条目。"""
        with self._lock:
            rows = [dict(r) for r in self.partitions.get(day, [])]
            return rows, self.data_version.get(day, 0)

    # ---------------- 更正 ----------------
    def correct(self, day: str, fn, desc: str = "") -> int:
        """对某日分区应用更正函数 fn(row)->None,并递增该分区数据版本。

        数据更正是失效传播的源头之一:调用方(Engine)随后通知缓存层。
        """
        with self._lock:
            if day not in self.partitions:
                raise KeyError(f"无此分区: {day}")
            n = 0
            for row in self.partitions[day]:
                before = dict(row)
                fn(row)
                if row != before:
                    n += 1
            self.data_version[day] += 1
            ver = self.data_version[day]
            self.corrections.append({"day": day, "rows_changed": n,
                                     "new_version": ver, "desc": desc})
            return ver


def preset_correction(dataset: Dataset, day: str, mode: str) -> int:
    """预置的几种更正模式,供页面/演示调用。"""
    if mode == "fix_null_amount":
        # 把 NULL 金额补为 0.0(典型数据修复)
        return dataset.correct(
            day,
            lambda r: r.update(amount=0.0) if r["amount"] is None else None,
            "空金额补 0")
    if mode == "fix_null_country":
        return dataset.correct(
            day,
            lambda r: r.update(country="CN") if r["country"] is None else None,
            "空国家补 CN")
    if mode == "restated_amount":
        # 财报重述:web 渠道金额上调 10%
        def f(r):
            if r["channel"] == "web" and r["amount"] is not None:
                r["amount"] = round(r["amount"] * 1.1, 2)
        return dataset.correct(day, f, "web 渠道金额重述 +10%")
    raise ValueError(f"未知更正模式: {mode}")
