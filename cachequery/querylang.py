"""有限查询语言:词法分析、递归下降解析、规范化、筛选条件蕴含证明。

语法(大小写不敏感的关键字):

    SELECT sum(amount), avg(amount), count_distinct(user_id)
    FROM events
    WHERE country = 'CN' AND date >= '2026-01-01' AND date < '2026-02-01'
          AND channel IN ('web', 'app') AND amount IS NOT NULL
    GROUP BY country, channel

支持聚合: sum / avg / count_distinct
支持条件: =  !=  <  <=  >  >=  IN (...)  IS NULL  IS NOT NULL,仅以 AND 连接
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


class ParseError(Exception):
    pass


# ---------------------------------------------------------------- AST

@dataclass(frozen=True)
class Measure:
    kind: str          # 'sum' | 'avg' | 'count_distinct'
    column: str

    def norm(self) -> str:
        return f"{self.kind}({self.column})"


def _fmt(v) -> str:
    if v is None:
        return "NULL"
    if isinstance(v, str):
        return "'%s'" % v.replace("'", "''")
    if isinstance(v, float) and v.is_integer():
        return repr(int(v))
    return repr(v)


@dataclass(frozen=True)
class Cond:
    column: str
    op: str                 # = != < <= > >= IN 'IS NULL' 'IS NOT NULL'
    value: object = None    # str | int | float | tuple(IN 列表) | None

    def norm(self) -> str:
        if self.op in ("IS NULL", "IS NOT NULL"):
            return f"{self.column} {self.op}"
        if self.op == "IN":
            vals = ", ".join(sorted(_fmt(v) for v in self.value))
            return f"{self.column} IN ({vals})"
        return f"{self.column} {self.op} {_fmt(self.value)}"


@dataclass(frozen=True)
class Query:
    measures: tuple          # tuple[Measure] 已按 norm 排序去重
    table: str
    conds: tuple             # tuple[Cond]   已按 norm 排序去重
    group_by: tuple          # tuple[str]    已排序去重

    def norm(self) -> str:
        ms = ", ".join(m.norm() for m in self.measures)
        cs = " AND ".join(c.norm() for c in self.conds) if self.conds else "TRUE"
        gb = ", ".join(self.group_by) if self.group_by else "(不分组)"
        return f"SELECT {ms} FROM {self.table} WHERE {cs} GROUP BY {gb}"

    def filter_norm(self) -> str:
        """仅筛选部分的规范化串,用作缓存键组件。"""
        return " AND ".join(c.norm() for c in self.conds) if self.conds else "TRUE"


# ---------------------------------------------------------------- 词法

_TOKEN_RE = re.compile(r"""
    \s*(?:
        (?P<string>'(?:[^']|'')*')
      | (?P<number>-?\d+(?:\.\d+)?)
      | (?P<ident>[A-Za-z_][A-Za-z0-9_]*)
      | (?P<op><=|>=|!=|<>|=|<|>)
      | (?P<punct>[(),])
    )
""", re.VERBOSE)

_FLIP = {"<": ">", ">": "<", "<=": ">=", ">=": "<=", "=": "=", "!=": "!=", "<>": "!="}


def _tokenize(text: str):
    tokens, pos = [], 0
    while pos < len(text):
        if text[pos:].strip() == "":
            break
        m = _TOKEN_RE.match(text, pos)
        if not m:
            raise ParseError(f"无法识别的输入: {text[pos:pos+20]!r}")
        pos = m.end()
        kind = m.lastgroup
        raw = m.group(kind)
        if kind == "string":
            tokens.append(("STRING", raw[1:-1].replace("''", "'")))
        elif kind == "number":
            tokens.append(("NUMBER", float(raw) if "." in raw else int(raw)))
        elif kind == "ident":
            tokens.append(("IDENT", raw))
        elif kind == "op":
            tokens.append(("OP", "!=" if raw == "<>" else raw))
        else:
            tokens.append(("PUNCT", raw))
    tokens.append(("EOF", None))
    return tokens


# ---------------------------------------------------------------- 语法

class _Parser:
    def __init__(self, text: str):
        self.toks = _tokenize(text)
        self.i = 0

    def peek(self):
        return self.toks[self.i]

    def next(self):
        t = self.toks[self.i]
        self.i += 1
        return t

    def expect_kw(self, *kws):
        t = self.next()
        if t[0] != "IDENT" or t[1].upper() not in kws:
            raise ParseError(f"期望 {'/'.join(kws)},得到 {t[1]!r}")
        return t[1].upper()

    def expect(self, kind, value=None):
        t = self.next()
        if t[0] != kind or (value is not None and t[1] != value):
            raise ParseError(f"期望 {kind} {value or ''},得到 {t!r}")
        return t[1]

    # ---- 文法 ----
    def parse(self) -> "Query":
        self.expect_kw("SELECT")
        measures = [self.parse_measure()]
        while self.peek() == ("PUNCT", ","):
            self.next()
            measures.append(self.parse_measure())
        self.expect_kw("FROM")
        table = self.expect("IDENT")
        conds = []
        if self.peek()[0] == "IDENT" and self.peek()[1].upper() == "WHERE":
            self.next()
            conds = self.parse_conds()
        group_by = []
        if self.peek()[0] == "IDENT" and self.peek()[1].upper() == "GROUP":
            self.next()
            self.expect_kw("BY")
            group_by = [self.expect("IDENT")]
            while self.peek() == ("PUNCT", ","):
                self.next()
                group_by.append(self.expect("IDENT"))
        if self.peek()[0] != "EOF":
            raise ParseError(f"查询末尾有多余内容: {self.peek()[1]!r}")
        return normalize(Query(
            measures=tuple(measures), table=table.lower(),
            conds=tuple(conds), group_by=tuple(group_by)))

    def parse_measure(self) -> Measure:
        kw = self.expect_kw("SUM", "AVG", "COUNT_DISTINCT")
        self.expect("PUNCT", "(")
        col = self.expect("IDENT")
        self.expect("PUNCT", ")")
        return Measure(kind=kw.lower(), column=col.lower())

    def parse_conds(self):
        conds = [self.parse_cond()]
        while self.peek()[0] == "IDENT" and self.peek()[1].upper() == "AND":
            self.next()
            conds.append(self.parse_cond())
        return conds

    def parse_operand(self):
        t = self.next()
        if t[0] == "IDENT":
            if t[1].upper() == "NULL":
                return ("lit", None)
            return ("col", t[1].lower())
        if t[0] in ("STRING", "NUMBER"):
            return ("lit", t[1])
        raise ParseError(f"期望列名或字面量,得到 {t[1]!r}")

    def parse_cond(self) -> Cond:
        left = self.parse_operand()
        t = self.peek()
        # IS NULL / IS NOT NULL
        if t[0] == "IDENT" and t[1].upper() == "IS":
            self.next()
            neg = False
            if self.peek()[0] == "IDENT" and self.peek()[1].upper() == "NOT":
                self.next()
                neg = True
            self.expect_kw("NULL")
            if left[0] != "col":
                raise ParseError("IS NULL 只能作用于列")
            return Cond(left[1], "IS NOT NULL" if neg else "IS NULL")
        # IN (...)
        if t[0] == "IDENT" and t[1].upper() == "IN":
            self.next()
            if left[0] != "col":
                raise ParseError("IN 左侧必须是列名")
            self.expect("PUNCT", "(")
            vals = [self.parse_operand()]
            while self.peek() == ("PUNCT", ","):
                self.next()
                vals.append(self.parse_operand())
            self.expect("PUNCT", ")")
            if any(v[0] != "lit" for v in vals):
                raise ParseError("IN 列表只接受字面量")
            return Cond(left[1], "IN", tuple(v[1] for v in vals))
        # 比较运算
        op = self.expect("OP")
        right = self.parse_operand()
        if left[0] == "col" and right[0] == "lit":
            if right[1] is None:
                raise ParseError("与 NULL 的比较请使用 IS NULL / IS NOT NULL")
            return Cond(left[1], op, right[1])
        if left[0] == "lit" and right[0] == "col":
            if left[1] is None:
                raise ParseError("与 NULL 的比较请使用 IS NULL / IS NOT NULL")
            return Cond(right[1], _FLIP[op], left[1])   # 字面量在左 → 翻转
        raise ParseError("比较运算必须一侧是列、一侧是字面量")


def parse(text: str) -> Query:
    """解析并规范化查询文本。"""
    return _Parser(text).parse()


def normalize(q: Query) -> Query:
    """规范化:条件排序去重、度量排序去重、分组列排序去重。

    使得 `WHERE a=1 AND b=2` 与 `WHERE b=2 AND a=1` 得到同一缓存键。
    """
    conds = tuple(sorted({c.norm(): c for c in q.conds}.values(), key=lambda c: c.norm()))
    measures = tuple(sorted({m.norm(): m for m in q.measures}.values(), key=lambda m: m.norm()))
    group_by = tuple(sorted(set(q.group_by)))
    return Query(measures=measures, table=q.table, conds=conds, group_by=group_by)


# ---------------------------------------------------------------- 行匹配

def match_cond(row: dict, c: Cond) -> bool:
    """SQL 风格空值语义:除 IS NULL/IS NOT NULL 外,任何涉及 NULL 的比较都为假。"""
    v = row.get(c.column)
    if c.op == "IS NULL":
        return v is None
    if c.op == "IS NOT NULL":
        return v is not None
    if v is None:
        return False
    if c.op == "IN":
        return v in c.value
    if c.op == "=":
        return v == c.value
    if c.op == "!=":
        return v != c.value
    try:
        if c.op == "<":
            return v < c.value
        if c.op == "<=":
            return v <= c.value
        if c.op == ">":
            return v > c.value
        if c.op == ">=":
            return v >= c.value
    except TypeError:
        return False
    raise ParseError(f"未知运算符 {c.op}")


def match_all(row: dict, conds) -> bool:
    return all(match_cond(row, c) for c in conds)


# ---------------------------------------------------------------- 蕴含证明
#
# 目标:判断「筛选 F2 的行集合 ⊆ 筛选 F1 的行集合」,即 F2 ⇒ F1。
# 只给出可证明时为真;证明不了就返回假(调用方真实重算,绝不猜测)。
#
# 方法:把每个列上的合取条件抽象为 (集合, 区间, 空值标记) 三元组,
# 再逐条验证 F1 的每个条件都被 F2 的抽象域所包含。

@dataclass
class _Dom:
    set_: frozenset | None = None        # 取值集合(来自 = / IN 的交集)
    lo: tuple | None = None              # (值, 是否闭区间)
    hi: tuple | None = None
    null: str = "any"                    # 'only' | 'no' | 'any'


def _domain(conds) -> dict:
    doms: dict[str, _Dom] = {}

    def dom(col):
        return doms.setdefault(col, _Dom())

    for c in conds:
        d = dom(c.column)
        if c.op == "=":
            d.set_ = frozenset([c.value]) if d.set_ is None else d.set_ & {c.value}
            d.null = "no"
        elif c.op == "IN":
            s = frozenset(c.value)
            d.set_ = s if d.set_ is None else d.set_ & s
            d.null = "no"
        elif c.op == "IS NULL":
            d.null = "only"
        elif c.op == "IS NOT NULL":
            if d.null != "only":
                d.null = "no"
        elif c.op in ("<", "<="):
            v = (c.value, c.op == "<=")
            if d.hi is None or (v[0], not v[1]) < (d.hi[0], not d.hi[1]):
                d.hi = v
            d.null = "no"
        elif c.op in (">", ">="):
            v = (c.value, c.op == ">=")
            if d.lo is None or (v[0], v[1]) > (d.lo[0], d.lo[1]):
                d.lo = v
            d.null = "no"
        # '!=' 无法纳入抽象域,保守忽略(只会让我们证明不了,不会证错)
    return doms


def _cond_implied(doms: dict, conds2, c1: Cond) -> tuple[bool, str]:
    """判断 doms(F2 的抽象域) 是否蕴含条件 c1,并给出证明串。"""
    d = doms.get(c1.column)
    if c1.op == "IS NULL":
        if d and d.null == "only":
            return True, f"{c1.column}: F2 已限定 IS NULL"
        return False, ""
    if c1.op == "IS NOT NULL":
        if d and d.null in ("no",):
            return True, f"{c1.column}: F2 已限定非空"
        return False, ""
    if c1.op == "=":
        if d and d.set_ is not None and d.set_ == {c1.value}:
            return True, f"{c1.column}: F2 同样限定 = {_fmt(c1.value)}"
        return False, ""
    if c1.op == "IN":
        if d and d.set_ is not None and d.set_ <= frozenset(c1.value):
            return True, f"{c1.column}: F2 集合 ⊆ IN 列表"
        return False, ""
    if c1.op == "!=":
        if d and d.set_ is not None and c1.value not in d.set_:
            return True, f"{c1.column}: F2 集合不含 {_fmt(c1.value)}"
        if any(c2.column == c1.column and c2.op == "!=" and c2.value == c1.value for c2 in conds2):
            return True, f"{c1.column}: F2 含相同 != 条件"
        return False, ""
    if c1.op in ("<", "<="):
        if d and d.hi is not None:
            v, incl = d.hi
            if (v, incl) <= (c1.value, c1.op == "<=") or (v < c1.value):
                return True, f"{c1.column}: F2 上界 {_fmt(v)}{'闭' if incl else '开'} ⊆ {_fmt(c1.value)}"
        return False, ""
    if c1.op in (">", ">="):
        if d and d.lo is not None:
            v, incl = d.lo
            if (v, incl) >= (c1.value, c1.op == ">=") or (v > c1.value):
                return True, f"{c1.column}: F2 下界 {_fmt(v)}{'闭' if incl else '开'} ⊇ {_fmt(c1.value)}"
        return False, ""
    return False, ""


def implies(conds2, conds1) -> tuple[bool, list]:
    """判断 F2 ⇒ F1(即 F2 的行集是 F1 行集的子集)。

    返回 (是否可证明, 证明串列表)。证明不了就返回 False —— 调用方必须重算。
    """
    doms2 = _domain(conds2)
    proofs = []
    for c1 in conds1:
        ok, why = _cond_implied(doms2, conds2, c1)
        if not ok:
            return False, []
        proofs.append(why)
    return True, proofs


# ---------------------------------------------------------------- 分区裁剪

def select_days(conds, all_days) -> list:
    """用 date 条件对可用分区(日)做裁剪。字符串日期字典序即时间序。"""
    date_conds = [c for c in conds if c.column == "date"]
    if not date_conds:
        return list(all_days)
    return [d for d in all_days if match_all({"date": d}, date_conds)]
