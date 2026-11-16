"""HTTP 服务与网页:查询执行、复用来源与证明条件展示、失效控制台、缓存检查器、基准对比。

用法: python -m cachequery.server [port]   然后浏览器打开 http://127.0.0.1:8000/
"""
from __future__ import annotations

import json
import shutil
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .cache import CacheStore, CrashSimulated
from .data import Dataset
from .executor import Engine, SEMANTICS
from . import samples

STORE_PATH = "./cqstore"
_lock = threading.Lock()
engine: Engine = None


def _new_engine(persistent=True):
    data = Dataset()
    store = CacheStore(STORE_PATH if persistent else None)
    return Engine(data, store)


def get_engine() -> Engine:
    global engine
    if engine is None:
        engine = _new_engine(persistent=True)
        engine.store.recover()          # 启动即恢复:丢弃上次中断留下的半成品
    return engine


# ---------------------------------------------------------------- API

def api_query(body):
    eng = get_engine()
    q, user = body["q"], body.get("user", "alice")
    cached = eng.run(q, user=user, use_cache=bool(body.get("cache", True)))
    base = eng.run(q, user=user, use_cache=False)
    return {"cached": cached, "baseline": {"rows": base["rows"], "columns": base["columns"],
                                           "elapsed_ms": base["elapsed_ms"],
                                           "rows_scanned": base["rows_scanned"]},
            "equal": cached["rows"] == base["rows"]}


def api_state(_body):
    eng = get_engine()
    return {"entries": eng.store.list_entries(),
            "stats": eng.store.level_stats(),
            "events": eng.store.events[-50:],
            "epoch": eng.store.epoch,
            "semantics_version": eng.semantics_version,
            "semantics_desc": SEMANTICS[eng.semantics_version]["desc"],
            "perm_epoch": eng.perm_epoch,
            "permissions": {u: (list(s) if s else "ALL") for u, s in eng.permissions.items()},
            "restricted": sorted(eng.restricted),
            "data_versions": eng.data.data_version,
            "corrections": eng.data.corrections[-10:]}


def api_correct(body):
    return get_engine().correct_data(body["day"], body.get("mode", "restated_amount"))


def api_narrow(body):
    return get_engine().narrow_permission(body["user"], body["countries"])


def api_restrict(body):
    return get_engine().restrict_country(body["country"])


def api_semantics(body):
    return get_engine().upgrade_semantics(int(body["version"]))


def api_crash(_body):
    """模拟缓存生成中断:下一次构建在 building 落盘后崩溃。"""
    eng = get_engine()
    eng.store.crash_after_building = True
    try:
        eng.run("SELECT sum(amount) FROM events WHERE channel = 'app' "
                "AND date >= '2026-01-07' AND date < '2026-01-09' GROUP BY country")
    except CrashSimulated as exc:
        return {"crashed": True, "detail": str(exc)}
    return {"crashed": False, "detail": "本次未触发构建(可能全部命中),换个筛选再试"}


def api_recover(_body):
    eng = get_engine()
    recovered = eng.store.recover()
    return {"recovered": recovered}


def api_reset(_body):
    global engine
    with _lock:
        engine = None
        shutil.rmtree(STORE_PATH, ignore_errors=True)
    get_engine()
    return {"ok": True}


def api_benchmark(_body):
    return samples.benchmark(get_engine())


def api_sample(body):
    """样例在独立的全新引擎上运行,不污染主引擎状态。"""
    name = body["name"]
    eng = samples.new_engine()
    if name == "same_shape":
        return samples.sample_same_shape(eng)
    if name == "nulls":
        return samples.sample_nulls(eng)
    if name == "cross_version":
        return samples.sample_cross_version(eng)
    if name == "containment":
        return samples.sample_containment(eng)
    if name == "concurrency":
        return samples.concurrency_check(eng)
    if name == "recovery":
        with tempfile.TemporaryDirectory() as tmp:
            return samples.recovery_check(tmp)
    raise ValueError(f"未知样例 {name}")


ROUTES = {
    "/api/query": api_query,
    "/api/state": api_state,
    "/api/correct": api_correct,
    "/api/narrow": api_narrow,
    "/api/restrict": api_restrict,
    "/api/semantics": api_semantics,
    "/api/crash": api_crash,
    "/api/recover": api_recover,
    "/api/reset": api_reset,
    "/api/benchmark": api_benchmark,
    "/api/sample": api_sample,
}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # 静音
        pass

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/?"):
            self._send(200, INDEX_HTML.encode(), "text/html; charset=utf-8")
        else:
            self._send(404, b"not found")

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
            fn = ROUTES.get(self.path)
            if fn is None:
                self._send(404, json.dumps({"error": "unknown route"}).encode(),
                           "application/json")
                return
            with _lock:
                out = fn(body)
            self._send(200, json.dumps(out, ensure_ascii=False, default=str).encode(),
                       "application/json; charset=utf-8")
        except PermissionError as exc:
            self._send(403, json.dumps({"error": str(exc)}, ensure_ascii=False).encode(),
                       "application/json; charset=utf-8")
        except Exception as exc:  # noqa: BLE001
            self._send(400, json.dumps(
                {"error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False).encode(),
                "application/json; charset=utf-8")

    def _send(self, code, data, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


# ---------------------------------------------------------------- 页面

INDEX_HTML = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>cachequery · 分层复用查询缓存</title>
<style>
  :root { --bg:#0f1420; --panel:#182034; --line:#2a3550; --fg:#dbe4f5; --dim:#8b98b8;
          --ok:#4ade80; --bad:#f87171; --l1:#f59e0b; --l2:#38bdf8; --l3:#c084fc; --raw:#94a3b8; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--fg);
         font:14px/1.5 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; }
  header { padding:14px 20px; border-bottom:1px solid var(--line); }
  header h1 { margin:0; font-size:17px; }
  header .sub { color:var(--dim); font-size:12px; margin-top:4px; }
  main { display:grid; grid-template-columns:1.2fr 1fr; gap:14px; padding:14px 20px; }
  section { background:var(--panel); border:1px solid var(--line); border-radius:8px;
            padding:12px 14px; margin-bottom:14px; }
  h2 { font-size:13px; margin:0 0 10px; color:#9fb4dd; letter-spacing:.05em; }
  textarea { width:100%; background:#0b1020; color:var(--fg); border:1px solid var(--line);
             border-radius:6px; padding:8px; font:inherit; resize:vertical; }
  select,button,input { background:#0b1020; color:var(--fg); border:1px solid var(--line);
             border-radius:6px; padding:6px 10px; font:inherit; cursor:pointer; }
  button:hover { border-color:#4c64a8; }
  button.primary { background:#25407a; border-color:#3a5aa8; }
  table { border-collapse:collapse; width:100%; font-size:12.5px; }
  th,td { border:1px solid var(--line); padding:4px 8px; text-align:left; }
  th { color:#9fb4dd; }
  .tag { display:inline-block; padding:1px 7px; border-radius:10px; font-size:11px;
         font-weight:700; margin-right:6px; }
  .L1{background:rgba(245,158,11,.15);color:var(--l1);border:1px solid var(--l1);}
  .L2{background:rgba(56,189,248,.15);color:var(--l2);border:1px solid var(--l2);}
  .L3{background:rgba(192,132,252,.15);color:var(--l3);border:1px solid var(--l3);}
  .RAW{background:rgba(148,163,184,.15);color:var(--raw);border:1px solid var(--raw);}
  .proof { color:var(--dim); font-size:12px; margin:2px 0 2px 18px; }
  .proof li { margin:1px 0; }
  .ok { color:var(--ok); } .bad { color:var(--bad); }
  .row { display:flex; gap:8px; align-items:center; flex-wrap:wrap; margin:6px 0; }
  .cards { display:grid; grid-template-columns:repeat(3,1fr); gap:8px; }
  .card { background:#0b1020; border:1px solid var(--line); border-radius:6px; padding:8px 10px; }
  .card b { font-size:16px; }
  .muted { color:var(--dim); font-size:12px; }
  .rej { color:#fbbf24; font-size:12.5px; }
  pre { background:#0b1020; border:1px solid var(--line); border-radius:6px;
        padding:8px; overflow:auto; max-height:300px; font-size:12px; }
  .evt { font-size:12px; color:var(--dim); border-bottom:1px dashed var(--line); padding:3px 0; }
</style>
</head>
<body>
<header>
  <h1>cachequery — 有限查询语言 · 三层复用缓存 · 失效传播</h1>
  <div class="sub" id="hdr"></div>
</header>
<main>
<div>
  <section>
    <h2>① 查询(复用来源与证明条件)</h2>
    <textarea id="q" rows="4">SELECT sum(amount), avg(amount), count_distinct(user_id)
FROM events
WHERE date >= '2026-01-01' AND date < '2026-01-06'
GROUP BY country</textarea>
    <div class="row">
      用户 <select id="user"><option>alice</option><option>bob</option></select>
      <label><input type="checkbox" id="usecache" checked> 启用缓存</label>
      <button class="primary" onclick="runQuery()">执行</button>
      <span class="muted">每次执行同时跑无缓存基线对照</span>
    </div>
    <div id="qerr"></div>
    <div id="qmeta"></div>
    <div id="qresult"></div>
    <div id="qprov"></div>
  </section>

  <section>
    <h2>② 失效控制台(数据更正 / 权限收窄 / 口径升级 / 中断恢复)</h2>
    <div class="row">
      数据更正:分区 <select id="cday"></select>
      <select id="cmode">
        <option value="restated_amount">web 金额重述 +10%</option>
        <option value="fix_null_amount">空金额补 0</option>
        <option value="fix_null_country">空国家补 CN</option>
      </select>
      <button onclick="post('/api/correct',{day:cday.value,mode:cmode.value})">更正</button>
    </div>
    <div class="row">
      权限收窄:用户 <select id="nuser"><option>alice</option><option>bob</option></select>
      可见 <input id="ncountries" value="CN" size="10">
      <button onclick="post('/api/narrow',{user:nuser.value,countries:ncountries.value.split(',').map(s=>s.trim())})">收窄</button>
      全局限制国家 <select id="rc"><option>US</option><option>JP</option><option>CN</option></select>
      <button onclick="post('/api/restrict',{country:rc.value})">限制</button>
    </div>
    <div class="row">
      口径升级:
      <button onclick="post('/api/semantics',{version:1})">v1 毛额</button>
      <button onclick="post('/api/semantics',{version:2})">v2 净额</button>
      <span class="muted" id="semdesc"></span>
    </div>
    <div class="row">
      中断演示:
      <button onclick="doCrash()">模拟构建中断</button>
      <button onclick="post('/api/recover',{})">恢复(丢弃半成品)</button>
      <button onclick="post('/api/reset',{})">重置全部</button>
    </div>
    <pre id="console"></pre>
  </section>

  <section>
    <h2>③ 样例(同形不同含义 / 空值 / 跨版本 / 范围包含误用 / 并发 / 恢复)</h2>
    <div class="row">
      <button onclick="sample('same_shape')">同形不同含义</button>
      <button onclick="sample('nulls')">空值</button>
      <button onclick="sample('cross_version')">跨版本</button>
      <button onclick="sample('containment')">范围包含误用</button>
      <button onclick="sample('concurrency')">并发失效</button>
      <button onclick="sample('recovery')">中断恢复</button>
    </div>
    <pre id="sampleout">点击按钮运行样例(在独立引擎实例上,不影响主缓存)</pre>
  </section>
</div>

<div>
  <section>
    <h2>④ 缓存层次(空间 vs 计算)</h2>
    <div class="cards" id="lv"></div>
    <div class="muted" style="margin-top:6px">
      L1 原始分片:空间最大,可在「行集包含」证明下再过滤复用 ·
      L2 可组合状态:空间小,跨分区合并/分组上卷 ·
      L3 最终结果:空间最小,仅完全相同查询
    </div>
  </section>
  <section>
    <h2>⑤ 缓存条目(依赖登记)</h2>
    <div id="entries" style="max-height:320px;overflow:auto"></div>
  </section>
  <section>
    <h2>⑥ 失效事件</h2>
    <div id="events" style="max-height:180px;overflow:auto"></div>
  </section>
  <section>
    <h2>⑦ 基准对比(缓存 vs 无缓存)</h2>
    <button onclick="doBench()">运行基准负载</button>
    <div id="bench"></div>
  </section>
</div>
</main>
<script>
const $ = id => document.getElementById(id);
async function api(path, body){
  const r = await fetch(path, {method:'POST', headers:{'Content-Type':'application/json'},
                               body: JSON.stringify(body||{})});
  const j = await r.json();
  if (!r.ok) throw new Error(j.error || r.statusText);
  return j;
}
function esc(s){ return String(s).replace(/[&<>]/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c])); }
function table(cols, rows){
  let h = '<table><tr>' + cols.map(c=>'<th>'+esc(c)+'</th>').join('') + '</tr>';
  for (const r of rows)
    h += '<tr>' + r.map(v=>'<td>'+(v===null?'<i>NULL</i>':esc(v))+'</td>').join('') + '</tr>';
  return h + '</table>';
}
async function runQuery(){
  $('qerr').innerHTML=''; $('qresult').innerHTML=''; $('qprov').innerHTML=''; $('qmeta').innerHTML='';
  try {
    const j = await api('/api/query', {q:$('q').value, user:$('user').value,
                                       cache:$('usecache').checked});
    const c = j.cached, b = j.baseline;
    $('qmeta').innerHTML =
      `结果与基线一致: <b class="${j.equal?'ok':'bad'}">${j.equal?'✓ 一致':'✗ 不一致'}</b> · `+
      `缓存 ${c.elapsed_ms} ms / 基线 ${b.elapsed_ms} ms · `+
      `扫描行 ${c.rows_scanned} / 基线 ${b.rows_scanned}`;
    $('qresult').innerHTML = '<h2>结果</h2>' + table(c.columns, c.rows);
    let p = '<h2>复用来源与证明条件</h2>';
    for (const s of c.provenance){
      p += `<div><span class="tag ${s.source}">${s.source}</span>`+
           `<span class="muted">分区 ${esc(s.day||'*')}${s.entry?' · 条目 '+s.entry:''}</span>`+
           '<ul class="proof">' + s.proof.map(x=>'<li>'+esc(x)+'</li>').join('') + '</ul></div>';
    }
    if (c.rejected && c.rejected.length){
      p += '<h2>拒绝复用(无法证明等价 → 真实重算)</h2>';
      for (const r of c.rejected)
        p += `<div class="rej">✗ [${r.level}] ${esc(r.reason)}</div>`;
    }
    $('qprov').innerHTML = p;
    refresh();
  } catch(e){ $('qerr').innerHTML = '<span class="bad">'+esc(e.message)+'</span>'; }
}
async function post(path, body){
  try { const j = await api(path, body);
        $('console').textContent = JSON.stringify(j, null, 2); }
  catch(e){ $('console').textContent = '错误: ' + e.message; }
  refresh();
}
async function doCrash(){
  try { const j = await api('/api/crash', {});
        $('console').textContent = JSON.stringify(j, null, 2); }
  catch(e){ $('console').textContent = '错误: ' + e.message; }
  refresh();
}
async function sample(name){
  $('sampleout').textContent = '运行中…';
  try {
    const j = await api('/api/sample', {name});
    let t = j.title + '\n\n';
    for (const [k,v] of Object.entries(j.check||{}))
      t += (v?'[OK] ':'[FAIL] ') + k + '\n';
    t += '\n' + JSON.stringify(j.steps ? j.steps.map(s=>s.action) : j, null, 2);
    $('sampleout').textContent = t;
  } catch(e){ $('sampleout').textContent = '错误: ' + e.message; }
}
async function doBench(){
  $('bench').innerHTML = '运行中…';
  const j = await api('/api/benchmark', {});
  const s = j.summary;
  let h = `<p>全部一致: <b class="${s['全部一致']?'ok':'bad'}">${s['全部一致']}</b> · `+
          `冷遍 ${s['冷遍缓存_ms']} ms · 热遍 ${s['热遍缓存_ms']} ms · 基线 ${s['基线总_ms']} ms<br>`+
          `扫描行: 冷遍 ${s['冷遍扫描行']} · 热遍 ${s['热遍扫描行']} · 基线 ${s['基线扫描行']}</p>`;
  h += table(['用户','一致','缓存ms','基线ms','缓存扫描','基线扫描','查询'],
             j.rows.map(r=>[r.user, r['一致']?'✓':'✗', r['缓存_ms'], r['基线_ms'],
                            r['缓存扫描行'], r['基线扫描行'], r.query]));
  $('bench').innerHTML = h;
  refresh();
}
async function refresh(){
  const j = await api('/api/state', {});
  $('hdr').textContent =
    `口径 v${j.semantics_version}(${j.semantics_desc}) · 权限纪元 ${j.perm_epoch} · `+
    `失效纪元 ${j.epoch} · 被限制国家: ${j.restricted.join(',')||'无'} · `+
    `权限: ` + Object.entries(j.permissions).map(([u,s])=>u+'='+(s==='ALL'?'ALL':s.join('/'))).join(' ');
  $('semdesc').textContent = j.semantics_desc;
  $('cday').innerHTML = Object.keys(j.data_versions)
    .map(d=>`<option>${d}</option>`).join('');
  let lv = '';
  for (const k of ['L1','L2','L3']){
    const s = j.stats[k];
    lv += `<div class="card"><span class="tag ${k}">${k}</span><b>${s.entries}</b> 条 · `+
          `${(s.bytes/1024).toFixed(1)} KB<br><span class="muted">命中 ${s.hits} · 构建 ${s.builds}</span></div>`;
  }
  $('lv').innerHTML = lv +
    `<div class="card" style="grid-column:span 3"><span class="muted">累计失效 ${j.stats.invalidated} 条 · 恢复丢弃 ${j.stats.recovered} 条</span></div>`;
  let e = '<table><tr><th>层</th><th>键</th><th>资源版本</th><th>依赖</th><th>大小</th><th>状态</th></tr>';
  for (const x of j.entries.slice().reverse()){
    e += `<tr><td><span class="tag ${x.level}">${x.level}</span></td>`+
         `<td class="muted">${esc(JSON.stringify(x.key)).slice(0,90)}</td>`+
         `<td class="muted">${esc(JSON.stringify(x.resources))}</td>`+
         `<td class="muted">${x.deps.join(',')||'-'}</td>`+
         `<td>${x.size_bytes}</td>`+
         `<td class="${x.valid?'ok':'bad'}">${x.status}${x.valid?'':'/失效'}</td></tr>`;
  }
  $('entries').innerHTML = e + '</table>';
  $('events').innerHTML = j.events.slice().reverse()
    .map(x=>`<div class="evt">[${x.kind}] ${esc(x.reason)} → 移除 ${x.removed.length} 条`+
            (x.removed.length?` <span class="muted">${x.removed.join(',')}</span>`:'')+`</div>`)
    .join('');
}
refresh();
</script>
</body>
</html>
"""


def main(port: int = 8000):
    get_engine()
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"cachequery 服务已启动: http://127.0.0.1:{port}/  (Ctrl+C 停止)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    import sys
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 8000)
