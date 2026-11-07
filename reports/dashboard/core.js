  // Shared state and helpers. dashboard.py joins the modules into one
  // function scope and calls start() after the last module.
  var B = window.BENCH || { runs: [] };
  var NS = "http://www.w3.org/2000/svg";
  var MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  var CHARTS = [];     // chart cards, added by the chart modules
  var RENDERERS = [];  // {order, fn}: called on every redraw
  var SETUP = [];      // called once before the first redraw
  function byOrder(a, b) { return a.order - b.order; }

  // ---------- helpers ----------
  function store(key, value) {
    try {
      if (value === undefined) return JSON.parse(localStorage.getItem(key));
      localStorage.setItem(key, JSON.stringify(value));
    } catch (e) { return null; }
    return null;
  }
  function h(tag, attrs, parent, text) {
    var e = document.createElement(tag);
    if (attrs) for (var k in attrs) e.setAttribute(k, attrs[k]);
    if (text != null) e.textContent = text;
    if (parent) parent.appendChild(e);
    return e;
  }
  function isNum(v) { return typeof v === "number" && isFinite(v); }
  function mean(a) { var v = a.filter(isNum); return v.length ? v.reduce(function (x, y) { return x + y; }, 0) / v.length : null; }
  function fmt(v, d) {
    if (!isNum(v)) return "–";
    return v.toLocaleString("en-US", { minimumFractionDigits: d || 0, maximumFractionDigits: d || 0 });
  }
  function fmtMs(v) {
    if (!isNum(v)) return "–";
    return v >= 1000 ? fmt(v / 1000, v >= 10000 ? 1 : 2) + " s" : fmt(v, v < 1 ? 2 : v < 10 ? 1 : 0) + " ms";
  }
  function fmtTokens(n) {
    if (!isNum(n)) return "–";
    if (n >= 1024 && n % 1024 === 0) return n / 1024 + "K";
    return String(n);
  }
  function fmtPct(v) { return isNum(v) ? Math.round(v * 100) + "%" : "–"; }
  function fmtDate(iso) {
    var m = /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})/.exec(iso || "");
    if (!m) return iso || "";
    return +m[3] + " " + MONTHS[+m[2] - 1] + " " + m[1] + ", " + m[4] + ":" + m[5];
  }

  // ---------- runs ----------
  var nameCount = {};
  B.runs.forEach(function (r) { nameCount[r.model] = (nameCount[r.model] || 0) + 1; });
  var RUNS = B.runs.map(function (r, i) {
    var label = nameCount[r.model] > 1 ? r.model + " · " + fmtDate(r.started) : r.model;
    return Object.assign({}, r, {
      idx: i,
      label: label,
      color: i < 8 ? "var(--s" + (i + 1) + ")" : "var(--muted)",
    });
  });
  var visible = (function () {
    var saved = store("bench.visible");
    var ids = RUNS.map(function (r) { return r.id; });
    if (Array.isArray(saved)) {
      var kept = saved.filter(function (id) { return ids.indexOf(id) >= 0; });
      // Show runs that are new since the last visit.
      var seen = store("bench.seen") || [];
      ids.forEach(function (id) { if (seen.indexOf(id) < 0) kept.push(id); });
      if (kept.length) return new Set(kept);
    }
    return new Set(ids);
  })();
  store("bench.seen", RUNS.map(function (r) { return r.id; }));
  function shown() { return RUNS.filter(function (r) { return visible.has(r.id); }); }

  // ---------- header ----------
  function drawHeader() {
    var mem = B.host && B.host.memory_gib;
    document.getElementById("host-line").textContent = "DGX Spark · " + ((B.host && B.host.gpu) || "GPU") + (mem ? " · " + fmt(mem, 1) + " GiB unified memory" : "");
    var models = {};
    RUNS.forEach(function (r) { models[r.model] = 1; });
    var last = RUNS.length ? fmtDate(RUNS[RUNS.length - 1].started) : "none";
    document.getElementById("lede").textContent =
      RUNS.length + " runs of " + Object.keys(models).length + " model configs on vLLM. Newest run: " + last + ".";
    document.getElementById("generated").textContent = "Built " + fmtDate((B.generated_at || "").replace("+00:00", "")) + " UTC from results/";
  }
  function drawSpectrum() {
    var host = document.getElementById("spectrum");
    host.replaceChildren();
    shown().forEach(function (r) { var sp = h("span", null, host); sp.style.background = r.color; });
  }
  RENDERERS.push({ order: 20, fn: drawSpectrum });

  function renderAll() {
    RENDERERS.slice().sort(byOrder).forEach(function (r) { r.fn(); });
  }

  // Theme toggle: system, light, dark.
  function setupTheme() {
    var themeBtn = document.getElementById("theme-btn");
    var themes = ["system", "light", "dark"];
    function applyTheme(t) {
      if (t === "system") document.documentElement.removeAttribute("data-theme");
      else document.documentElement.setAttribute("data-theme", t);
      themeBtn.textContent = "Theme: " + t;
    }
    var theme = store("bench.theme");
    if (themes.indexOf(theme) < 0) theme = document.documentElement.getAttribute("data-theme") ? null : "system";
    if (theme) applyTheme(theme); else themeBtn.textContent = "Theme: " + document.documentElement.getAttribute("data-theme");
    themeBtn.addEventListener("click", function () {
      var curT = theme || document.documentElement.getAttribute("data-theme") || "system";
      theme = themes[(themes.indexOf(curT) + 1) % themes.length];
      store("bench.theme", theme);
      applyTheme(theme);
    });
  }

  function start() {
    setupTheme();
    drawHeader();
    SETUP.forEach(function (fn) { fn(); });
    renderAll();
  }
