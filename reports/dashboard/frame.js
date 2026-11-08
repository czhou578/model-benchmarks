  // Run filter, chart cards, tooltip, tables, and the all-runs table.
  // ---------- tooltip ----------
  var tip = document.getElementById("tip");
  function showTip(x, y, title, rows) {
    tip.replaceChildren();
    if (title) h("div", { class: "t" }, tip, title);
    rows.forEach(function (row) {
      var r = h("div", { class: "r" }, tip);
      var key = h("i", null, r);
      if (row.color) key.style.background = row.color;
      h("b", null, r, row.value);
      h("span", null, r, row.label || "");
    });
    tip.hidden = false;
    var w = tip.offsetWidth, ht = tip.offsetHeight;
    var left = x + 14, top = y + 14;
    if (left + w > window.innerWidth - 8) left = x - w - 14;
    if (top + ht > window.innerHeight - 8) top = y - ht - 14;
    tip.style.left = Math.max(8, left) + "px";
    tip.style.top = Math.max(8, top) + "px";
  }
  function hideTip() { tip.hidden = true; }
  function bindTip(el, getContent) {
    el.setAttribute("tabindex", "0");
    el.addEventListener("pointermove", function (ev) { var c = getContent(); showTip(ev.clientX, ev.clientY, c.title, c.rows); });
    el.addEventListener("pointerleave", hideTip);
    el.addEventListener("focus", function () {
      var b = el.getBoundingClientRect(); var c = getContent();
      showTip(b.left + b.width / 2, b.top, c.title, c.rows);
    });
    el.addEventListener("blur", hideTip);
  }

  // ---------- legend and tables ----------
  function legend(host, items) {
    host.replaceChildren();
    items.forEach(function (it) {
      var sp = h("span", null, host);
      var k = h("i", { class: it.kind || "line" }, sp);
      k.style.background = it.color;
      sp.appendChild(document.createTextNode(it.label));
    });
  }
  function runCell(tr, r) {
    var td = h("td", { class: "run-cell" }, tr);
    var sp = h("span", null, td);
    var dot = h("i", null, sp); dot.style.background = r.color;
    sp.appendChild(document.createTextNode(r.label));
    return td;
  }
  function table(host, cols, rows) {
    host.replaceChildren();
    var wrap = h("div", { class: "scroll" }, host);
    var t = h("table", null, wrap);
    var hr = h("tr", null, h("thead", null, t));
    cols.forEach(function (c) { h("th", { scope: "col" }, hr, c); });
    var tb = h("tbody", null, t);
    rows.forEach(function (row) {
      var tr = h("tr", null, tb);
      runCell(tr, row.run);
      row.cells.forEach(function (c) { h("td", c === "–" ? { class: "na" } : null, tr, c); });
    });
  }

  // ---------- data helpers ----------
  function xUnion(runs, key) {
    var xs = [];
    runs.forEach(function (r) { (r[key] || []).forEach(function (p) { if (xs.indexOf(p.x) < 0) xs.push(p.x); }); });
    return xs.sort(function (a, b) { return a - b; });
  }
  function at(rows, x, field) {
    var p = (rows || []).filter(function (q) { return q.x === x; })[0];
    return p ? p[field] : null;
  }
  function withData(runs, key) { return runs.filter(function (r) { return r[key] && r[key].length; }); }
  function decodeMean(r) { return mean((r.decode || []).map(function (d) { return d.avg; })); }
  function bestConcurrency(r) {
    var best = null;
    (r.concurrency || []).forEach(function (c) { if (isNum(c.throughput) && (!best || c.throughput > best.throughput)) best = c; });
    return best;
  }

  CHARTS.push({
    order: 100,
    id: "details", title: "All runs", wide: true, tableOnly: true,
    sub: "Key numbers and server settings for each run.",
    draw: function (plot, runs) {
      return {
        ok: true, runs: runs,
        cols: ["Run", "Started", "Status", "vLLM", "Spec", "Prefix cache", "GPU mem", "Decode", "TTFT 2K", "Peak tok/s", "Tool score", "Power avg", "Run energy", "Min free mem"],
        rows: runs.map(function (r) {
          var best = bestConcurrency(r);
          var tc = r.tool_calling;
          return { run: r, cells: [
            fmtDate(r.started), r.status || "–", r.env.vllm || "–",
            r.server.speculative ? "MTP" : "off", r.server.prefix_caching ? "on" : "off",
            r.server.gpu_memory_utilization || "–",
            isNum(decodeMean(r)) ? fmt(decodeMean(r), 1) + " tok/s" : "–",
            fmtMs(at(r.latency, 2048, "ttft_median_ms")),
            best ? fmt(best.throughput, 1) + " @" + best.x : "–",
            tc ? fmt(tc.composite, 2) + (tc.task_set === "lite" ? " (lite)" : "") : "–",
            isNum(r.gpu.power_avg_w) ? fmt(r.gpu.power_avg_w, 1) + " W" : "–",
            isNum(r.gpu.energy_wh) ? fmt(r.gpu.energy_wh, 1) + " Wh" : "–",
            r.memory ? fmt(r.memory.min_available_gib, 1) + " GiB" : "–",
          ] };
        }),
      };
    },
  });

  // ---------- chart cards ----------
  var chartHost = document.getElementById("charts");
  var cardEls = {};
  var tableMode = store("bench.tables") || {};
  function buildCards() {
    CHARTS.sort(byOrder);
    CHARTS.forEach(function (c) {
      var card = h("section", { class: "card" + (c.wide ? " wide" : ""), "aria-labelledby": "h-" + c.id }, chartHost);
      var head = h("div", { class: "card-head" }, card);
      var titles = h("div", null, head);
      h("h2", { id: "h-" + c.id }, titles, c.title);
      h("p", { class: "sub" }, titles, c.sub);
      var btn = null;
      if (!c.tableOnly) {
        btn = h("button", { class: "tbl-btn", type: "button", "aria-pressed": tableMode[c.id] ? "true" : "false" }, head, "Table");
        btn.addEventListener("click", function () {
          tableMode[c.id] = !tableMode[c.id];
          btn.setAttribute("aria-pressed", tableMode[c.id] ? "true" : "false");
          store("bench.tables", tableMode);
          drawChart(c);
        });
      }
      var leg = h("div", { class: "legend" }, card);
      var plot = h("div", { class: "plot" }, card);
      var tbl = h("div", null, card);
      var note = c.note ? h("p", { class: "note" }, card, c.note) : null;
      cardEls[c.id] = { card: card, legend: leg, plot: plot, table: tbl, note: note };
    });
  }
  function drawChart(c) {
    var el = cardEls[c.id];
    var runs = shown();
    var applies = !c.when || c.when(RUNS);
    el.card.hidden = !applies;
    if (!applies) return;
    el.plot.replaceChildren(); el.table.replaceChildren(); el.legend.replaceChildren();
    hideTip();
    var asTable = c.tableOnly || tableMode[c.id];
    var width = Math.max(260, Math.floor(el.plot.clientWidth || el.card.clientWidth - 32));
    var res = c.draw(asTable ? document.createElement("div") : el.plot, runs, width);
    if (!res.runs.length) {
      el.plot.replaceChildren();
      h("p", { class: "empty" }, el.plot, "No shown run has this data. Turn on more runs above.");
      el.legend.hidden = true;
      return;
    }
    if (asTable) {
      el.plot.hidden = true; el.table.hidden = false;
      table(el.table, res.cols, res.rows);
    } else {
      el.plot.hidden = false; el.table.hidden = true;
    }
    var items = res.legendItems || res.runs.map(function (r) { return { color: r.color, label: legendLabel(r, res.runs), kind: res.legendKind || "line" }; });
    el.legend.hidden = res.legendKind === "none" || c.tableOnly || items.length < 2;
    if (!el.legend.hidden) legend(el.legend, items);
    if (el.note) el.note.hidden = !runs.some(function (r) { return r.pre_fix; });
  }
  function drawCharts() { CHARTS.forEach(drawChart); }
  function watchWidth() {
    var lastW = chartHost.clientWidth, timer = null;
    if (window.ResizeObserver) {
      new ResizeObserver(function () {
        if (Math.abs(chartHost.clientWidth - lastW) < 2) return;
        lastW = chartHost.clientWidth;
        clearTimeout(timer);
        timer = setTimeout(function () { CHARTS.forEach(drawChart); }, 120);
      }).observe(chartHost);
    }
  }
  SETUP.push(buildCards, watchWidth);
  RENDERERS.push({ order: 50, fn: drawCharts });

  // ---------- filter and callout ----------
  function drawFilters() {
    var host = document.getElementById("filters");
    host.replaceChildren();
    h("span", { class: "label" }, host, "Runs");
    RUNS.forEach(function (r) {
      var b = h("button", { class: "chip", type: "button", "aria-pressed": visible.has(r.id) ? "true" : "false", title: r.id }, host);
      var dot = h("span", { class: "dot" }, b); dot.style.background = r.color;
      h("span", { class: "name" }, b, r.label);
      b.addEventListener("click", function () {
        if (visible.has(r.id)) visible.delete(r.id); else visible.add(r.id);
        store("bench.visible", Array.from(visible));
        renderAll();
      });
    });
    if (RUNS.length > 1) {
      var all = h("button", { class: "chip chip-plain", type: "button" }, host, "Show all");
      all.addEventListener("click", function () { RUNS.forEach(function (r) { visible.add(r.id); }); store("bench.visible", Array.from(visible)); renderAll(); });
      var latest = h("button", { class: "chip chip-plain", type: "button" }, host, "Newest only");
      latest.addEventListener("click", function () { visible = new Set([RUNS[RUNS.length - 1].id]); store("bench.visible", Array.from(visible)); renderAll(); });
    }
  }
  function drawCallout() {
    var pre = shown().filter(function (r) { return r.pre_fix; });
    var box = document.getElementById("callout");
    box.hidden = !pre.length;
    if (pre.length) {
      document.getElementById("callout-text").textContent =
        pre.length + " shown run" + (pre.length > 1 ? "s were" : " was") + " made before the " + fmtDate(B.scoring_fix_date + "T00:00").split(",")[0] +
        " fix to the cache salt and the tool-calling scorer. Their TTFT, long-context, and tool-calling numbers are not comparable with newer runs. Decode, concurrency, and reasoning numbers are not affected.";
    }
  }
  RENDERERS.push({ order: 10, fn: drawFilters }, { order: 30, fn: drawCallout });
