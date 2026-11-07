  // Score heatmaps, the MTP comparison, and the server-side TTFT split.
  // Score grid: rows are runs, cells are 0..1 on the sequential ramp.
  function heat(plot, cfg) {
    if (!cfg.rows.length) return false;
    var wrap = h("div", { class: "scroll" }, plot);
    var t = h("table", { class: "heat" }, wrap);
    var hr = h("tr", null, h("thead", null, t));
    h("th", { scope: "col" }, hr, "Run");
    cfg.cols.forEach(function (c) { h("th", { scope: "col" }, hr, c.label); });
    (cfg.extra || []).forEach(function (c) { h("th", { scope: "col" }, hr, c.label); });
    var tb = h("tbody", null, t);
    cfg.rows.forEach(function (row) {
      var tr = h("tr", null, tb);
      runCell(tr, row.run);
      cfg.cols.forEach(function (c) {
        var v = row.values[c.key];
        var td = h("td", { class: "cell" }, tr, isNum(v) ? fmtPct(v) : "–");
        if (isNum(v)) {
          var k = Math.max(0, Math.min(6, Math.floor(v * 7 - 1e-9)));
          td.style.background = "var(--heat-" + k + ")";
          td.style.color = "var(--heat-ink-" + k + ")";
        } else {
          td.classList.add("na");
        }
        bindTip(td, function () {
          return { title: row.run.label, rows: [{ color: isNum(v) ? "var(--heat-" + Math.max(0, Math.min(6, Math.floor(v * 7 - 1e-9))) + ")" : "var(--grid)", value: isNum(v) ? fmtPct(v) : "no data", label: c.label + (c.hint ? " · " + c.hint : "") }] };
        });
      });
      (cfg.extra || []).forEach(function (c) { h("td", null, tr, c.get(row.run)); });
    });
    return true;
  }
  var TOOL_COLS = [
    { key: "tool_accuracy", label: "Tool choice", hint: "single-tool tasks" },
    { key: "param_completeness", label: "Params complete", hint: "single-tool tasks" },
    { key: "param_correctness", label: "Params correct", hint: "single-tool tasks" },
    { key: "multi_tool", label: "Multi-tool" },
    { key: "schema_compliance", label: "Schema" },
    { key: "refusal", label: "Refusal" },
  ];

  CHARTS.push({
    order: 60,
    id: "spec", title: "MTP speculative decoding",
    sub: "Decode speed with the MTP draft head on and off, from --compare-spec.",
    when: function (runs) { return runs.some(function (r) { return r.spec && r.spec.length; }); },
    draw: function (plot, runs, width) {
      var rs = withData(runs, "spec");
      var xs = xUnion(rs, "spec");
      var ok = gcol(plot, {
        width: width, aria: "Decode speed with and without MTP", yFmt: function (v) { return fmt(v, 1) + " tok/s"; },
        groups: xs.map(function (x) {
          var bars = [];
          rs.forEach(function (r) {
            bars.push({ color: "var(--baseline)", name: r.label + " · MTP off", value: at(r.spec, x, "baseline") });
            bars.push({ color: r.color, name: r.label + " · MTP on", value: at(r.spec, x, "spec") });
          });
          return { label: fmtTokens(x) + " out", bars: bars };
        }),
      });
      var items = [{ color: "var(--baseline)", label: "MTP off", kind: "rect" }].concat(rs.map(function (r) { return { color: r.color, label: r.label + " · MTP on", kind: "rect" }; }));
      return {
        ok: ok, legendItems: items, runs: rs,
        cols: ["Run"].concat(xs.map(function (x) { return fmtTokens(x) + " off"; }), xs.map(function (x) { return fmtTokens(x) + " on"; }), ["Mean speedup"]),
        rows: rs.map(function (r) {
          var ratios = xs.map(function (x) { var a = at(r.spec, x, "spec"), b = at(r.spec, x, "baseline"); return isNum(a) && isNum(b) && b > 0 ? a / b : null; });
          var mr = mean(ratios);
          return { run: r, cells: xs.map(function (x) { return fmt(at(r.spec, x, "baseline"), 1); }).concat(xs.map(function (x) { return fmt(at(r.spec, x, "spec"), 1); }), [isNum(mr) ? fmt(mr, 2) + "×" : "–"]) };
        }),
      };
    },
  });
  CHARTS.push({
    order: 70,
    id: "tools", title: "Tool-calling scores", wide: true,
    sub: "Composite components from 85 tasks. Each cell is a 0–100% score; darker means higher.",
    note: "Runs before the scoring fix used the old scorer, which scored single-tool tasks as 0%.",
    draw: function (plot, runs) {
      var rs = runs.filter(function (r) { return r.tool_calling; });
      var extra = [
        { label: "Composite (of 6)", get: function (r) { return fmt(r.tool_calling.composite, 2); } },
        { label: "Correct", get: function (r) { var t = r.tool_calling; return isNum(t.correct) ? t.correct + " / " + t.total : "–"; } },
      ];
      var ok = heat(plot, { cols: TOOL_COLS, extra: extra, rows: rs.map(function (r) { return { run: r, values: r.tool_calling.components || {} }; }) });
      return {
        ok: ok, legendKind: "none", runs: rs,
        cols: ["Run"].concat(TOOL_COLS.map(function (c) { return c.label; }), extra.map(function (c) { return c.label; })),
        rows: rs.map(function (r) { return { run: r, cells: TOOL_COLS.map(function (c) { return fmtPct((r.tool_calling.components || {})[c.key]); }).concat(extra.map(function (c) { return c.get(r); })) }; }),
      };
    },
  });
  CHARTS.push({
    order: 80,
    id: "categories", title: "Tool-calling pass rate by category", wide: true,
    sub: "Share of tasks passed in each task category. Needs tool_calling.json in the run folder.",
    when: function (runs) { return runs.some(function (r) { return r.tool_calling && r.tool_calling.categories; }); },
    draw: function (plot, runs) {
      var rs = runs.filter(function (r) { return r.tool_calling && r.tool_calling.categories; });
      var names = [];
      rs.forEach(function (r) { Object.keys(r.tool_calling.categories).forEach(function (n) { if (names.indexOf(n) < 0) names.push(n); }); });
      names.sort(function (a, b) { return (a === "single_tool") - (b === "single_tool") || a.localeCompare(b); });
      var cols = names.map(function (n) {
        var total = null;
        rs.forEach(function (r) { var c = r.tool_calling.categories[n]; if (c && isNum(c.total)) total = c.total; });
        return { key: n, label: n.replace(/_/g, " "), hint: total != null ? total + " tasks" : "" };
      });
      var ok = heat(plot, { cols: cols, rows: rs.map(function (r) {
        var v = {}; names.forEach(function (n) { var c = r.tool_calling.categories[n]; v[n] = c ? c.pass_rate : null; });
        return { run: r, values: v };
      }) });
      return {
        ok: ok, legendKind: "none", runs: rs,
        cols: ["Run"].concat(cols.map(function (c) { return c.label; })),
        rows: rs.map(function (r) { return { run: r, cells: names.map(function (n) { var c = r.tool_calling.categories[n]; return c ? fmtPct(c.pass_rate) : "–"; }) }; }),
      };
    },
  });
  CHARTS.push({
    order: 95,
    id: "split", title: "Server-side TTFT split", wide: true, tableOnly: true,
    sub: "vLLM per-request metrics, median of 10 requests: time in the queue, then time after the queue (prefill and the first token). vLLM reports no separate prefill time.",
    when: function (runs) { return runs.some(function (r) { return (r.ttft_breakdown || []).some(function (p) { return isNum(p.queue_median_ms); }); }); },
    draw: function (plot, runs) {
      var rs = runs.filter(function (r) { return (r.ttft_breakdown || []).some(function (p) { return isNum(p.queue_median_ms); }); });
      var xs = xUnion(rs, "ttft_breakdown");
      return {
        ok: true, runs: rs,
        cols: ["Run"].concat(xs.map(function (x) { return fmtTokens(x) + " queue"; }), xs.map(function (x) { return fmtTokens(x) + " after queue"; })),
        rows: rs.map(function (r) {
          return { run: r, cells: xs.map(function (x) { return fmtMs(at(r.ttft_breakdown, x, "queue_median_ms")); }).concat(xs.map(function (x) { return fmtMs(at(r.ttft_breakdown, x, "post_queue_median_ms")); })) };
        }),
      };
    },
  });
