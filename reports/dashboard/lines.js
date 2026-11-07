  // Line chart with a crosshair tooltip, and the charts that use it.
  function lineChart(plot, cfg) {
    var W = cfg.width, H = cfg.height || 250;
    var m = { l: 52, r: 18, t: 12, b: 34 };
    var iw = Math.max(40, W - m.l - m.r), ih = H - m.t - m.b;
    var series = cfg.series.filter(function (sr) { return sr.points.some(function (p) { return isNum(p.y); }); });
    var xs = [];
    series.forEach(function (sr) { sr.points.forEach(function (p) { if (xs.indexOf(p.x) < 0) xs.push(p.x); }); });
    xs.sort(function (a, b) { return a - b; });
    if (!xs.length) return false;
    var x0 = cfg.xMin != null ? cfg.xMin : xs[0], x1 = xs[xs.length - 1];
    var fx = cfg.xLog
      ? function (v) { return m.l + (Math.log2(v) - Math.log2(x0)) / ((Math.log2(x1) - Math.log2(x0)) || 1) * iw; }
      : function (v) { return m.l + (v - x0) / ((x1 - x0) || 1) * iw; };
    var ymax = 0;
    series.forEach(function (sr) { sr.points.forEach(function (p) { if (isNum(p.y)) ymax = Math.max(ymax, p.y); }); });
    (cfg.refs || []).forEach(function (r) { ymax = Math.max(ymax, r.y); });
    if (isNum(cfg.yMax)) ymax = Math.max(ymax, cfg.yMax);
    var yt = niceTicks(ymax, 4), ytop = yt[yt.length - 1];
    var fy = function (v) { return m.t + ih - v / ytop * ih; };

    var svg = s("svg", { viewBox: "0 0 " + W + " " + H, height: H, role: "img", "aria-label": cfg.aria || "" }, plot);
    yt.forEach(function (t) {
      s("line", { x1: m.l, x2: m.l + iw, y1: fy(t), y2: fy(t), "stroke-width": 1 }, svg).style.stroke = t === 0 ? "var(--axis)" : "var(--grid)";
      var tx = s("text", { x: m.l - 8, y: fy(t) + 4, "text-anchor": "end", "font-size": 11 }, svg);
      tx.style.fill = "var(--muted)"; tx.textContent = (cfg.yTick || fmtTick)(t);
    });
    var xticks = cfg.xTicks || (cfg.xLog ? xs : niceTicks(x1, Math.max(2, Math.floor(iw / 90))));
    var minGap = 30, lastX = -1e9;
    xticks.forEach(function (t) {
      if (t < x0 || t > x1 * 1.0001) return;
      var px = fx(t);
      if (px - lastX < minGap) return;
      lastX = px;
      var tx = s("text", { x: px, y: m.t + ih + 20, "text-anchor": "middle", "font-size": 11 }, svg);
      tx.style.fill = "var(--muted)"; tx.textContent = (cfg.xTick || fmtTick)(t);
    });
    if (cfg.xLabel) {
      var xl = s("text", { x: m.l + iw, y: H - 1, "text-anchor": "end", "font-size": 11 }, svg);
      xl.style.fill = "var(--muted)"; xl.textContent = cfg.xLabel;
    }
    (cfg.refs || []).forEach(function (r) {
      var ln = s("line", { x1: m.l, x2: m.l + iw, y1: fy(r.y), y2: fy(r.y), "stroke-width": 1.5 }, svg);
      ln.style.stroke = r.color;
      var left = r.anchor === "start";
      var tx = s("text", { x: left ? m.l + 6 : m.l + iw, y: fy(r.y) - 6, "text-anchor": left ? "start" : "end", "font-size": 11 }, svg);
      tx.style.fill = "var(--ink-2)"; tx.textContent = r.label;
    });
    series.forEach(function (sr) {
      var d = "", pen = false, area = "", first = null, last = null;
      sr.points.forEach(function (p) {
        if (!isNum(p.y)) { pen = false; return; }
        d += (pen ? "L" : "M") + fx(p.x).toFixed(1) + " " + fy(p.y).toFixed(1);
        pen = true;
        if (!first) first = p; last = p;
      });
      if (cfg.area && first) {
        area = d + "L" + fx(last.x).toFixed(1) + " " + fy(0) + "L" + fx(first.x).toFixed(1) + " " + fy(0) + "Z";
        var a = s("path", { d: area }, svg); a.style.fill = sr.color; a.style.fillOpacity = "0.10";
      }
      var path = s("path", { d: d, fill: "none", "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round" }, svg);
      path.style.stroke = sr.color;
      if (sr.points.length <= 40) {
        sr.points.forEach(function (p) {
          if (!isNum(p.y)) return;
          var c = s("circle", { cx: fx(p.x), cy: fy(p.y), r: 4, "stroke-width": 2 }, svg);
          c.style.fill = sr.color; c.style.stroke = "var(--surface)";
        });
      }
    });

    // Crosshair and one tooltip for every series at the nearest x.
    var hover = s("g", { "pointer-events": "none" }, svg);
    var hit = s("rect", { x: m.l, y: m.t, width: iw, height: ih, fill: "transparent", tabindex: 0 }, svg);
    hit.style.cursor = "crosshair";
    var cur = Math.floor(xs.length / 2);
    function nearest(pts, xv) {
      var best = null, bd = Infinity;
      pts.forEach(function (p) { if (!isNum(p.y)) return; var dd = Math.abs(p.x - xv); if (dd < bd) { bd = dd; best = p; } });
      return best;
    }
    function show(i, cx, cy) {
      cur = Math.max(0, Math.min(xs.length - 1, i));
      var xv = xs[cur], px = fx(xv);
      hover.replaceChildren();
      s("line", { x1: px, x2: px, y1: m.t, y2: m.t + ih, "stroke-width": 1 }, hover).style.stroke = "var(--ink-2)";
      var rows = [];
      series.forEach(function (sr) {
        var p = nearest(sr.points, xv);
        if (!p) return;
        if (cfg.exactX && p.x !== xv) return;
        var c = s("circle", { cx: fx(p.x), cy: fy(p.y), r: 5, "stroke-width": 2 }, hover);
        c.style.fill = sr.color; c.style.stroke = "var(--surface)";
        rows.push({ y: p.y, color: sr.color, value: (cfg.yFmt || fmt)(p.y), label: sr.label });
      });
      rows.sort(function (a, b) { return b.y - a.y; });
      var title = (cfg.xFmt || fmtTick)(xv);
      if (cx == null) { var b = svg.getBoundingClientRect(); cx = b.left + px * b.width / W; cy = b.top + m.t; }
      showTip(cx, cy, title, rows);
    }
    function idxAt(ev) {
      var b = svg.getBoundingClientRect();
      var px = (ev.clientX - b.left) * W / b.width;
      var best = 0, bd = Infinity;
      xs.forEach(function (xv, i) { var dd = Math.abs(fx(xv) - px); if (dd < bd) { bd = dd; best = i; } });
      return best;
    }
    hit.addEventListener("pointermove", function (ev) { show(idxAt(ev), ev.clientX, ev.clientY); });
    hit.addEventListener("pointerleave", function () { hover.replaceChildren(); hideTip(); });
    hit.addEventListener("focus", function () { show(cur); });
    hit.addEventListener("blur", function () { hover.replaceChildren(); hideTip(); });
    hit.addEventListener("keydown", function (ev) {
      if (ev.key === "ArrowRight") { show(cur + 1); ev.preventDefault(); }
      if (ev.key === "ArrowLeft") { show(cur - 1); ev.preventDefault(); }
    });
    return true;
  }

  CHARTS.push({
    order: 20,
    id: "ttft", title: "Time to first token",
    sub: "Median TTFT by prompt length, 10 requests per length, salted to skip the prefix cache.",
    note: "Runs before the scoring fix can include prefix-cache hits.",
    draw: function (plot, runs, width) {
      var rs = withData(runs, "latency");
      var xs = xUnion(rs, "latency");
      var ok = lineChart(plot, {
        width: width, xLog: true, exactX: true, aria: "Median TTFT by prompt length",
        xTick: fmtTokens, xFmt: function (x) { return fmtTokens(x) + " prompt tokens"; }, xLabel: "prompt tokens",
        yTick: function (v) { return fmt(v) + " ms"; }, yFmt: fmtMs,
        series: rs.map(function (r) { return { label: r.label, color: r.color, points: r.latency.map(function (p) { return { x: p.x, y: p.ttft_median_ms }; }) }; }),
      });
      return {
        ok: ok, runs: rs,
        cols: ["Run"].concat(xs.map(function (x) { return fmtTokens(x); })),
        rows: rs.map(function (r) { return { run: r, cells: xs.map(function (x) { return fmtMs(at(r.latency, x, "ttft_median_ms")); }) }; }),
      };
    },
  });
  CHARTS.push({
    order: 30,
    id: "concurrency", title: "Throughput under load",
    sub: "Aggregate output tokens per second, 16 requests of 256 tokens per level.",
    draw: function (plot, runs, width) {
      var rs = withData(runs, "concurrency");
      var xs = xUnion(rs, "concurrency");
      var ok = lineChart(plot, {
        width: width, xLog: true, exactX: true, aria: "Throughput by concurrency",
        xTick: function (x) { return String(x); }, xFmt: function (x) { return x + " concurrent requests"; }, xLabel: "concurrent requests",
        yFmt: function (v) { return fmt(v, 1) + " tok/s"; },
        series: rs.map(function (r) { return { label: r.label, color: r.color, points: r.concurrency.map(function (p) { return { x: p.x, y: p.throughput }; }) }; }),
      });
      return {
        ok: ok, runs: rs,
        cols: ["Run"].concat(xs.map(function (x) { return "×" + x + " tok/s"; }), xs.length ? ["TTFT ×" + xs[xs.length - 1]] : []),
        rows: rs.map(function (r) {
          return { run: r, cells: xs.map(function (x) { return fmt(at(r.concurrency, x, "throughput"), 1); }).concat(xs.length ? [fmtMs(at(r.concurrency, xs[xs.length - 1], "ttft_median_ms"))] : []) };
        }),
      };
    },
  });
  CHARTS.push({
    order: 90,
    id: "memory", title: "Memory headroom", wide: true,
    sub: "Available unified memory (MemAvailable) during the run, sampled by memory_guard.py.",
    when: function (runs) { return runs.some(function (r) { return r.memory; }); },
    draw: function (plot, runs, width) {
      var rs = runs.filter(function (r) { return r.memory; });
      var floor = null, warn = null, total = 0;
      rs.forEach(function (r) {
        if (isNum(r.memory.floor_gib)) floor = r.memory.floor_gib;
        if (isNum(r.memory.warn_gib)) warn = r.memory.warn_gib;
        total = Math.max(total, r.memory.total_gib || 0);
      });
      var refs = [];
      if (isNum(warn)) refs.push({ y: warn, color: "var(--warning)", label: "Guard warns below " + fmt(warn) + " GiB", anchor: "start" });
      if (isNum(floor)) refs.push({ y: floor, color: "var(--critical)", label: "Guard stops the run below " + fmt(floor) + " GiB" });
      var ok = lineChart(plot, {
        width: width, height: 260, area: true, xMin: 0, yMax: total, refs: refs, aria: "Available memory over time",
        xTick: function (v) { return fmt(v) + " min"; }, xFmt: function (v) { return fmt(v, 1) + " min into the run"; },
        yTick: function (v) { return fmt(v) + " GiB"; }, yFmt: function (v) { return fmt(v, 1) + " GiB free"; },
        series: rs.map(function (r) { return { label: r.label, color: r.color, points: r.memory.points.map(function (p) { return { x: p[0], y: p[1] }; }) }; }),
      });
      return {
        ok: ok, runs: rs,
        cols: ["Run", "Total", "Minimum free", "Guard floor", "Duration", "Stopped by guard"],
        rows: rs.map(function (r) {
          var q = r.memory;
          return { run: r, cells: [fmt(q.total_gib, 1) + " GiB", fmt(q.min_available_gib, 1) + " GiB", isNum(q.floor_gib) ? fmt(q.floor_gib) + " GiB" : "–", fmt(q.duration_min) + " min", q.stopped_for_memory ? "yes" : "no"] };
        }),
      };
    },
  });
