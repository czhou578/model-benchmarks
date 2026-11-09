  // SVG helpers, bar and column charts, and the charts that use them.
  function s(tag, attrs, parent) {
    var e = document.createElementNS(NS, tag);
    if (attrs) for (var k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }
  function fmtTick(v) {
    if (Math.abs(v) >= 10000) return fmt(v / 1000, v % 1000 ? 1 : 0) + "K";
    return fmt(v, v % 1 ? 1 : 0);
  }
  function niceTicks(max, count) {
    if (!isNum(max) || max <= 0) max = 1;
    var raw = max / (count || 4);
    var mag = Math.pow(10, Math.floor(Math.log10(raw)));
    var steps = [1, 2, 2.5, 5, 10];
    var step = mag;
    for (var i = 0; i < steps.length; i++) { if (steps[i] * mag >= raw) { step = steps[i] * mag; break; } }
    var ticks = [];
    for (var t = 0; t <= max + step * 0.001; t += step) ticks.push(+t.toFixed(10));
    if (ticks[ticks.length - 1] < max) ticks.push(+(ticks[ticks.length - 1] + step).toFixed(10));
    return ticks;
  }

  function roundedBarH(x0, y, w, hgt, r) {
    r = Math.min(r, w, hgt / 2);
    return "M" + x0 + " " + y + "H" + (x0 + w - r) + "A" + r + " " + r + " 0 0 1 " + (x0 + w) + " " + (y + r) +
      "V" + (y + hgt - r) + "A" + r + " " + r + " 0 0 1 " + (x0 + w - r) + " " + (y + hgt) + "H" + x0 + "Z";
  }
  function roundedBarV(x, y0, w, hgt, r) {
    r = Math.min(r, hgt, w / 2);
    var top = y0 - hgt;
    return "M" + x + " " + y0 + "V" + (top + r) + "A" + r + " " + r + " 0 0 1 " + (x + r) + " " + top +
      "H" + (x + w - r) + "A" + r + " " + r + " 0 0 1 " + (x + w) + " " + (top + r) + "V" + y0 + "Z";
  }
  // One bar per run, name above, value at the tip.
  function hbar(plot, cfg) {
    var rows = cfg.rows.filter(function (r) { return isNum(r.value); });
    if (!rows.length) return false;
    var W = cfg.width, rowH = 44, H = rows.length * rowH + 4;
    var valueRoom = 84, iw = Math.max(40, W - valueRoom);
    var max = Math.max.apply(null, rows.map(function (r) { return r.value; }));
    var svg = s("svg", { viewBox: "0 0 " + W + " " + H, height: H, role: "img", "aria-label": cfg.aria || "" }, plot);
    s("line", { x1: 0.5, x2: 0.5, y1: 0, y2: H, "stroke-width": 1 }, svg).style.stroke = "var(--axis)";
    rows.forEach(function (r, i) {
      var y = i * rowH;
      var g = s("g", null, svg);
      var name = s("text", { x: 6, y: y + 14, "font-size": 12 }, g);
      name.style.fill = "var(--ink-2)"; name.textContent = r.label;
      var w = Math.max(2, r.value / max * iw);
      var bar = s("path", { d: roundedBarH(0, y + 20, w, 16, 4) }, g);
      bar.style.fill = r.color;
      var val = s("text", { x: w + 8, y: y + 32, "font-size": 12, "font-weight": 600 }, g);
      val.style.fill = "var(--ink)"; val.textContent = cfg.fmt(r.value);
      var hitR = s("rect", { x: 0, y: y, width: W, height: rowH, fill: "transparent" }, g);
      bindTip(hitR, function () { return { title: r.label, rows: r.tip }; });
      hitR.addEventListener("pointerenter", function () { bar.style.opacity = "0.82"; });
      hitR.addEventListener("pointerleave", function () { bar.style.opacity = ""; });
    });
    return true;
  }
  // Columns grouped by x, one column per run.
  function gcol(plot, cfg) {
    var groups = cfg.groups.filter(function (g) { return g.bars.some(function (b) { return isNum(b.value); }); });
    if (!groups.length) return false;
    var W = cfg.width, H = cfg.height || 240;
    var m = { l: 52, r: 12, t: 12, b: 34 };
    var iw = Math.max(40, W - m.l - m.r), ih = H - m.t - m.b;
    var max = 0;
    groups.forEach(function (g) { g.bars.forEach(function (b) { if (isNum(b.value)) max = Math.max(max, b.value); }); });
    var yt = niceTicks(max, 4), ytop = yt[yt.length - 1];
    var fy = function (v) { return m.t + ih - v / ytop * ih; };
    var svg = s("svg", { viewBox: "0 0 " + W + " " + H, height: H, role: "img", "aria-label": cfg.aria || "" }, plot);
    yt.forEach(function (t) {
      s("line", { x1: m.l, x2: m.l + iw, y1: fy(t), y2: fy(t), "stroke-width": 1 }, svg).style.stroke = t === 0 ? "var(--axis)" : "var(--grid)";
      var tx = s("text", { x: m.l - 8, y: fy(t) + 4, "text-anchor": "end", "font-size": 11 }, svg);
      tx.style.fill = "var(--muted)"; tx.textContent = (cfg.yTick || fmtTick)(t);
    });
    var gw = iw / groups.length;
    groups.forEach(function (g, gi) {
      var bars = g.bars;
      var n = bars.length, gap = 2;
      var bw = Math.max(4, Math.min(24, (gw * 0.72 - gap * (n - 1)) / n));
      var total = n * bw + gap * (n - 1);
      var start = m.l + gi * gw + (gw - total) / 2;
      bars.forEach(function (b, bi) {
        if (!isNum(b.value)) return;
        var x = start + bi * (bw + gap);
        var hgt = Math.max(1, fy(0) - fy(b.value));
        var p = s("path", { d: roundedBarV(x, fy(0), bw, hgt, 4) }, svg);
        p.style.fill = b.color;
        var hitR = s("rect", { x: x - gap / 2, y: m.t, width: bw + gap, height: ih, fill: "transparent" }, svg);
        bindTip(hitR, function () { return { title: g.label, rows: [{ color: b.color, value: cfg.yFmt(b.value), label: b.name }].concat(b.extra || []) }; });
        hitR.addEventListener("pointerenter", function () { p.style.opacity = "0.82"; });
        hitR.addEventListener("pointerleave", function () { p.style.opacity = ""; });
      });
      var tx = s("text", { x: m.l + gi * gw + gw / 2, y: m.t + ih + 20, "text-anchor": "middle", "font-size": 11 }, svg);
      tx.style.fill = "var(--muted)"; tx.textContent = g.label;
    });
    return true;
  }

  // A peak is a token rate only when the server sent per-chunk usage.
  // An event_gap peak counts stream events, not tokens, so it is hidden.
  function tokenPeak(d) { return d.rate_method === "token_delta" && isNum(d.peak); }

  CHARTS.push({
    order: 10,
    id: "decode", title: "Decode speed",
    sub: "Mean output tokens per second for one request, over 512, 1K and 2K tokens.",
    draw: function (plot, runs, width) {
      var rs = withData(runs, "decode");
      var xs = xUnion(rs, "decode");
      var ok = hbar(plot, {
        width: width, fmt: function (v) { return fmt(v, 1) + " tok/s"; }, aria: "Decode speed by run",
        rows: rs.map(function (r) {
          return {
            label: r.label, color: r.color, value: decodeMean(r),
            tip: r.decode.map(function (d) { return { color: r.color, value: fmt(d.avg, 1) + " tok/s", label: fmtTokens(d.x) + " out" + (tokenPeak(d) ? " · peak " + fmt(d.peak, 1) : "") }; }),
          };
        }),
      });
      return {
        ok: ok, legendKind: "rect", runs: rs,
        cols: ["Run"].concat(xs.map(function (x) { return fmtTokens(x) + " tok/s"; }), ["Mean", "Peak"]),
        rows: rs.map(function (r) {
          var peaks = r.decode.filter(tokenPeak).map(function (d) { return d.peak; });
          return { run: r, cells: xs.map(function (x) { return fmt(at(r.decode, x, "avg"), 1); }).concat([fmt(decodeMean(r), 1), peaks.length ? fmt(Math.max.apply(null, peaks), 1) : "–"]) };
        }),
      };
    },
  });
  CHARTS.push({
    order: 40,
    id: "deep", title: "Long-context TTFT",
    sub: "Median TTFT for one request with a 32K or 64K token prompt.",
    note: "Runs before the scoring fix can include prefix-cache hits.",
    draw: function (plot, runs, width) {
      var rs = withData(runs, "deep_context");
      var xs = xUnion(rs, "deep_context");
      var ok = gcol(plot, {
        width: width, aria: "Long-context TTFT", yTick: function (v) { return fmtTick(v / 1000) + " s"; }, yFmt: fmtMs,
        groups: xs.map(function (x) {
          return {
            label: fmtTokens(x) + " tokens",
            bars: rs.map(function (r) {
              return { color: r.color, name: r.label, value: at(r.deep_context, x, "ttft_median_ms"),
                extra: [{ value: fmt(at(r.deep_context, x, "prefill_tps")) + " tok/s", label: "prefill rate" }] };
            }),
          };
        }),
      });
      return {
        ok: ok, legendKind: "rect", runs: rs,
        cols: ["Run"].concat(xs.map(function (x) { return fmtTokens(x) + " TTFT"; }), xs.map(function (x) { return fmtTokens(x) + " status"; })),
        rows: rs.map(function (r) {
          return { run: r, cells: xs.map(function (x) { return fmtMs(at(r.deep_context, x, "ttft_median_ms")); }).concat(xs.map(function (x) { return at(r.deep_context, x, "status") || "–"; })) };
        }),
      };
    },
  });
  CHARTS.push({
    order: 50,
    id: "reasoning", title: "Thinking tokens",
    sub: "Mean reasoning tokens per prompt, capped at 1,024 output tokens.",
    draw: function (plot, runs, width) {
      var rs = runs.filter(function (r) { return r.reasoning && isNum(r.reasoning.thinking_avg); });
      var ok = hbar(plot, {
        width: width, fmt: function (v) { return fmt(v) + " tok"; }, aria: "Thinking tokens by run",
        rows: rs.map(function (r) {
          var q = r.reasoning;
          return { label: r.label, color: r.color, value: q.thinking_avg,
            tip: [
              { color: r.color, value: fmt(q.thinking_avg), label: "thinking tokens, mean" },
              { value: fmt(q.thinking_max), label: "thinking tokens, max" },
              { value: fmt(q.answer_avg, 1), label: "answer tokens, mean" },
            ] };
        }),
      });
      return {
        ok: ok, legendKind: "rect", runs: rs,
        cols: ["Run", "Thinking mean", "Thinking median", "Thinking max", "Answer mean", "Prompts"],
        rows: rs.map(function (r) { var q = r.reasoning; return { run: r, cells: [fmt(q.thinking_avg), fmt(q.thinking_median, 1), fmt(q.thinking_max), fmt(q.answer_avg, 1), fmt(q.prompts)] }; }),
      };
    },
  });
