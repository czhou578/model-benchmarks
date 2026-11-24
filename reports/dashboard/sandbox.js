
  // Agent results of the sandbox step (docs/sandbox.md): the rates of each task set with their 95% intervals, and a
  // table of the stops, the times and the prefix cache. The held-out set comes first, because it is the main score.
  var SET_ORDER = ["heldout", "dev"];
  function withSandbox(runs) { return runs.filter(function (r) { return r.sandbox && Object.keys(r.sandbox.sets || {}).length; }); }
  function sandboxSets(rs) {
    var names = [];
    rs.forEach(function (r) { Object.keys(r.sandbox.sets).forEach(function (n) { if (n !== "all" && names.indexOf(n) < 0) names.push(n); }); });
    var rank = function (n) { var i = SET_ORDER.indexOf(n); return i < 0 ? 99 : i; };
    return names.sort(function (a, b) { return rank(a) - rank(b) || a.localeCompare(b); });
  }
  function pctText(v) { return isNum(v) ? fmt(v, 0) + "%" : "–"; }
  function rangeText(q) { return q && isNum(q.lo) ? pctText(q.lo) + " to " + pctText(q.hi) : "–"; }
  function rateOf(r, set, key) { var s = r.sandbox.sets[set]; return (s && s[key]) || {}; }
  function sandboxChart(cfg) {
    CHARTS.push({
      order: cfg.order, id: cfg.id, title: cfg.title, sub: cfg.sub,
      when: function (runs) { return withSandbox(runs).length > 0; },
      draw: function (plot, runs, width) {
        var rs = withSandbox(runs), sets = sandboxSets(rs);
        var ok = gcol(plot, {
          width: width, aria: cfg.title, yFmt: pctText, yTick: function (v) { return fmt(v, 0) + "%"; },
          groups: sets.map(function (n) {
            return { label: n, bars: rs.map(function (r) {
              var q = rateOf(r, n, cfg.key), s = r.sandbox.sets[n] || {};
              return { color: r.color, name: r.label, value: q.rate, lo: q.lo, hi: q.hi,
                       extra: [{ color: "var(--grid)", value: rangeText(q), label: "95% interval, " + (s.tasks || "?") + " tasks, " + (s.runs || "?") + " runs" }] };
            }) };
          }),
        });
        return {
          ok: ok, legendKind: "rect", runs: rs,
          cols: ["Run"].concat(sets.map(function (n) { return n; }), sets.map(function (n) { return n + " 95% interval"; })),
          rows: rs.map(function (r) {
            return { run: r, cells: sets.map(function (n) { return pctText(rateOf(r, n, cfg.key).rate); })
              .concat(sets.map(function (n) { return rangeText(rateOf(r, n, cfg.key)); })) };
          }),
        };
      },
    });
  }
  sandboxChart({
    order: 74, id: "agentpass", key: "pass", title: "Agent tasks passed",
    sub: "Sandbox step: the share of task runs that pass the hidden tests, for each task set. The whisker is the 95% interval from a bootstrap over the tasks. The held-out set is the main score.",
  });
  sandboxChart({
    order: 75, id: "agenthalluc", key: "hallucination", title: "Agent hallucinations",
    sub: "Sandbox step: the share of task runs with an event H1 to H4: an invalid tool call, false evidence, a false test result or a changed test, or a trap reported as done. Lower is better.",
  });
  CHARTS.push({
    order: 96, id: "agentruns", title: "Sandbox runs", wide: true, tableOnly: true,
    sub: "One row for each task set. Cache hits: the measured prefix-cache hit rate, and the best rate for the block size of the server. Only a measured rate below the best rate shows a problem.",
    when: function (runs) { return withSandbox(runs).length > 0; },
    draw: function (plot, runs) {
      var rs = withSandbox(runs), rows = [];
      rs.forEach(function (r) {
        var sb = r.sandbox;
        sandboxSets([r]).forEach(function (n) {
          var s = sb.sets[n], stops = Object.keys(s.stops || {}).sort().map(function (k) { return k + " " + s.stops[k]; }).join(", ");
          rows.push({ run: r, cells: [n, fmt(s.tasks), fmt(s.runs), fmt(s.median_min, 1), fmt(s.median_tokens), stops || "–",
            pctText(sb.hit_rate) + " / " + pctText(sb.best_rate), fmt(sb.agents), (sb.commits || {}).agent_eval || "–"] });
        });
      });
      return { ok: true, runs: rs, rows: rows,
        cols: ["Run", "Set", "Tasks", "Runs", "Median min", "Median tokens", "Stops", "Cache hits / best", "Agents", "agent-eval"] };
    },
  });
