
  // Worker-role tests: the side-by-side judge score, the decode speed of real sessions, and the memory peak.
  // A run holds this data when its run dir has task_eval.json. The file also lists reference models, which are
  // not runs of this harness. Each reference draws as a neutral column.
  function withTasks(runs) { return runs.filter(function (r) { return r.task_eval && r.task_eval.tests; }); }
  function taskChart(cfg) {
    CHARTS.push({
      order: cfg.order, id: cfg.id, title: cfg.title, sub: cfg.sub,
      when: function (runs) { return withTasks(runs).some(function (r) { return isNum(r.task_eval.tests[cfg.key]); }); },
      draw: function (plot, runs, width) {
        var rs = withTasks(runs).filter(function (r) { return isNum(r.task_eval.tests[cfg.key]); });
        var refs = [], seen = {};
        rs.forEach(function (r) {
          (r.task_eval.reference || []).forEach(function (q) {
            if (isNum(q[cfg.key]) && !seen[q.label]) { seen[q.label] = 1; refs.push(q); }
          });
        });
        var bars = refs.map(function (q) { return { color: q.color || "var(--baseline)", name: q.label, value: q[cfg.key] }; });
        rs.forEach(function (r) { bars.push({ color: r.color, name: r.label, value: r.task_eval.tests[cfg.key] }); });
        var ok = gcol(plot, {
          width: width, aria: cfg.title, yFmt: cfg.fmt, yTick: cfg.tick,
          groups: [{ label: cfg.unit, bars: bars }],
        });
        var items = refs.map(function (q) { return { color: q.color || "var(--baseline)", label: q.label, kind: "rect" }; });
        return {
          ok: ok, legendItems: items.concat(rs.map(function (r) { return { color: r.color, label: legendLabel(r, rs), kind: "rect" }; })), runs: rs,
          cols: ["Model", cfg.unit],
          rows: refs.map(function (q) { return { run: { label: q.label, color: q.color || "var(--baseline)" }, cells: [cfg.fmt(q[cfg.key])] }; })
            .concat(rs.map(function (r) { return { run: r, cells: [cfg.fmt(r.task_eval.tests[cfg.key])] }; })),
        };
      },
    });
  }
  taskChart({
    order: 80, id: "judge", key: "judge_score", unit: "Side-by-side score",
    title: "Side-by-side judged turns",
    sub: "A judge compares a new turn with the recorded turn of the full model, blind, in both orders. Score = (wins + ties / 2) / verdicts. 0.5 means equal.",
    fmt: function (v) { return fmt(v, 2); }, tick: function (v) { return fmt(v, 1); },
  });
  taskChart({
    order: 81, id: "botspeed", key: "decode_tok_s", unit: "Decode, real sessions",
    title: "Decode speed in real sessions",
    sub: "Output tokens per second over 5 recorded sessions of 30K to 73K tokens, with the speculative draft.",
    fmt: function (v) { return fmt(v, 1) + " tok/s"; }, tick: function (v) { return fmt(v, 0); },
  });
  taskChart({
    order: 82, id: "peakmem", key: "peak_gib", unit: "Memory at the peak",
    title: "Memory at the peak, 128K context",
    sub: "Memory used at the peak with images on. Lower is better. About 9 GiB stay free beside the live DeepSeek.",
    fmt: function (v) { return fmt(v, 1) + " GiB"; }, tick: function (v) { return fmt(v, 0); },
  });
