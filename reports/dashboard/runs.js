  // One tinted summary card per shown run.
  function badge(host, kind, text) {
    var b = h("span", { class: "badge" + (kind ? "" : " plain") }, host);
    if (kind) h("span", { class: "icon " + kind, "aria-hidden": "true" }, b, kind === "good" ? "✓" : kind === "critical" ? "✕" : "!");
    b.appendChild(document.createTextNode(text));
  }
  function stat(dl, label, value, unit) {
    var d = h("div", null, dl);
    h("dt", null, d, label);
    var dd = h("dd", null, d, value);
    if (unit && value !== "–") h("small", null, dd, unit);
  }
  function drawRunCards() {
    var host = document.getElementById("runs");
    host.replaceChildren();
    var runs = shown();
    if (!runs.length) { h("p", { class: "empty" }, host, "No runs shown. Pick a run above."); return; }
    runs.forEach(function (r) {
      var card = h("article", { class: "run", "aria-label": r.label }, host);
      card.style.setProperty("--run", r.color);
      var head = h("div", { class: "run-head" }, card);
      h("span", { class: "swatch", "aria-hidden": "true" }, head, String(r.idx + 1));
      var t = h("div", { style: "min-width:0" }, head);
      h("h3", { class: "run-title" }, t, r.model);
      h("p", { class: "run-meta" }, t, fmtDate(r.started) + " · vLLM " + (r.env.vllm || "?"));
      var badges = h("div", { class: "badges" }, card);
      if (r.status === "completed") badge(badges, "good", "Completed");
      else if (r.status === "failed") badge(badges, "critical", "Failed");
      else badge(badges, "warning", r.status || "Unknown");
      badge(badges, null, r.server.speculative ? "MTP spec on" : "No spec decode");
      if (r.pre_fix) badge(badges, "warning", "Pre-fix scoring");
      if (r.memory && r.memory.stopped_for_memory) badge(badges, "critical", "Stopped for memory");
      var dl = h("dl", { class: "stats" }, card);
      var dm = decodeMean(r);
      stat(dl, "Decode", isNum(dm) ? fmt(dm, 1) : "–", "tok/s");
      var t2k = at(r.latency, 2048, "ttft_median_ms");
      stat(dl, "TTFT at 2K", isNum(t2k) ? fmt(t2k) : "–", "ms");
      var best = bestConcurrency(r);
      stat(dl, best ? "Peak throughput ×" + best.x : "Peak throughput", best ? fmt(best.throughput) : "–", "tok/s");
      var tc = r.tool_calling;
      stat(dl, "Tool tasks passed", tc && isNum(tc.correct) ? String(tc.correct) : "–", tc ? "of " + tc.total : "");
      if (r.memory) stat(dl, "Lowest free memory", fmt(r.memory.min_available_gib, 1), "GiB");
      if (r.gpu && isNum(r.gpu.power_avg_w)) stat(dl, "GPU power, mean", fmt(r.gpu.power_avg_w, 1), "W");
      if (r.error) h("p", { class: "served" }, card, r.error);
      if (r.server.served_model) h("p", { class: "served" }, card, r.server.served_model);
    });
  }
  RENDERERS.push({ order: 40, fn: drawRunCards });
