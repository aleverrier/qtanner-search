(() => {
  "use strict";

  // ---------------- utils ----------------
  const escHtml = (s) =>
    String(s ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");

  const toInt = (x) => {
    if (x === null || x === undefined) return null;
    if (typeof x === "string") {
      const t = x.trim();
      if (!t) return null;
      x = t;
    }
    const n = Number(x);
    return Number.isFinite(n) ? Math.trunc(n) : null;
  };

  const getFirst = (obj, paths) => {
    for (const path of paths) {
      const parts = path.split(".");
      let cur = obj;
      let ok = true;
      for (const p of parts) {
        if (cur && typeof cur === "object" && p in cur) cur = cur[p];
        else { ok = false; break; }
      }
      if (ok && cur !== null && cur !== undefined) return cur;
    }
    return null;
  };

  const parseSuffix = (codeId, key) => {
    const re = new RegExp(`${key}(\\d+)(?:_|$)`);
    const m = String(codeId).match(re);
    return m ? parseInt(m[1], 10) : null;
  };

  // ---------------- group display ----------------
  function groupRawFromCodeId(codeId) {
    const s = String(codeId);
    const m = s.match(/^(SmallGroup[_\(\s]*\d+[, _]+\d+\)?)(?:__|_)/i);
    if (m) return m[1];
    const i = s.indexOf("_");
    return i >= 0 ? s.slice(0, i) : s;
  }

  function groupRawValue(groupField, codeId) {
    if (groupField === null || groupField === undefined) return groupRawFromCodeId(codeId);
    if (typeof groupField === "string") return groupField;
    if (typeof groupField === "object") {
      if (typeof groupField.spec === "string") return groupField.spec;
      if (typeof groupField.id === "string") return groupField.id;
      if (typeof groupField.name === "string") return groupField.name;
      if (typeof groupField.group === "string") return groupField.group;
    }
    return String(groupField);
  }

  function groupDisplay(raw) {
    if (!raw) return "";
    const s0 = String(raw);

    const m = s0.match(/smallgroup[_\(\s]*?(\d+)[, _]+(\d+)\)?/i);
    if (m) {
      const n = Number(m[1]), k = Number(m[2]);
      const key = `${n},${k}`;
      const map = {
        "1,1":"Trivial","2,1":"C2","3,1":"C3","4,1":"C4","5,1":"C5","6,1":"C6",
        "7,1":"C7","8,1":"C8","8,2":"C4 × C2","8,5":"C2 × C2 × C2",
        "9,1":"C9","9,2":"C3 × C3","10,1":"C10",
        "12,1":"C12","12,2":"C6 × C2","12,3":"D12","12,4":"A4",
        "16,5":"C2 × C2 × C2 × C2",
      };
      return map[key] || `SmallGroup(${n},${k})`;
    }

    if (s0.includes("x")) return s0.split("x").join(" × ");
    return s0.replace("⋊", " ⋊ ");
  }

  // ---------------- normalize ----------------
  function normalize(rec) {
    const codeId = rec.code_id || rec.id || rec.name || "";
    const groupRaw = groupRawValue(rec.group ?? rec.G, codeId);
    const group = groupDisplay(groupRaw);

    const n = toInt(getFirst(rec, ["n"])) ?? null;
    const k = toInt(getFirst(rec, ["k"])) ?? parseSuffix(codeId, "_k");
    const d = toInt(getFirst(rec, ["d_ub","d","distance.d_ub","distance.d"])) ?? parseSuffix(codeId, "_d");

    // IMPORTANT: include dist-m4ri progressive fields (distance.steps_used_total)
    const trials = toInt(getFirst(rec, [
      "m4ri_steps",
      "m4ri_trials",
      "trials",
      "steps",
      "steps_used",
      "steps_used_total",
      "distance_steps",
      "distance_trials",
      "distance.trials",
      "distance.steps",
      "distance.steps_used_total",
      "distance.steps_used_x",
      "distance.steps_used_z",
      "distance.steps_fast",
      "distance.steps_slow",
    ])) ?? null;

    // dX / dZ appear under different spellings depending on generator
    const dX = toInt(getFirst(rec, ["dX_ub","dx_ub","distance.dX_ub","distance.dx_ub","distance.dX_best","distance.dX"])) ?? null;
    const dZ = toInt(getFirst(rec, ["dZ_ub","dz_ub","distance.dZ_ub","distance.dz_ub","distance.dZ_best","distance.dZ"])) ?? null;

    return { rec, codeId, groupRaw: String(groupRaw), group, n, k, d, trials, dX, dZ };
  }

  async function loadData() {
    const url = "data.json?cb=" + Date.now();
    const resp = await fetch(url, { cache: "no-store" });
    if (!resp.ok) throw new Error(`Failed to load data.json: ${resp.status}`);
    const data = await resp.json();

    let codes = data.codes;
    if (Array.isArray(codes)) {
      // ok
    } else if (codes && typeof codes === "object") {
      codes = Object.values(codes);
    } else {
      codes = [];
    }

    const norm = codes
      .filter(x => x && typeof x === "object")
      .map(normalize)
      .filter(x => x.codeId);

    return { data, codes: norm };
  }

  // ---------------- best-per-(group,n,k) ----------------
  function better(a, b) {
    const da = (a.d ?? -1), db = (b.d ?? -1);
    if (da !== db) return da > db;

    const ta = (a.trials ?? -1), tb = (b.trials ?? -1);
    if (ta !== tb) return ta > tb;

    return String(a.codeId) < String(b.codeId);
  }

  function dedupeBestPerGroupNK(codes) {
    const best = new Map();
    for (const c of codes) {
      if (c.groupRaw === null || c.n === null || c.k === null) continue;
      const key = `${c.groupRaw}|${c.n}|${c.k}`;
      const cur = best.get(key);
      if (!cur || better(c, cur)) best.set(key, c);
    }
    return Array.from(best.values());
  }

  // ---------------- pivot (rows=n, cols=k) ----------------
  function buildPivotNKAllTies(codes) {
    const ns = Array.from(new Set(codes.map(c => c.n).filter(v => v !== null))).sort((a,b)=>a-b);
    const ks = Array.from(new Set(codes.map(c => c.k).filter(v => v !== null))).sort((a,b)=>a-b);

    const best = new Map(); // key "n|k" -> [codes... ties across groups]

    for (const c of codes) {
      if (c.n === null || c.k === null) continue;
      const key = `${c.n}|${c.k}`;

      const curList = best.get(key);
      if (!curList) { best.set(key, [c]); continue; }

      const cur = curList[0];
      if (better(c, cur)) {
        best.set(key, [c]);
      } else {
        const da = (c.d ?? -1), db = (cur.d ?? -1);
        const ta = (c.trials ?? -1), tb = (cur.trials ?? -1);
        if (da === db && ta === tb) {
          if (!curList.some(x => x.codeId === c.codeId)) curList.push(c);
        }
      }
    }

    // stable sort ties by group name for readability
    for (const [k, arr] of best.entries()) {
      arr.sort((a,b)=>String(a.group).localeCompare(String(b.group)));
      best.set(k, arr);
    }

    return { ns, ks, best };
  }

  // ---------------- heatmap colors ----------------
  function cssVar(name, fallback) {
    const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    return v || fallback;
  }

  function parseColorToRgb(s) {
    const t = String(s || "").trim();
    if (!t) return null;
    if (t.startsWith("#")) {
      const hex = t.slice(1);
      if (hex.length === 6) {
        const r = parseInt(hex.slice(0,2), 16);
        const g = parseInt(hex.slice(2,4), 16);
        const b = parseInt(hex.slice(4,6), 16);
        if ([r,g,b].every(Number.isFinite)) return {r,g,b};
      }
      return null;
    }
    const m = t.match(/rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)/i);
    if (m) return { r: Number(m[1]), g: Number(m[2]), b: Number(m[3]) };
    return null;
  }

  function lerp(a,b,t){ return a + (b-a)*t; }

  function colorFor(d, minD, maxD) {
    const cMin = parseColorToRgb(cssVar("--cell-min", "#f6f8ff")) || {r:246,g:248,b:255};
    const cMax = parseColorToRgb(cssVar("--cell-max", "#173a8a")) || {r:23,g:58,b:138};

    if (d === null || d === undefined || minD === null || maxD === null) {
      const miss = cssVar("--cell-missing", "#f3f4f6");
      return { bg: miss, fg: "#111827" };
    }

    const denom = (maxD - minD);
    const t = denom > 0 ? (d - minD) / denom : 1;
    const tt = Math.max(0, Math.min(1, t));

    const r = Math.round(lerp(cMin.r, cMax.r, tt));
    const g = Math.round(lerp(cMin.g, cMax.g, tt));
    const b = Math.round(lerp(cMin.b, cMax.b, tt));

    // perceived luminance for text color
    const lum = 0.2126*r + 0.7152*g + 0.0722*b;
    const fg = lum > 150 ? "#111827" : "#ffffff";
    return { bg: `rgb(${r},${g},${b})`, fg };
  }

  // ---------------- modal ----------------
  const modal = {
    root: () => document.getElementById("qt-modal"),
    title: () => document.getElementById("qt-modal-title"),
    sub: () => document.getElementById("qt-modal-sub"),
    body: () => document.getElementById("qt-modal-body"),
    closeBtn: () => document.getElementById("qt-modal-close"),
  };

  function showModal(title, subtitle, bodyHTML) {
    modal.title().textContent = title || "";
    modal.sub().textContent = subtitle || "";
    modal.body().innerHTML = bodyHTML || "";
    modal.root().classList.remove("hidden");
    modal.root().setAttribute("aria-hidden", "false");
  }
  function hideModal() {
    modal.root().classList.add("hidden");
    modal.root().setAttribute("aria-hidden", "true");
    modal.body().innerHTML = "";
  }

  function initModal() {
    modal.closeBtn().addEventListener("click", hideModal);
    modal.root().addEventListener("click", (e) => {
      if (e.target === modal.root()) hideModal();
    });
    window.addEventListener("keydown", (e) => {
      if (e.key === "Escape") hideModal();
    });
  }

  async function fetchJSON(url) {
    const r = await fetch(url, { cache: "no-store" });
    if (!r.ok) throw new Error(`${url} -> ${r.status}`);
    return await r.json();
  }

  function matrixLinks(codeId) {
    const base = `matrices/${encodeURIComponent(codeId)}`;
    const candidates = [
      {label:"Hx", url:`${base}__Hx.mtx`},
      {label:"Hz", url:`${base}__Hz.mtx`},
      {label:"HX", url:`${base}__HX.mtx`},
      {label:"HZ", url:`${base}__HZ.mtx`},
    ];
    return candidates.map(c => `<a href="${c.url}" target="_blank" rel="noopener">${escHtml(c.label)}</a>`).join(" • ");
  }

  function extractTrials(meta, fallback) {
    return (
      meta?.m4ri_steps ??
      meta?.m4ri_trials ??
      meta?.trials ??
      meta?.steps ??
      meta?.steps_used ??
      meta?.distance?.steps_used_total ??
      meta?.distance?.steps_used_x ??
      meta?.distance?.steps_used_z ??
      meta?.distance?.steps_fast ??
      meta?.distance?.steps_slow ??
      meta?.distance?.steps ??
      meta?.distance?.trials ??
      fallback ??
      null
    );
  }

  function extractDx(meta, fallback) {
    return (
      meta?.dX_ub ??
      meta?.dx_ub ??
      meta?.distance?.dX_ub ??
      meta?.distance?.dx_ub ??
      meta?.distance?.dX_best ??
      meta?.distance?.dx?.d_ub ??
      fallback ??
      null
    );
  }

  function extractDz(meta, fallback) {
    return (
      meta?.dZ_ub ??
      meta?.dz_ub ??
      meta?.distance?.dZ_ub ??
      meta?.distance?.dz_ub ??
      meta?.distance?.dZ_best ??
      meta?.distance?.dz?.d_ub ??
      fallback ??
      null
    );
  }

  async function openDetails(code, backFn) {
    const codeId = code.codeId;
    const cb = Date.now();
    const urls = [
      `meta/${encodeURIComponent(codeId)}.json?cb=${cb}`,
      `collected/${encodeURIComponent(codeId)}/meta.json?cb=${cb}`,
    ];

    let meta = null;
    let source = null;
    for (const u of urls) {
      try { meta = await fetchJSON(u); source = u; break; } catch (_) {}
    }

    if (!meta) {
      showModal("Code details", codeId, `
        <p><b>Could not load metadata</b> for this code.</p>
        <p class="muted">Tried:</p>
        <pre>${escHtml(urls.join("\n"))}</pre>
      `);
      return;
    }

    const n = meta.n ?? code.n;
    const k = meta.k ?? code.k;
    const d = meta.d_ub ?? meta.d ?? meta.distance?.d_ub ?? meta.distance?.d ?? code.d;
    const trials = extractTrials(meta, code.trials);
    const dx = extractDx(meta, code.dX);
    const dz = extractDz(meta, code.dZ);
    const method = meta.distance?.method ?? meta.distance_backend ?? meta.distance?.backend ?? meta.method ?? "";

    const group = groupDisplay(groupRawValue(meta.group ?? meta.G ?? code.groupRaw, codeId));
    const prov = meta.provenance ?? meta.run_dir ?? meta.source ?? meta.results_dir ?? "";

    const backBtn = backFn
      ? `<button id="qt-back" type="button" class="secondary" style="margin-bottom:10px">← Back</button>`
      : "";

    const body = `
      ${backBtn}
      <p><code>${escHtml(codeId)}</code></p>
      <p>
        <span class="badge">${escHtml(group)}</span>
        ${method ? `<span class="badge">${escHtml(method)}</span>` : ``}
        ${prov ? `<span class="badge">prov: ${escHtml(prov)}</span>` : ``}
      </p>

      <p>
        Parameters: <b>n</b> ${escHtml(n ?? "")} · <b>k</b> ${escHtml(k ?? "")} · <b>d_ub</b> ${escHtml(d ?? "")}
        ${dx!=null || dz!=null ? ` · <b>dX</b> ${escHtml(dx ?? "")} · <b>dZ</b> ${escHtml(dz ?? "")}` : ``}
      </p>

      <p><b>m4ri trials</b>: ${escHtml(trials ?? "")}</p>

      <p><b>Matrices</b>: ${matrixLinks(codeId)}</p>

      <p class="muted">meta source: <code>${escHtml(source || "")}</code></p>

      <details>
        <summary><b>Full embedded meta.json</b></summary>
        <pre>${escHtml(JSON.stringify(meta, null, 2))}</pre>
      </details>
    `;

    showModal("Code details", "", body);

    if (backFn) {
      const btn = document.getElementById("qt-back");
      if (btn) btn.addEventListener("click", backFn);
    }
  }

  // ---------------- rendering ----------------
  const els = {
    stats: () => document.getElementById("stats"),
    legend: () => document.getElementById("legend"),
    table: () => document.getElementById("table"),
    groupSel: () => document.getElementById("groupSel"),
    searchBox: () => document.getElementById("searchBox"),
    minTrials: () => document.getElementById("minTrials"),
    nMin: () => document.getElementById("nMin"),
    nMax: () => document.getElementById("nMax"),
    kMin: () => document.getElementById("kMin"),
    kMax: () => document.getElementById("kMax"),
    modeSel: () => document.getElementById("modeSel"),
    btnRender: () => document.getElementById("btnRender"),
    btnReset: () => document.getElementById("btnReset"),
  };

  function fillGroupSelect(bestCodes) {
    const sel = els.groupSel();
    sel.innerHTML = "";

    const groups = Array.from(new Set(bestCodes.map(c => c.groupRaw))).sort((a,b) => {
      return groupDisplay(a).localeCompare(groupDisplay(b));
    });

    const optAll = document.createElement("option");
    optAll.value = "";
    optAll.textContent = "All groups";
    sel.appendChild(optAll);

    for (const g of groups) {
      const opt = document.createElement("option");
      opt.value = g;
      opt.textContent = groupDisplay(g);
      sel.appendChild(opt);
    }

    const url = new URL(window.location.href);
    const qg = url.searchParams.get("group") || "";
    if (qg && groups.includes(qg)) sel.value = qg;
  }

  function readFilters() {
    const groupRaw = els.groupSel().value || "";
    const search = (els.searchBox().value || "").trim().toLowerCase();
    const minTrials = toInt(els.minTrials().value) ?? 0;

    const nMin = toInt(els.nMin().value);
    const nMax = toInt(els.nMax().value);
    const kMin = toInt(els.kMin().value);
    const kMax = toInt(els.kMax().value);

    const mode = els.modeSel().value || "d";

    return { groupRaw, search, minTrials, nMin, nMax, kMin, kMax, mode };
  }

  function passesRange(v, lo, hi) {
    if (v === null || v === undefined) return false;
    if (lo !== null && lo !== undefined && v < lo) return false;
    if (hi !== null && hi !== undefined && v > hi) return false;
    return true;
  }

  function applyFilters(bestCodes, f) {
    return bestCodes.filter(c => {
      if (f.groupRaw && c.groupRaw !== f.groupRaw) return false;

      const t = c.trials ?? 0;
      if (f.minTrials > 0 && t < f.minTrials) return false;

      if (!passesRange(c.n, f.nMin, f.nMax)) return false;
      if (!passesRange(c.k, f.kMin, f.kMax)) return false;

      if (f.search) {
        const hay = [
          c.codeId,
          c.groupRaw,
          c.group,
          c.n, c.k, c.d,
        ].map(x => String(x ?? "").toLowerCase()).join(" | ");
        if (!hay.includes(f.search)) return false;
      }

      return true;
    });
  }

  function updateLegend(minD, maxD) {
    const el = els.legend();
    if (minD === null || maxD === null) { el.innerHTML = ""; return; }
    const c1 = cssVar("--cell-min", "#f6f8ff");
    const c2 = cssVar("--cell-max", "#173a8a");
    el.innerHTML = `
      <span><b>d_ub</b></span>
      <span>${escHtml(minD)}</span>
      <span class="legend-bar" style="background:linear-gradient(90deg, ${c1}, ${c2})"></span>
      <span>${escHtml(maxD)}</span>
      <span class="muted">(heatmap)</span>
    `;
  }

  let LAST_PIVOT = null;

  function renderTable(filteredCodes, mode) {
    const pivot = buildPivotNKAllTies(filteredCodes);
    LAST_PIVOT = pivot;

    const { ns, ks, best } = pivot;
    const tableEl = els.table();

    if (ns.length === 0 || ks.length === 0) {
      tableEl.innerHTML = `<div style="padding:12px" class="muted">No codes match your filters.</div>`;
      updateLegend(null, null);
      return;
    }

    // distance range among non-empty cells
    let minD = null, maxD = null;
    for (const arr of best.values()) {
      const d = arr[0]?.d;
      if (d === null || d === undefined) continue;
      if (minD === null || d < minD) minD = d;
      if (maxD === null || d > maxD) maxD = d;
    }
    updateLegend(minD, maxD);

    let html = `<table class="grid"><thead><tr>`;
    html += `<th class="corner">n \\ k</th>`;
    for (const k of ks) {
      // NOTE: no "k=" prefix (requested)
      html += `<th class="colhead">${escHtml(k)}</th>`;
    }
    html += `</tr></thead><tbody>`;

    for (const n of ns) {
      // NOTE: no "n=" prefix (requested)
      html += `<tr><th class="rowhead">${escHtml(n)}</th>`;

      for (const k of ks) {
        const key = `${n}|${k}`;
        const arr = best.get(key);

        if (!arr || arr.length === 0) {
          html += `<td class="missing"></td>`;
          continue;
        }

        const d = arr[0].d;
        const { bg, fg } = colorFor(d, minD, maxD);

        const groups = arr.map(x => x.group).join(" • ");
        const title = `n=${n}, k=${k}, d_ub=${d}, ties=${arr.length}`;

        let inner = `<div class="cell-d">${escHtml(d ?? "")}</div>`;
        if (mode === "dg") inner += `<div class="cell-g">${escHtml(groups)}</div>`;
        if (arr.length > 1) inner += `<div class="cell-ties">${escHtml(arr.length)} tie(s)</div>`;

        // IMPORTANT: no trials displayed in the cell (requested)
        html += `<td class="cell clickable" data-n="${escHtml(n)}" data-k="${escHtml(k)}" style="background:${bg};color:${fg}" title="${escHtml(title)}">${inner}</td>`;
      }

      html += `</tr>`;
    }

    html += `</tbody></table>`;
    tableEl.innerHTML = html;

    // attach click handlers
    tableEl.querySelectorAll("td.clickable").forEach(td => {
      td.addEventListener("click", () => {
        const n = td.getAttribute("data-n");
        const k = td.getAttribute("data-k");
        const key = `${n}|${k}`;
        const arr = LAST_PIVOT?.best?.get(key) || [];
        openCellList(n, k, arr);
      });
    });
  }

  function openCellList(n, k, arr) {
    if (!arr || arr.length === 0) return;

    const rows = arr.map(c => {
      const trials = c.trials ?? "";
      return `
        <tr>
          <td>${escHtml(c.group)}</td>
          <td>${escHtml(c.d ?? "")}</td>
          <td>${escHtml(trials)}</td>
          <td><a href="#" data-code="${escHtml(c.codeId)}"><code>${escHtml(c.codeId)}</code></a></td>
        </tr>
      `;
    }).join("");

    const body = `
      <p class="muted">Trials are shown here (not in the heatmap). Click a code_id for full details.</p>
      <table class="list">
        <thead>
          <tr>
            <th>Group</th><th>d_ub</th><th>m4ri trials</th><th>code_id</th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    `;

    showModal(`n ${n}  ·  k ${k}`, `${arr.length} code(s)`, body);

    modal.body().querySelectorAll("a[data-code]").forEach(a => {
      a.addEventListener("click", async (e) => {
        e.preventDefault();
        const cid = a.getAttribute("data-code");
        const code = arr.find(x => x.codeId === cid) || arr[0];
        await openDetails(code, () => openCellList(n, k, arr));
      });
    });
  }

  // ---------------- main ----------------
  async function main() {
    initModal();

    const { data, codes } = await loadData();
    const bestCodes = dedupeBestPerGroupNK(codes);

    fillGroupSelect(bestCodes);

    const render = () => {
      const f = readFilters();
      const filtered = applyFilters(bestCodes, f);

      // stats
      const groupLabel = f.groupRaw ? groupDisplay(f.groupRaw) : "All groups";
      els.stats().textContent =
        `generated_at_utc: ${data.generated_at_utc || ""} · raw codes: ${codes.length} · best/group,n,k: ${bestCodes.length} · displayed: ${filtered.length} · group: ${groupLabel}`;

      renderTable(filtered, f.mode);
    };

    els.btnRender().addEventListener("click", render);
    els.btnReset().addEventListener("click", () => {
      els.groupSel().value = "";
      els.searchBox().value = "";
      els.minTrials().value = "0";
      els.nMin().value = "";
      els.nMax().value = "";
      els.kMin().value = "";
      els.kMax().value = "";
      els.modeSel().value = "d";
      render();
    });

    // auto-render on change (nice UX)
    for (const id of ["groupSel","searchBox","minTrials","nMin","nMax","kMin","kMax","modeSel"]) {
      const el = document.getElementById(id);
      if (el) el.addEventListener("input", () => render());
      if (el) el.addEventListener("change", () => render());
    }

    render();
  }

  main().catch((e) => {
    console.error(e);
    const msg = (e && e.stack) ? e.stack : String(e);
    document.getElementById("stats").textContent = "ERROR";
    document.getElementById("table").innerHTML = `<div style="padding:12px"><pre>${escHtml(msg)}</pre></div>`;
  });
})();
