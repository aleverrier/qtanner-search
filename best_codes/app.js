(async function () {
  function $(sel) { return document.querySelector(sel); }

  function getFirst(obj, paths) {
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
  }

  function toInt(x) {
    const n = Number(x);
    return Number.isFinite(n) ? Math.trunc(n) : null;
  }

  function parseSuffix(codeId, key) {
    // key like "_k" or "_d"
    const re = new RegExp(`${key}(\\d+)(?:_|$)`);
    const m = String(codeId).match(re);
    return m ? parseInt(m[1], 10) : null;
  }

  function groupOf(codeId) {
    const s = String(codeId);
    const i = s.indexOf("_");
    return i >= 0 ? s.slice(0, i) : s;
  }

  function normalize(rec) {
    const codeId = rec.code_id || rec.id || rec.name || "";
    const group = rec.group || groupOf(codeId);

    const n = toInt(getFirst(rec, ["n"])) ?? null;
    const k = toInt(getFirst(rec, ["k"])) ?? parseSuffix(codeId, "_k");

    // Distance: prefer refined UB, then d, then nested
    const d =
      toInt(getFirst(rec, ["d_ub", "d", "distance.d_ub", "distance.d"])) ??
      parseSuffix(codeId, "_d");

    // Trials: THIS is the important part: prefer m4ri steps
    const trials =
      toInt(getFirst(rec, [
        "m4ri_steps",
        "trials",
        "steps",
        "steps_used",
        "distance_trials",
        "distance_steps",
        "distance.steps",
        "distance.trials",
      ])) ?? null;

    const dX = toInt(getFirst(rec, ["dX_ub", "distance.dX_ub"])) ?? null;
    const dZ = toInt(getFirst(rec, ["dZ_ub", "distance.dZ_ub"])) ?? null;

    return { raw: rec, codeId, group, n, k, d, trials, dX, dZ };
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

    const normalized = codes
      .filter(x => x && typeof x === "object")
      .map(normalize)
      .filter(x => x.codeId);

    return { data, codes: normalized };
  }

  function render({ data, codes }) {
    // Build UI shell
    document.body.innerHTML = `
      <div style="max-width: 1200px; margin: 20px auto; padding: 0 12px; font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;">
        <h2 style="margin: 0 0 6px 0;">Best qTanner codes</h2>
        <div style="opacity: 0.75; margin-bottom: 14px;">
          generated_at_utc: <span id="genat"></span> • total codes: <span id="total"></span>
        </div>

        <div style="display:flex; gap:10px; flex-wrap: wrap; align-items: center; margin-bottom: 14px;">
          <label>Group:
            <select id="groupSel"></select>
          </label>
          <label>Search:
            <input id="searchBox" placeholder="substring of code_id" style="width: 320px;"/>
          </label>
          <label>Min m4ri trials:
            <input id="minTrials" type="number" min="0" step="1000" style="width: 140px;" />
          </label>
        </div>

        <div style="overflow:auto; border:1px solid #ddd; border-radius: 10px;">
          <table style="border-collapse: collapse; width: 100%; font-size: 14px;">
            <thead>
              <tr style="text-align:left; background:#f7f7f7;">
                <th style="padding: 8px; border-bottom:1px solid #ddd;">Group</th>
                <th style="padding: 8px; border-bottom:1px solid #ddd;">n</th>
                <th style="padding: 8px; border-bottom:1px solid #ddd;">k</th>
                <th style="padding: 8px; border-bottom:1px solid #ddd;">d_ub</th>
                <th style="padding: 8px; border-bottom:1px solid #ddd;">m4ri_trials</th>
                <th style="padding: 8px; border-bottom:1px solid #ddd;">dX_ub</th>
                <th style="padding: 8px; border-bottom:1px solid #ddd;">dZ_ub</th>
                <th style="padding: 8px; border-bottom:1px solid #ddd;">code_id</th>
              </tr>
            </thead>
            <tbody id="tbody"></tbody>
          </table>
        </div>
      </div>
    `;

    $("#genat").textContent = String(data.generated_at_utc || "");
    $("#total").textContent = String(codes.length);

    // Populate group selector
    const groups = Array.from(new Set(codes.map(c => c.group))).sort();
    const sel = $("#groupSel");
    sel.innerHTML = `<option value="">(all)</option>` + groups.map(g => `<option value="${g}">${g}</option>`).join("");

    // Default minTrials: if user refined a group recently, they care about big numbers
    $("#minTrials").value = "0";

    function rowHTML(c) {
      const td = (x) => `<td style="padding: 7px 8px; border-bottom:1px solid #eee; white-space: nowrap;">${x ?? ""}</td>`;
      const code = `<code style="font-size: 12px; white-space: nowrap;">${c.codeId}</code>`;
      return `<tr>
        ${td(c.group)}
        ${td(c.n)}
        ${td(c.k)}
        ${td(c.d)}
        ${td(c.trials)}
        ${td(c.dX)}
        ${td(c.dZ)}
        ${td(code)}
      </tr>`;
    }

    function applyFilters() {
      const g = sel.value;
      const q = ($("#searchBox").value || "").trim();
      const minT = Number($("#minTrials").value || "0");

      let out = codes.slice();
      if (g) out = out.filter(c => c.group === g);
      if (q) out = out.filter(c => c.codeId.includes(q));
      if (Number.isFinite(minT) && minT > 0) out = out.filter(c => (c.trials ?? 0) >= minT);

      // Sort: group, then k asc, then d desc, then trials desc
      out.sort((a,b) => {
        if (a.group !== b.group) return a.group.localeCompare(b.group);
        const ak = a.k ?? 10**9, bk = b.k ?? 10**9;
        if (ak !== bk) return ak - bk;
        const ad = a.d ?? -1, bd = b.d ?? -1;
        if (ad !== bd) return bd - ad;
        const at = a.trials ?? -1, bt = b.trials ?? -1;
        return bt - at;
      });

      $("#tbody").innerHTML = out.map(rowHTML).join("");
    }

    sel.addEventListener("change", applyFilters);
    $("#searchBox").addEventListener("input", applyFilters);
    $("#minTrials").addEventListener("input", applyFilters);

    applyFilters();
  }

  try {
    const payload = await loadData();
    render(payload);
  } catch (e) {
    document.body.innerHTML = `<pre style="white-space: pre-wrap; color: #b00; padding: 16px;">${String(e)}</pre>`;
    console.error(e);
  }
})();
