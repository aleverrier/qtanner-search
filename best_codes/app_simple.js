(async function () {
  const esc = (s) => String(s ?? "").replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
  const toInt = (x) => {
    if (x === null || x === undefined) return null;
    if (typeof x === "string" && x.trim() === "") return null;
    const n = Number(x);
    return Number.isFinite(n) ? Math.trunc(n) : null;
  };
  const parseSuffix = (codeId, key) => {
    const re = new RegExp(`${key}(\\d+)(?:_|$)`);
    const m = String(codeId).match(re);
    return m ? parseInt(m[1], 10) : null;
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

  function groupRawFromCodeId(codeId) {
    const s = String(codeId);
    const m = s.match(/^(SmallGroup[_\(\s]*\d+[, _]+\d+\)?)(?:__|_)/i);
    if (m) return m[1];
    const i = s.indexOf("_");
    return i >= 0 ? s.slice(0, i) : s;
  }

  function groupDisplay(raw) {
    if (!raw) return "";
    const s0 = String(raw);
    const m = s0.match(/smallgroup[_\(\s]*?(\d+)[, _]+(\d+)\)?/i);
    if (m) {
      const n = Number(m[1]), k = Number(m[2]);
      const key = `${n},${k}`;
      const map = {
        "2,1":"C2","3,1":"C3","4,1":"C4","6,1":"C6",
        "8,5":"C2 × C2 × C2","12,2":"C6 × C2","16,5":"C2 × C2 × C2 × C2",
      };
      return map[key] || `SmallGroup(${n},${k})`;
    }
    if (s0.includes("x")) return s0.split("x").join(" × ");
    return s0.replace("⋊"," ⋊ ");
  }

  function normalize(rec) {
    const codeId = rec.code_id || rec.id || rec.name || "";
    const groupRaw = rec.group || rec.G || groupRawFromCodeId(codeId);
    const group = groupDisplay(groupRaw);

    const n = toInt(getFirst(rec, ["n"])) ?? null;
    const k = toInt(getFirst(rec, ["k"])) ?? parseSuffix(codeId, "_k");
    const d = toInt(getFirst(rec, ["d_ub","d","distance.d_ub","distance.d"])) ?? parseSuffix(codeId, "_d");
    const trials = toInt(getFirst(rec, ["m4ri_steps","trials","steps","steps_used","distance_steps","distance.trials","distance.steps"])) ?? null;

    return { codeId, group, n, k, d, trials };
  }

  async function loadData() {
    const resp = await fetch("data.json?cb=" + Date.now(), { cache: "no-store" });
    if (!resp.ok) throw new Error(`Failed to load data.json: ${resp.status}`);
    const data = await resp.json();

    let codes = data.codes;
    if (Array.isArray(codes)) {}
    else if (codes && typeof codes === "object") codes = Object.values(codes);
    else codes = [];

    const norm = codes.filter(x => x && typeof x === "object").map(normalize).filter(x => x.codeId);
    return { data, codes: norm };
  }

  function dedupeBestPerGroupNK(codes) {
    // Keep only best per (group,n,k): max d then max trials
    const best = new Map();
    for (const c of codes) {
      if (c.n === null || c.k === null) continue;
      const key = c.group + "|" + c.n + "|" + c.k;
      const cur = best.get(key);
      if (!cur) { best.set(key, c); continue; }
      const cd = c.d ?? -1, rd = cur.d ?? -1;
      if (cd > rd) best.set(key, c);
      else if (cd === rd) {
        const ct = c.trials ?? -1, rt = cur.trials ?? -1;
        if (ct > rt) best.set(key, c);
      }
    }
    return Array.from(best.values());
  }

  function render({ data, codes }) {
    const deduped = dedupeBestPerGroupNK(codes);

    // Sort by length n first, then k, then group, then d desc, then trials desc
    deduped.sort((a,b)=>
      ((a.n??1e18)-(b.n??1e18)) ||
      ((a.k??1e18)-(b.k??1e18)) ||
      (a.group.localeCompare(b.group)) ||
      ((b.d??-1)-(a.d??-1)) ||
      ((b.trials??-1)-(a.trials??-1))
    );

    document.body.innerHTML = `
      <div style="max-width: 1200px; margin: 18px auto; padding: 0 12px; font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;">
        <div style="display:flex; justify-content:space-between; gap:12px; align-items:baseline;">
          <h2 style="margin:0;">Best qTanner codes (list view)</h2>
          <div><a href="index.html">Back to table</a></div>
        </div>
        <div style="opacity:.75; margin: 6px 0 14px 0;">
          generated_at_utc: <code>${esc(data.generated_at_utc || "")}</code> • codes: ${deduped.length}
        </div>

        <div style="overflow:auto; border:1px solid #ddd; border-radius:10px;">
          <table style="border-collapse:collapse; width:100%; font-size: 14px;">
            <thead>
              <tr style="background:#f7f7f7; text-align:left;">
                <th style="padding:8px; border-bottom:1px solid #ddd;">Group</th>
                <th style="padding:8px; border-bottom:1px solid #ddd;">n</th>
                <th style="padding:8px; border-bottom:1px solid #ddd;">k</th>
                <th style="padding:8px; border-bottom:1px solid #ddd;">d_ub</th>
                <th style="padding:8px; border-bottom:1px solid #ddd;">m4ri trials</th>
                <th style="padding:8px; border-bottom:1px solid #ddd;">code_id</th>
              </tr>
            </thead>
            <tbody>
              ${deduped.map(c=>`
                <tr>
                  <td style="padding:8px; border-bottom:1px solid #eee;">${esc(c.group)}</td>
                  <td style="padding:8px; border-bottom:1px solid #eee;">${esc(c.n ?? "")}</td>
                  <td style="padding:8px; border-bottom:1px solid #eee;">${esc(c.k ?? "")}</td>
                  <td style="padding:8px; border-bottom:1px solid #eee;">${esc(c.d ?? "")}</td>
                  <td style="padding:8px; border-bottom:1px solid #eee;">${esc(c.trials ?? "")}</td>
                  <td style="padding:8px; border-bottom:1px solid #eee;"><code style="font-size:12px;">${esc(c.codeId)}</code></td>
                </tr>
              `).join("")}
            </tbody>
          </table>
        </div>
      </div>
    `;
  }

  try {
    render(await loadData());
  } catch (e) {
    document.body.innerHTML = `<pre style="white-space: pre-wrap; color: #b00; padding: 16px;">${esc(String(e))}</pre>`;
    console.error(e);
  }
})();
