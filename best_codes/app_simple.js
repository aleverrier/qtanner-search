(async function () {
  function groupDisplay(raw) {
    if (!raw) return "";
    const s = String(raw);
    if (s.startsWith("SmallGroup_")) {
      const m = s.match(/^SmallGroup_(\d+)_(\d+)/);
      if (m) {
        const n = Number(m[1]), k = Number(m[2]);
        const key = `${n},${k}`;
        const map = {
          "1,1":"Trivial","2,1":"C2","3,1":"C3","4,1":"C4","6,1":"C6",
          "8,1":"C8","8,2":"C4 × C2","8,3":"D8","8,4":"Q8","8,5":"C2 × C2 × C2",
          "12,1":"C12","12,2":"C6 × C2","12,3":"D12","12,4":"A4",
        };
        return map[key] || `SmallGroup(${n},${k})`;
      }
    }
    if (s.includes("x")) return s.split("x").join(" × ");
    return s.replace("⋊"," ⋊ ");
  }

  function toInt(x) {
    const n = Number(x);
    return Number.isFinite(n) ? Math.trunc(n) : null;
  }

  function parseSuffix(codeId, key) {
    const re = new RegExp(`${key}(\\d+)(?:_|$)`);
    const m = String(codeId).match(re);
    return m ? parseInt(m[1], 10) : null;
  }

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

  function normalize(rec) {
    const codeId = rec.code_id || rec.id || rec.name || "";
    const groupRaw = rec.group || rec.G || codeId.split("_", 1)[0];
    const group = groupDisplay(groupRaw);

    const n = toInt(getFirst(rec, ["n"])) ?? null;
    const k = toInt(getFirst(rec, ["k"])) ?? parseSuffix(codeId, "_k");
    const d = toInt(getFirst(rec, ["d_ub","d","distance.d_ub","distance.d"])) ?? parseSuffix(codeId, "_d");
    const trials = toInt(getFirst(rec, ["m4ri_steps","trials","steps","steps_used","distance_steps","distance.trials","distance.steps"])) ?? null;
    const dX = toInt(getFirst(rec, ["dX_ub","distance.dX_ub"])) ?? null;
    const dZ = toInt(getFirst(rec, ["dZ_ub","distance.dZ_ub"])) ?? null;

    return { codeId, group, n, k, d, trials, dX, dZ };
  }

  async function loadData() {
    const resp = await fetch("data.json?cb=" + Date.now(), { cache: "no-store" });
    if (!resp.ok) throw new Error(`Failed to load data.json: ${resp.status}`);
    const data = await resp.json();
    let codes = data.codes;
    if (Array.isArray(codes)) {}
    else if (codes && typeof codes === "object") codes = Object.values(codes);
    else codes = [];
    return { data, codes: codes.filter(x => x && typeof x === "object").map(normalize) };
  }

  function render({ data, codes }) {
    document.body.innerHTML = `
      <div style="max-width: 1200px; margin: 18px auto; padding: 0 12px; font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;">
        <div style="display:flex; justify-content:space-between; gap:12px; align-items:baseline;">
          <h2 style="margin:0;">Best qTanner codes (list view)</h2>
          <div><a href="index.html">Back to table view</a></div>
        </div>
        <div style="opacity:.75; margin: 6px 0 14px 0;">
          generated_at_utc: <code>${String(data.generated_at_utc || "")}</code> • codes: ${codes.length}
        </div>
        <div style="overflow:auto; border:1px solid #ddd; border-radius:10px;">
          <table style="border-collapse:collapse; width:100%; font-size: 14px;">
            <thead>
              <tr style="background:#f7f7f7; text-align:left;">
                <th style="padding:8px; border-bottom:1px solid #ddd;">Group</th>
                <th style="padding:8px; border-bottom:1px solid #ddd;">n</th>
                <th style="padding:8px; border-bottom:1px solid #ddd;">k</th>
                <th style="padding:8px; border-bottom:1px solid #ddd;">d_ub</th>
                <th style="padding:8px; border-bottom:1px solid #ddd;">m4ri_trials</th>
                <th style="padding:8px; border-bottom:1px solid #ddd;">dX_ub</th>
                <th style="padding:8px; border-bottom:1px solid #ddd;">dZ_ub</th>
                <th style="padding:8px; border-bottom:1px solid #ddd;">code_id</th>
              </tr>
            </thead>
            <tbody>
              ${codes.sort((a,b)=> (a.group.localeCompare(b.group)) || ((a.k??1e9)-(b.k??1e9)) || ((b.d??-1)-(a.d??-1))).map(c=>`
                <tr>
                  <td style="padding:8px; border-bottom:1px solid #eee;">${c.group}</td>
                  <td style="padding:8px; border-bottom:1px solid #eee;">${c.n ?? ""}</td>
                  <td style="padding:8px; border-bottom:1px solid #eee;">${c.k ?? ""}</td>
                  <td style="padding:8px; border-bottom:1px solid #eee;">${c.d ?? ""}</td>
                  <td style="padding:8px; border-bottom:1px solid #eee;">${c.trials ?? ""}</td>
                  <td style="padding:8px; border-bottom:1px solid #eee;">${c.dX ?? ""}</td>
                  <td style="padding:8px; border-bottom:1px solid #eee;">${c.dZ ?? ""}</td>
                  <td style="padding:8px; border-bottom:1px solid #eee;"><code style="font-size:12px;">${c.codeId}</code></td>
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
    document.body.innerHTML = `<pre style="white-space: pre-wrap; color: #b00; padding: 16px;">${String(e)}</pre>`;
    console.error(e);
  }
})();
