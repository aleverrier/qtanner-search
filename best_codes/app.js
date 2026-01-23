(async function () {
  function groupDisplay(raw) {
    if (!raw) return "";
    const s = String(raw);

    // SmallGroup_n_m -> try common names, else show SmallGroup(n,m)
    if (s.startsWith("SmallGroup_")) {
      const m = s.match(/^SmallGroup_(\d+)_(\d+)/);
      if (m) {
        const n = Number(m[1]), k = Number(m[2]);
        const key = `${n},${k}`;
        const map = {
          "1,1": "Trivial",
          "2,1": "C2",
          "3,1": "C3",
          "4,1": "C4",
          "6,1": "C6",
          "8,1": "C8",
          "8,2": "C4 × C2",
          "8,3": "D8",
          "8,4": "Q8",
          "8,5": "C2 × C2 × C2",
          "12,1": "C12",
          "12,2": "C6 × C2",
          "12,3": "D12",
          "12,4": "A4",
        };
        return map[key] || `SmallGroup(${n},${k})`;
      }
    }

    // Pretty-print direct products "C2xC2xC2" -> "C2 × C2 × C2"
    if (s.includes("x")) return s.split("x").join(" × ");

    // Add spacing around semidirect if present
    return s.replace("⋊", " ⋊ ");
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
    const groupRaw = rec.group || rec.G || parseSuffix(codeId, "") || codeId.split("_", 1)[0];
    const group = groupDisplay(groupRaw);

    const n = toInt(getFirst(rec, ["n"])) ?? null;
    const k = toInt(getFirst(rec, ["k"])) ?? parseSuffix(codeId, "_k");

    const d = toInt(getFirst(rec, ["d_ub","d","distance.d_ub","distance.d"])) ?? parseSuffix(codeId, "_d");
    const trials = toInt(getFirst(rec, ["m4ri_steps","trials","steps","steps_used","distance_steps","distance.trials","distance.steps"])) ?? null;

    const dX = toInt(getFirst(rec, ["dX_ub","distance.dX_ub"])) ?? null;
    const dZ = toInt(getFirst(rec, ["dZ_ub","distance.dZ_ub"])) ?? null;

    return { rec, codeId, groupRaw: String(groupRaw), group, n, k, d, trials, dX, dZ };
  }

  async function loadData() {
    const resp = await fetch("data.json?cb=" + Date.now(), { cache: "no-store" });
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
    const norm = codes.filter(x => x && typeof x === "object").map(normalize).filter(x => x.codeId);
    return { data, codes: norm };
  }

  function buildPivot(codes) {
    const groups = Array.from(new Set(codes.map(c => c.group))).sort((a,b)=>a.localeCompare(b));
    const ks = Array.from(new Set(codes.map(c => c.k).filter(k => k !== null))).sort((a,b)=>a-b);

    // Choose best per (group,k): max d, then max trials
    const best = new Map();
    for (const c of codes) {
      if (c.k === null) continue;
      const key = c.group + "|" + c.k;
      const cur = best.get(key);
      if (!cur) { best.set(key, c); continue; }
      const cd = c.d ?? -1, rd = cur.d ?? -1;
      if (cd > rd) best.set(key, c);
      else if (cd === rd) {
        const ct = c.trials ?? -1, rt = cur.trials ?? -1;
        if (ct > rt) best.set(key, c);
      }
    }
    return { groups, ks, best };
  }

  function render({ data, codes }) {
    const { groups, ks, best } = buildPivot(codes);

    const esc = (s) => String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
    const cellText = (c) => {
      if (!c) return "";
      const d = c.d ?? "";
      const t = c.trials ?? "";
      return `d=${d}<br><span style="opacity:.75;font-size:12px;">${t}</span>`;
    };

    document.body.innerHTML = `
      <div style="max-width: 1400px; margin: 18px auto; padding: 0 12px; font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;">
        <div style="display:flex; justify-content:space-between; gap:12px; align-items:baseline;">
          <h2 style="margin:0;">Best qTanner codes</h2>
          <div><a href="simple.html">Switch to list view</a></div>
        </div>
        <div style="opacity:.75; margin: 6px 0 14px 0;">
          generated_at_utc: <code>${esc(data.generated_at_utc || "")}</code> • codes: ${codes.length}
        </div>

        <div style="overflow:auto; border:1px solid #ddd; border-radius: 10px;">
          <table style="border-collapse:collapse; width:100%; font-size: 13px;">
            <thead>
              <tr style="background:#f7f7f7; text-align:left;">
                <th style="padding:8px; border-bottom:1px solid #ddd; position:sticky; left:0; background:#f7f7f7; z-index:2;">k \\ group</th>
                ${groups.map(g=>`<th style="padding:8px; border-bottom:1px solid #ddd; white-space:nowrap;">${esc(g)}</th>`).join("")}
              </tr>
            </thead>
            <tbody>
              ${ks.map(k=>{
                return `<tr>
                  <td style="padding:8px; border-bottom:1px solid #eee; position:sticky; left:0; background:white; z-index:1;"><b>${k}</b></td>
                  ${groups.map(g=>{
                    const c = best.get(g + "|" + k);
                    const title = c ? `code_id=${c.codeId}\n(n=${c.n}, k=${c.k}, d=${c.d}, trials=${c.trials}, dX=${c.dX}, dZ=${c.dZ})` : "";
                    return `<td style="padding:8px; border-bottom:1px solid #eee; vertical-align:top;" title="${esc(title)}">${cellText(c)}</td>`;
                  }).join("")}
                </tr>`;
              }).join("")}
            </tbody>
          </table>
        </div>

        <div style="opacity:.75; margin-top:10px;">
          Cell shows <b>d</b> and <b>m4ri trials</b>. Hover a cell to see code_id and (dX_ub, dZ_ub) when available.
        </div>
      </div>
    `;
  }

  try {
    const payload = await loadData();
    render(payload);
  } catch (e) {
    document.body.innerHTML = `<pre style="white-space: pre-wrap; color: #b00; padding: 16px;">${String(e)}</pre>`;
    console.error(e);
  }
})();
