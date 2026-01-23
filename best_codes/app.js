(async function () {
  // ---------- tiny utilities ----------
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

  // ---------- group pretty printing ----------
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

    // Accept SmallGroup_3_1 / SmallGroup(3,1) / smallgroup(3,1)
    const m = s0.match(/smallgroup[_\(\s]*?(\d+)[, _]+(\d+)\)?/i);
    if (m) {
      const n = Number(m[1]), k = Number(m[2]);
      const key = `${n},${k}`;
      // Common ones mapped to standard names; others keep SmallGroup(n,k) (GAP standard).
      const map = {
        "1,1":"Trivial","2,1":"C2","3,1":"C3","4,1":"C4","5,1":"C5","6,1":"C6",
        "7,1":"C7","8,1":"C8","8,2":"C4 × C2","8,5":"C2 × C2 × C2",
        "9,1":"C9","9,2":"C3 × C3","10,1":"C10",
        "12,1":"C12","12,2":"C6 × C2","12,3":"D12","12,4":"A4",
        "16,5":"C2 × C2 × C2 × C2"
      };
      return map[key] || `SmallGroup(${n},${k})`;
    }

    if (s0.includes("x")) return s0.split("x").join(" × ");
    return s0.replace("⋊", " ⋊ ");
  }

  // ---------- normalize code record from data.json ----------
  function normalize(rec) {
    const codeId = rec.code_id || rec.id || rec.name || "";
    const groupRaw = rec.group || rec.G || groupRawFromCodeId(codeId);
    const group = groupDisplay(groupRaw);

    const n = toInt(getFirst(rec, ["n"])) ?? null;
    const k = toInt(getFirst(rec, ["k"])) ?? parseSuffix(codeId, "_k");

    const d =
      toInt(getFirst(rec, ["d_ub","d","distance.d_ub","distance.d"])) ??
      parseSuffix(codeId, "_d");

    const trials =
      toInt(getFirst(rec, [
        "m4ri_steps","trials","steps","steps_used",
        "distance_steps","distance.trials","distance.steps","distance_trials","distance_steps"
      ])) ?? null;

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

    const norm = codes
      .filter(x => x && typeof x === "object")
      .map(normalize)
      .filter(x => x.codeId);

    return { data, codes: norm };
  }

  // ---------- pivot: rows=n, cols=k ----------
  function buildPivotNK(codes) {
    const ns = Array.from(new Set(codes.map(c => c.n).filter(v => v !== null))).sort((a,b)=>a-b);
    const ks = Array.from(new Set(codes.map(c => c.k).filter(v => v !== null))).sort((a,b)=>a-b);

    // best per (n,k): max d, then max trials
    const best = new Map();
    for (const c of codes) {
      if (c.n === null || c.k === null) continue;
      const key = c.n + "|" + c.k;
      const cur = best.get(key);
      if (!cur) { best.set(key, c); continue; }
      const cd = c.d ?? -1, rd = cur.d ?? -1;
      if (cd > rd) best.set(key, c);
      else if (cd === rd) {
        const ct = c.trials ?? -1, rt = cur.trials ?? -1;
        if (ct > rt) best.set(key, c);
      }
    }
    return { ns, ks, best };
  }

  // ---------- modal overlay ----------
  function ensureModal() {
    if (document.getElementById("qt-modal")) return;

    const modal = document.createElement("div");
    modal.id = "qt-modal";
    modal.style.cssText = `
      position:fixed; inset:0; background:rgba(0,0,0,.35);
      display:none; align-items:center; justify-content:center; z-index:9999;
    `;
    modal.innerHTML = `
      <div id="qt-modal-card" style="
        width:min(980px, calc(100vw - 24px));
        max-height: calc(100vh - 24px);
        overflow:auto;
        background:white; border-radius:12px; border:1px solid #ddd;
        box-shadow: 0 12px 40px rgba(0,0,0,.20);
        padding:14px 16px;
        font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
      ">
        <div style="display:flex; justify-content:space-between; align-items:baseline; gap:12px;">
          <h3 id="qt-modal-title" style="margin:0;">Code details</h3>
          <button id="qt-modal-close" style="border:1px solid #ccc;border-radius:10px;padding:6px 10px;background:#f7f7f7;cursor:pointer;">Close</button>
        </div>
        <div id="qt-modal-sub" style="opacity:.75;margin:6px 0 12px 0;"></div>
        <div id="qt-modal-body"></div>
      </div>
    `;
    document.body.appendChild(modal);

    modal.addEventListener("click", (e) => {
      if (e.target === modal) hideModal();
    });
    document.getElementById("qt-modal-close").addEventListener("click", hideModal);
    window.addEventListener("keydown", (e) => {
      if (e.key === "Escape") hideModal();
    });
  }

  function showModal(title, subtitle, bodyHTML) {
    ensureModal();
    document.getElementById("qt-modal-title").textContent = title;
    document.getElementById("qt-modal-sub").textContent = subtitle || "";
    document.getElementById("qt-modal-body").innerHTML = bodyHTML;
    document.getElementById("qt-modal").style.display = "flex";
  }

  function hideModal() {
    const m = document.getElementById("qt-modal");
    if (m) m.style.display = "none";
  }

  async function fetchJSON(url) {
    const r = await fetch(url, { cache: "no-store" });
    if (!r.ok) throw new Error(url + " -> " + r.status);
    return await r.json();
  }

  function fmtList(x) {
    if (Array.isArray(x)) return x.join(", ");
    if (x === null || x === undefined) return "";
    return String(x);
  }

  function matrixLinks(codeId) {
    // Most common names in your repo
    const candidates = [
      {label:"Hx", url:`matrices/${encodeURIComponent(codeId)}__Hx.mtx`},
      {label:"Hz", url:`matrices/${encodeURIComponent(codeId)}__Hz.mtx`},
      {label:"Hx", url:`matrices/${encodeURIComponent(codeId)}__HX.mtx`},
      {label:"Hz", url:`matrices/${encodeURIComponent(codeId)}__HZ.mtx`},
    ];
    // Show all; GitHub Pages will 404 if absent.
    return candidates.map(c => `<a href="${c.url}" target="_blank" rel="noopener">${c.label}</a>`).join(" • ");
  }

  function pickConstruction(meta) {
    // Try to extract A/B/perms from known keys (robust to schema changes)
    const A = meta.A_elems ?? meta.A ?? meta.Aset ?? meta.A_elements ?? null;
    const B = meta.B_elems ?? meta.B ?? meta.Bset ?? meta.B_elements ?? null;

    // permutations may be stored in different ways
    const pA = meta.A_perm ?? meta.permA ?? meta.perm_A ?? meta.permutation_A ?? null;
    const pB = meta.B_perm ?? meta.permB ?? meta.perm_B ?? meta.permutation_B ?? null;

    const perm = meta.permutation ?? meta.perm ?? null;

    return { A, B, pA, pB, perm };
  }

  async function openDetails(code) {
    const codeId = code.codeId;
    const cb = Date.now();

    // Prefer best_codes/meta/<code_id>.json, fallback to collected/<code_id>/meta.json
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
      showModal("Code details", codeId, `<div style="color:#b00;">Could not load metadata for this code.</div><div style="opacity:.75;">Tried:<br>${urls.map(esc).join("<br>")}</div>`);
      return;
    }

    const n = meta.n ?? code.n;
    const k = meta.k ?? code.k;
    const d = meta.d_ub ?? meta.d ?? code.d;
    const trials = meta.m4ri_steps ?? meta.trials ?? meta.steps ?? code.trials;
    const dX = meta.dX_ub ?? meta.dX ?? code.dX;
    const dZ = meta.dZ_ub ?? meta.dZ ?? code.dZ;
    const group = groupDisplay(meta.group ?? meta.G ?? code.groupRaw ?? code.group);

    const {A,B,pA,pB,perm} = pickConstruction(meta);

    // Classical codes info if present (your meta often has slice/local codes)
    const slice = meta.slice_codes ?? meta.slice ?? null;
    const local = meta.local_codes ?? meta.local ?? null;

    const body = `
      <div style="margin:10px 0 14px 0;">
        <div><b>code_id:</b> <code>${esc(codeId)}</code></div>
        <div><b>Group:</b> ${esc(group)}</div>
        <div><b>Parameters:</b> n=${esc(n)} , k=${esc(k)} , d_ub=${esc(d)} ${dX!=null||dZ!=null ? `(dX_ub=${esc(dX??"")}, dZ_ub=${esc(dZ??"")})` : ""}</div>
        <div><b>m4ri trials:</b> ${esc(trials ?? "")}</div>
        <div><b>Parity-check matrices:</b> ${matrixLinks(codeId)}</div>
        <div style="opacity:.6; font-size:12px; margin-top:6px;">meta source: ${esc(source)}</div>
      </div>

      <hr style="border:none;border-top:1px solid #eee; margin:14px 0;" />

      <div>
        <h4 style="margin:0 0 8px 0;">Construction</h4>
        <div><b>A:</b> ${esc(fmtList(A))}</div>
        <div><b>B:</b> ${esc(fmtList(B))}</div>
        ${pA ? `<div><b>perm(A):</b> ${esc(fmtList(pA))}</div>` : ``}
        ${pB ? `<div><b>perm(B):</b> ${esc(fmtList(pB))}</div>` : ``}
        ${perm ? `<div><b>perm:</b> ${esc(fmtList(perm))}</div>` : ``}
      </div>

      ${(slice || local) ? `
        <hr style="border:none;border-top:1px solid #eee; margin:14px 0;" />
        <div>
          <h4 style="margin:0 0 8px 0;">Local / slice codes</h4>
          ${slice ? `<div><b>slice_codes:</b><pre style="white-space:pre-wrap;border:1px solid #eee;border-radius:10px;padding:10px;">${esc(JSON.stringify(slice, null, 2))}</pre></div>` : ``}
          ${local ? `<div><b>local_codes:</b><pre style="white-space:pre-wrap;border:1px solid #eee;border-radius:10px;padding:10px;">${esc(JSON.stringify(local, null, 2))}</pre></div>` : ``}
        </div>
      ` : ``}
    `;

    showModal("Code details", "", body);
  }

  // ---------- main render ----------
  function render({ data, codes }) {
    const { ns, ks, best } = buildPivotNK(codes);

    // Controls: cell mode toggle
    const controls = `
      <div style="display:flex; justify-content:space-between; align-items:baseline; gap:12px;">
        <h2 style="margin:0;">Best qTanner codes</h2>
        <div style="display:flex; gap:14px; align-items:center;">
          <label style="cursor:pointer;"><input type="radio" name="cellmode" value="d" checked> d only</label>
          <label style="cursor:pointer;"><input type="radio" name="cellmode" value="dg"> d + G</label>
          <a href="simple.html">List view</a>
        </div>
      </div>
      <div style="opacity:.75; margin: 6px 0 14px 0;">
        generated_at_utc: <code>${esc(data.generated_at_utc || "")}</code> • codes: ${codes.length}
      </div>
    `;

    const cellHTML = (c, mode) => {
      if (!c) return "";
      const d = (c.d === null || c.d === undefined) ? "" : c.d;
      const g = c.group || "";
      if (mode === "dg") {
        return `<div><b>${esc(d)}</b> <span style="opacity:.75">${esc(g)}</span></div>`;
      }
      return `<div><b>${esc(d)}</b></div>`;
    };

    document.body.innerHTML = `
      <div style="max-width: 1500px; margin: 18px auto; padding: 0 12px; font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;">
        ${controls}
        <div style="overflow:auto; border:1px solid #ddd; border-radius: 10px;">
          <table style="border-collapse:collapse; width:100%; font-size: 13px;">
            <thead>
              <tr style="background:#f7f7f7; text-align:left;">
                <th style="padding:8px; border-bottom:1px solid #ddd; position:sticky; left:0; background:#f7f7f7; z-index:2;">n \\ k</th>
                ${ks.map(k=>`<th style="padding:8px; border-bottom:1px solid #ddd; white-space:nowrap;">k=${k}</th>`).join("")}
              </tr>
            </thead>
            <tbody id="tbody">
              ${ns.map(n=>{
                return `<tr>
                  <td style="padding:8px; border-bottom:1px solid #eee; position:sticky; left:0; background:white; z-index:1;"><b>n=${n}</b></td>
                  ${ks.map(k=>{
                    const c = best.get(n + "|" + k);
                    const title = c ? `code_id=${c.codeId}\\n(group=${c.group})\\n(n=${c.n}, k=${c.k}, d=${c.d}, trials=${c.trials ?? ""})` : "";
                    return `<td class="cell" data-codeid="${c ? esc(c.codeId) : ""}" style="padding:8px; border-bottom:1px solid #eee; vertical-align:top; cursor:${c ? "pointer" : "default"};" title="${esc(title)}">
                              ${cellHTML(c, "d")}
                            </td>`;
                  }).join("")}
                </tr>`;
              }).join("")}
            </tbody>
          </table>
        </div>
        <div style="opacity:.75; margin-top:10px;">
          Click a non-empty cell for details (parameters, construction, trials, matrices).
        </div>
      </div>
    `;

    ensureModal();

    function applyMode(mode) {
      // Re-render only cell contents without rebuilding table structure
      const tds = document.querySelectorAll("td.cell");
      for (const td of tds) {
        const cid = td.getAttribute("data-codeid");
        if (!cid) continue;
        // Find code object
        const c = codes.find(x => x.codeId === cid);
        if (!c) continue;
        td.innerHTML = cellHTML(c, mode);
      }
    }

    // Toggle handler
    document.querySelectorAll('input[name="cellmode"]').forEach(r => {
      r.addEventListener("change", () => {
        const mode = document.querySelector('input[name="cellmode"]:checked').value;
        applyMode(mode);
      });
    });

    // Click handler
    document.querySelectorAll("td.cell").forEach(td => {
      td.addEventListener("click", async () => {
        const cid = td.getAttribute("data-codeid");
        if (!cid) return;
        const c = codes.find(x => x.codeId === cid);
        if (!c) return;
        await openDetails(c);
      });
    });
  }

  try {
    const payload = await loadData();
    render(payload);
  } catch (e) {
    document.body.innerHTML = `<pre style="white-space: pre-wrap; color: #b00; padding: 16px;">${esc(String(e))}</pre>`;
    console.error(e);
  }
})();
