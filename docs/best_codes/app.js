async function loadData() {
  const resp = await fetch('data.json', {cache: 'no-store'});
  if (!resp.ok) throw new Error(`Failed to load data.json: ${resp.status}`);
  return await resp.json();
}

function uniqSorted(arr) {
  return Array.from(new Set(arr)).sort((a,b) => a-b);
}

function clampInt(v, fallback) {
  const x = Number(v);
  return Number.isFinite(x) ? Math.trunc(x) : fallback;
}

function makeCellColor(d, dMin, dMax) {
  // Simple blue heat: interpolate between a very light and a deep color.
  // If all equal, keep light.
  if (!Number.isFinite(dMin) || !Number.isFinite(dMax) || dMax <= dMin) return 'rgba(23,58,138,0.12)';
  const t = (d - dMin) / (dMax - dMin);
  // alpha and darkness vary with t
  const alpha = 0.10 + 0.55 * t;
  return `rgba(23,58,138,${alpha.toFixed(3)})`;
}

function groupOptions(codes) {
  const groups = Array.from(new Set(codes.map(c => c.group))).sort();
  return ['ALL', ...groups];
}

function filterCodes(codes, group, nMin, nMax, kMin, kMax) {
  return codes.filter(c => {
    if (group !== 'ALL' && c.group !== group) return false;
    if (Number.isFinite(nMin) && c.n < nMin) return false;
    if (Number.isFinite(nMax) && c.n > nMax) return false;
    if (Number.isFinite(kMin) && c.k < kMin) return false;
    if (Number.isFinite(kMax) && c.k > kMax) return false;
    return true;
  });
}

function buildIndex(codes) {
  // Map: n -> k -> list of codes
  const map = new Map();
  for (const c of codes) {
    if (!map.has(c.n)) map.set(c.n, new Map());
    const mk = map.get(c.n);
    if (!mk.has(c.k)) mk.set(c.k, []);
    mk.get(c.k).push(c);
  }
  // sort lists by d desc
  for (const [n, mk] of map.entries()) {
    for (const [k, lst] of mk.entries()) {
      lst.sort((a,b) => (b.d_recorded - a.d_recorded) || (a.code_id.localeCompare(b.code_id)));
    }
  }
  return map;
}

function openModal(cellInfo) {
  const {n, k, codes} = cellInfo;

  const backdrop = document.getElementById('modalBackdrop');
  const modal = document.getElementById('modal');
  const title = document.getElementById('modalTitle');
  const body = document.getElementById('modalBody');

  title.textContent = `n=${n}, k=${k} — ${codes.length} code(s)`;
  body.innerHTML = '';

  const bestD = Math.max(...codes.map(c => c.d_recorded));
  const p = document.createElement('p');
  p.className = 'kv';
  p.innerHTML = `Best recorded <code>d</code> in this cell: <b>${bestD}</b>.`;
  body.appendChild(p);

  for (const c of codes) {
    const card = document.createElement('div');
    card.className = 'codeCard';

    const hdr = document.createElement('div');
    hdr.className = 'codeCardHeader';

    const codeId = document.createElement('div');
    codeId.className = 'codeId';
    codeId.textContent = c.code_id;

    const b1 = document.createElement('span');
    b1.className = 'badge';
    b1.textContent = `G=${c.group}`;

    const b2 = document.createElement('span');
    b2.className = 'badge';
    b2.textContent = `d=${c.d_recorded}`;

    hdr.appendChild(codeId);
    hdr.appendChild(b1);
    hdr.appendChild(b2);
    card.appendChild(hdr);

    const kv = document.createElement('div');
    kv.className = 'kv';

    const A = Array.isArray(c.A_elems) ? `[${c.A_elems.join(', ')}]` : '(unknown)';
    const B = Array.isArray(c.B_elems) ? `[${c.B_elems.join(', ')}]` : '(unknown)';

    kv.innerHTML = `
      <div><b>Recorded:</b> d=${c.d_recorded} (${c.d_recorded_kind || 'recorded'})</div>
      <div><b>Multiset A:</b> <code>${c.A_id}</code> → <code>${A}</code></div>
      <div><b>Multiset B:</b> <code>${c.B_id}</code> → <code>${B}</code></div>
      <div><b>Provenance:</b> <code>${c.run_dir}</code></div>
    `;
    card.appendChild(kv);

    const links = document.createElement('div');
    links.className = 'kv links';
    const metaLink = c.meta_url_blob ? `<a href="${c.meta_url_blob}" target="_blank" rel="noreferrer">meta.json</a>` : '';
    const srcLink = c.src_dir_url_blob ? `<a href="${c.src_dir_url_blob}" target="_blank" rel="noreferrer">source folder</a>` : '';
    const collLink = c.collected_dir_url_blob ? `<a href="${c.collected_dir_url_blob}" target="_blank" rel="noreferrer">collected folder</a>` : '';
    links.innerHTML = `${metaLink} ${srcLink} ${collLink}`;
    card.appendChild(links);

    // Matrices
    if (Array.isArray(c.matrices) && c.matrices.length > 0) {
      const mdiv = document.createElement('div');
      mdiv.className = 'kv links';
      const parts = [];
      for (const m of c.matrices) {
        const label = m.kind ? m.kind : 'mtx';
        const url = m.url_blob || m.url_raw || '#';
        parts.push(`<a href="${url}" target="_blank" rel="noreferrer">${label}:${m.filename}</a>`);
      }
      mdiv.innerHTML = `<div><b>Matrices:</b> ${parts.join(' ')}</div>`;
      card.appendChild(mdiv);
    }

    // Other collected files (likely contains permutations, classical params, etc.)
    if (Array.isArray(c.extra_files) && c.extra_files.length > 0) {
      const fdiv = document.createElement('div');
      fdiv.className = 'kv';
      const list = document.createElement('ul');
      for (const f of c.extra_files.slice(0, 30)) {
        const li = document.createElement('li');
        if (f.url_blob) {
          const a = document.createElement('a');
          a.href = f.url_blob;
          a.target = '_blank';
          a.rel = 'noreferrer';
          a.textContent = f.path;
          li.appendChild(a);
        } else {
          li.textContent = f.path;
        }
        list.appendChild(li);
      }
      fdiv.innerHTML = `<div><b>Settings / auxiliary files (first 30):</b></div>`;
      fdiv.appendChild(list);
      card.appendChild(fdiv);
    }

    body.appendChild(card);
  }

  backdrop.classList.remove('hidden');
  modal.classList.remove('hidden');
}

function closeModal() {
  document.getElementById('modalBackdrop').classList.add('hidden');
  document.getElementById('modal').classList.add('hidden');
}

function renderTable(allCodes, ui) {
  const group = ui.groupSelect.value;
  const nMin = ui.nMin.value ? clampInt(ui.nMin.value, NaN) : NaN;
  const nMax = ui.nMax.value ? clampInt(ui.nMax.value, NaN) : NaN;
  const kMin = ui.kMin.value ? clampInt(ui.kMin.value, NaN) : NaN;
  const kMax = ui.kMax.value ? clampInt(ui.kMax.value, NaN) : NaN;

  const codes = filterCodes(allCodes, group, nMin, nMax, kMin, kMax);
  const index = buildIndex(codes);

  const ns = uniqSorted(codes.map(c => c.n));
  const ks = uniqSorted(codes.map(c => c.k));

  // Determine d range for coloring
  const ds = [];
  for (const n of ns) {
    const mk = index.get(n);
    for (const k of ks) {
      const lst = mk?.get(k);
      if (lst && lst.length > 0) ds.push(lst[0].d_recorded);
    }
  }
  const dMin = ds.length ? Math.min(...ds) : NaN;
  const dMax = ds.length ? Math.max(...ds) : NaN;

  // Summary
  const summary = document.getElementById('summary');
  summary.textContent = `Showing ${codes.length} code(s), ${ns.length} n-values, ${ks.length} k-values. d range in visible best-cells: ${Number.isFinite(dMin)?dMin:'?'}..${Number.isFinite(dMax)?dMax:'?'}.`;

  // Build table
  const container = document.getElementById('tableContainer');
  container.innerHTML = '';

  const table = document.createElement('table');

  const thead = document.createElement('thead');
  const hr = document.createElement('tr');

  const corner = document.createElement('th');
  corner.className = 'firstcol';
  corner.textContent = 'n \\ k';
  hr.appendChild(corner);

  for (const k of ks) {
    const th = document.createElement('th');
    th.textContent = String(k);
    hr.appendChild(th);
  }
  thead.appendChild(hr);
  table.appendChild(thead);

  const tbody = document.createElement('tbody');

  const annotateMode = ui.showGroupInCell.value;

  for (const n of ns) {
    const tr = document.createElement('tr');

    const ntd = document.createElement('td');
    ntd.className = 'firstcol';
    ntd.textContent = String(n);
    tr.appendChild(ntd);

    const mk = index.get(n);

    for (const k of ks) {
      const td = document.createElement('td');
      td.className = 'cell';

      const lst = mk?.get(k) || [];
      if (!lst.length) {
        td.classList.add('empty');
        td.textContent = '';
      } else {
        const best = lst[0];
        td.style.background = makeCellColor(best.d_recorded, dMin, dMax);

        const main = document.createElement('div');
        main.className = 'cellMain';
        main.textContent = String(best.d_recorded);

        const sub = document.createElement('div');
        sub.className = 'cellSub';

        if (annotateMode === 'group') {
          sub.textContent = best.group;
        } else if (annotateMode === 'count') {
          sub.textContent = `#${lst.length}`;
        } else {
          sub.textContent = '';
        }

        td.title = `n=${n}, k=${k}, best d=${best.d_recorded} (${best.group}), codes in cell: ${lst.length}`;
        td.appendChild(main);
        if (sub.textContent) td.appendChild(sub);

        td.addEventListener('click', () => openModal({n, k, codes: lst}));
      }

      tr.appendChild(td);
    }
    tbody.appendChild(tr);
  }

  table.appendChild(tbody);
  container.appendChild(table);
}

function setupUI(data) {
  const ui = {
    groupSelect: document.getElementById('groupSelect'),
    nMin: document.getElementById('nMin'),
    nMax: document.getElementById('nMax'),
    kMin: document.getElementById('kMin'),
    kMax: document.getElementById('kMax'),
    showGroupInCell: document.getElementById('showGroupInCell'),
    resetBtn: document.getElementById('resetBtn'),
  };

  // Populate group dropdown
  ui.groupSelect.innerHTML = '';
  for (const g of groupOptions(data.codes)) {
    const opt = document.createElement('option');
    opt.value = g;
    opt.textContent = g;
    ui.groupSelect.appendChild(opt);
  }

  const rerender = () => renderTable(data.codes, ui);

  ui.groupSelect.addEventListener('change', rerender);
  ui.nMin.addEventListener('input', rerender);
  ui.nMax.addEventListener('input', rerender);
  ui.kMin.addEventListener('input', rerender);
  ui.kMax.addEventListener('input', rerender);
  ui.showGroupInCell.addEventListener('change', rerender);

  ui.resetBtn.addEventListener('click', () => {
    ui.groupSelect.value = 'ALL';
    ui.nMin.value = '';
    ui.nMax.value = '';
    ui.kMin.value = '';
    ui.kMax.value = '';
    ui.showGroupInCell.value = 'none';
    rerender();
  });

  document.getElementById('closeModalBtn').addEventListener('click', closeModal);
  document.getElementById('modalBackdrop').addEventListener('click', closeModal);
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') closeModal();
  });

  rerender();
}

(async function main() {
  try {
    const data = await loadData();
    setupUI(data);
  } catch (e) {
    const container = document.getElementById('tableContainer');
    container.innerHTML = `<p style="color:#b00;">Error: ${e}</p>`;
    console.error(e);
  }
})();
