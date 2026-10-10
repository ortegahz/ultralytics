/* Video frame-by-frame annotation client.
 *
 * Design notes that matter for speed:
 *  - Two stacked canvases: the image is redrawn only when the frame changes, boxes only when they change,
 *    so dragging a handle never re-decodes or re-uploads the picture.
 *  - Zoom/pan is a CSS transform on #stage, so the GPU rescales and the canvas backing store stays at
 *    native resolution; stroke widths are divided by the zoom to stay visually constant.
 *  - Accepted proposals advance automatically; "next unreviewed frame" skips finished work, so the
 *    keyboard alone can carry an annotator through a sequence.
 *  - Propagation extrapolates a track with the velocity measured from the previous frame and stops at the
 *    first frame that already carries a nearby box, which is what makes it safe to run on a long stretch.
 */
'use strict';

const $ = (id) => document.getElementById(id);
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

const S = {
  sequences: [],
  seq: null,          // active sequence info
  index: 0,
  labels: [],         // committed boxes on the current frame, pixel xyxy
  prelabels: [],      // model proposals on the current frame
  scores: [],
  selected: -1,
  undo: [],
  redo: [],
  dirty: false,
  loading: false,
  drawMode: false,
  drag: null,
  view: { z: 1, x: 0, y: 0 },
  showProposals: true,
  minScore: 0,
  follow: false,
  summary: null,
  jobs: {},
  jobTimer: null,
};

const HANDLE = 4;     // half-size of a resize handle in screen pixels
const MIN_SIZE = 2;   // smallest box side we allow, in image pixels

/* ------------------------------------------------------------------ API */

async function api(path, options) {
  const response = await fetch(path, options);
  const text = await response.text();
  let payload = {};
  try { payload = text ? JSON.parse(text) : {}; } catch (e) { payload = { error: text }; }
  if (!response.ok) throw new Error(payload.error || `${response.status} ${response.statusText}`);
  return payload;
}

const postJSON = (path, body) => api(path, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(body || {}),
});

function status(text, kind) {
  const bar = $('statusbar');
  $('status-text').textContent = text;
  bar.className = kind === 'err' ? 'flash-err' : kind === 'ok' ? 'flash-ok' : '';
  clearTimeout(status._t);
  status._t = setTimeout(() => { bar.className = ''; }, 2500);
}

/* -------------------------------------------------------------- geometry */

function boxCenter(b) { return { x: (b.x1 + b.x2) / 2, y: (b.y1 + b.y2) / 2 }; }

function normalizeBox(x1, y1, x2, y2) {
  return {
    cls: 0,
    x1: Math.min(x1, x2), y1: Math.min(y1, y2),
    x2: Math.max(x1, x2), y2: Math.max(y1, y2),
  };
}

function nearBox(boxes, cx, cy, radius) {
  let best = null, bestD = radius * radius;
  for (const b of boxes) {
    const c = boxCenter(b);
    const d = (c.x - cx) ** 2 + (c.y - cy) ** 2;
    if (d <= bestD) { bestD = d; best = b; }
  }
  return best;
}

function visibleProposals() {
  if (!S.showProposals) return [];
  return S.prelabels.filter((_, i) => (S.scores[i] ?? 1) >= S.minScore);
}

/** Map an index in the filtered proposal list back to its index in the raw prelabel array. */
function proposalSourceIndex(filteredIndex) {
  let seen = 0;
  for (let i = 0; i < S.prelabels.length; i++) {
    if ((S.scores[i] ?? 1) < S.minScore) continue;
    if (seen === filteredIndex) return i;
    seen++;
  }
  return filteredIndex;
}

/* --------------------------------------------------------- image loading */

const imgCache = new Map();
const PREFETCH_AHEAD = 6;

function loadImage(index) {
  if (imgCache.has(index)) return imgCache.get(index).promise;
  const entry = { image: null, state: 'loading', promise: null };
  entry.promise = new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => { entry.image = img; entry.state = 'ready'; resolve(img); };
    img.onerror = () => { entry.state = 'error'; reject(new Error(`frame ${index} failed to load`)); };
    img.src = `/api/seq/${encodeURIComponent(S.seq.seq_id)}/frame/${index}`;
  });
  imgCache.set(index, entry);
  // Bound the cache; frames are 640x512 JPEG and the annotator only ever moves near the cursor.
  if (imgCache.size > 80) {
    const oldest = imgCache.keys().next().value;
    imgCache.delete(oldest);
  }
  return entry.promise;
}

function prefetch() {
  if (!S.seq) return;
  const last = S.seq.frame_count - 1;
  for (let i = S.index - 2; i <= S.index + PREFETCH_AHEAD; i++) {
    if (i >= 0 && i <= last && !imgCache.has(i)) loadImage(i).catch(() => {});
  }
}

/* ------------------------------------------------------------- rendering */

function fitView() {
  if (!S.seq) return;
  const viewport = $('viewport');
  const scale = Math.min(viewport.clientWidth / S.seq.width, viewport.clientHeight / S.seq.height) * 0.94;
  S.view.z = scale;
  S.view.x = (viewport.clientWidth - S.seq.width * scale) / 2;
  S.view.y = (viewport.clientHeight - S.seq.height * scale) / 2;
  applyView();
}

function applyView() {
  $('stage').style.transform = `translate(${S.view.x}px, ${S.view.y}px) scale(${S.view.z})`;
  drawBoxes();
}

function zoomAt(factor, clientX, clientY) {
  const viewport = $('viewport');
  const rect = viewport.getBoundingClientRect();
  const px = (clientX ?? rect.left + rect.width / 2) - rect.left;
  const py = (clientY ?? rect.top + rect.height / 2) - rect.top;
  const next = clamp(S.view.z * factor, 0.05, 24);
  const ratio = next / S.view.z;
  S.view.x = px - (px - S.view.x) * ratio;
  S.view.y = py - (py - S.view.y) * ratio;
  S.view.z = next;
  applyView();
}

function drawImage() {
  const canvas = $('img-canvas');
  if (!S.seq) return;
  if (canvas.width !== S.seq.width || canvas.height !== S.seq.height) {
    canvas.width = S.seq.width;
    canvas.height = S.seq.height;
    $('box-canvas').width = S.seq.width;
    $('box-canvas').height = S.seq.height;
  }
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const requested = S.index;
  loadImage(requested).then((img) => {
    // A slow frame must never overwrite a newer one that arrived while it was decoding.
    if (requested !== S.index) return;
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
  }).catch((error) => {
    if (requested === S.index) status(error.message, 'err');
  });
}

function drawBoxes() {
  const canvas = $('box-canvas');
  if (!canvas.width) return;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const z = S.view.z;
  const inv = 1 / z;

  const proposals = visibleProposals();
  ctx.lineWidth = 1 * inv;
  ctx.strokeStyle = 'rgba(122, 200, 255, 0.85)';
  ctx.setLineDash([4 * inv, 3 * inv]);
  for (const b of proposals) {
    ctx.strokeRect(b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1);
  }
  ctx.setLineDash([]);

  S.labels.forEach((b, i) => {
    const selected = i === S.selected;
    ctx.lineWidth = (selected ? 2 : 1.5) * inv;
    ctx.strokeStyle = selected ? '#ffd54a' : '#37c07a';
    const w = Math.max(b.x2 - b.x1, MIN_SIZE), h = Math.max(b.y2 - b.y1, MIN_SIZE);
    ctx.strokeRect(b.x1, b.y1, w, h);

    if (selected) {
      ctx.fillStyle = '#ffd54a';
      const size = HANDLE * 2 * inv;
      for (const [hx, hy] of handlePoints(b, w, h)) {
        ctx.fillRect(hx - size / 2, hy - size / 2, size, size);
      }
      const label = `${(w).toFixed(0)}×${(h).toFixed(0)}`;
      ctx.font = `${11 * inv}px ui-monospace, Menlo, monospace`;
      const textWidth = ctx.measureText(label).width;
      ctx.fillStyle = 'rgba(255, 213, 74, 0.92)';
      ctx.fillRect(b.x1, b.y1 - 14 * inv, textWidth + 6 * inv, 13 * inv);
      ctx.fillStyle = '#12151b';
      ctx.fillText(label, b.x1 + 3 * inv, b.y1 - 4 * inv);
    }
  });
}

function handlePoints(b) {
  const w = Math.max(b.x2 - b.x1, MIN_SIZE), h = Math.max(b.y2 - b.y1, MIN_SIZE);
  const mx = b.x1 + w / 2, my = b.y1 + h / 2;
  return [
    [b.x1, b.y1], [mx, b.y1], [b.x2, b.y1],
    [b.x1, my], [b.x2, my],
    [b.x1, b.y2], [mx, b.y2], [b.x2, b.y2],
  ];
}

function toImageCoords(event) {
  const rect = $('viewport').getBoundingClientRect();
  return {
    x: (event.clientX - rect.left - S.view.x) / S.view.z,
    y: (event.clientY - rect.top - S.view.y) / S.view.z,
  };
}

/* -------------------------------------------------------------- history */

function pushUndo() {
  S.undo.push({ labels: S.labels.map((b) => ({ ...b })), index: S.index });
  if (S.undo.length > 120) S.undo.shift();
  S.redo.length = 0;
  S.dirty = true;
}

/**
 * Record a multi-frame edit so Ctrl+Z can take it back.
 *
 * Propagating or interpolating writes a dozen-plus frames to disk in one request, and a single-frame
 * snapshot cannot reverse that. It is tempting to record these frames as "they were empty", since both
 * operations normally fill gaps — but neither guarantees it: propagation only stops at an annotation it
 * can match, and interpolation overwrites whatever sits between the two endpoints. So the real previous
 * content of every touched frame is captured, which also keeps redo exact.
 */
function pushBatchUndo(entries) {
  const copy = (boxes) => (boxes || []).map((b) => ({ ...b }));
  S.undo.push({
    kind: 'batch',
    index: S.index,
    previous: entries.map((entry) => ({ index: entry.index, boxes: copy(recentLabels.get(entry.index)) })),
    // `applied` is what is about to land on disk, not a second look at recentLabels — that still holds
    // the old content at this point, so capturing both sides the same way makes redo restore the undo.
    applied: entries.map((entry) => ({ index: entry.index, boxes: copy(entry.boxes) })),
  });
  if (S.undo.length > 120) S.undo.shift();
  S.redo.length = 0;
  S.dirty = true;
}

/** Write a batch snapshot back to the server and refresh the frame it was triggered from. */
async function restoreBatch(frames, index, label) {
  const entries = frames.map((frame) => ({ index: frame.index, boxes: frame.boxes || [] }));
  for (const entry of entries) recentLabels.set(entry.index, entry.boxes || []);
  S.index = index;
  S.selected = -1;
  drawTimeline();
  await saveBatch(entries, label);
  return goToFrame(index, { fromHistory: true });
}

function undo() {
  const entry = S.undo.pop();
  if (!entry) return status('没有可撤销的操作');
  if (entry.kind === 'batch') {
    // A batch entry is a self-contained OLD -> NEW transition, so redo replays the *same* record.
    // Swapping the two sides here (as the single-frame path does with a live snapshot) makes redo
    // restore the undo instead.
    S.redo.push(entry);
    return restoreBatch(entry.previous, entry.index, '撤销');
  }
  S.redo.push({ labels: S.labels.map((b) => ({ ...b })), index: S.index });
  S.index = entry.index;
  S.labels = entry.labels;
  S.selected = -1;
  return goToFrame(S.index, { fromHistory: true });
}

function redo() {
  const entry = S.redo.pop();
  if (!entry) return status('没有可重做的操作');
  if (entry.kind === 'batch') {
    S.undo.push(entry);
    return restoreBatch(entry.applied, entry.index, '重做');
  }
  S.undo.push({ labels: S.labels.map((b) => ({ ...b })), index: S.index });
  S.index = entry.index;
  S.labels = entry.labels;
  S.selected = -1;
  return goToFrame(S.index, { fromHistory: true });
}

/* --------------------------------------------------------------- loading */

async function goToFrame(index, options = {}) {
  if (!S.seq) return;
  const last = S.seq.frame_count - 1;
  const target = clamp(index, 0, last);
  if (target === S.index && !options.force && S.loaded) return;

  if (S.dirty && !options.fromHistory) await saveFrame({ silent: true, markReviewed: false });

  S.index = target;
  S.loaded = false;
  S.drawMode = false;
  S.drag = null;
  $('frame-input').value = target;
  $('frame-total').textContent = `/ ${S.seq.frame_count}`;

  try {
    const payload = await api(`/api/seq/${encodeURIComponent(S.seq.seq_id)}/labels/${target}`);
    S.labels = payload.labels || [];
    S.prelabels = payload.prelabels || [];
    S.scores = payload.scores || [];
    S.selected = S.labels.length ? 0 : -1;
    S.dirty = false;
    S.loaded = true;
    recentLabels.set(target, S.labels);
    trimRecent();
    await loadImage(target);
    drawImage();
    drawBoxes();
    drawTimeline();
    updateCursor();
    updateSidebars(payload);
    prefetch();
    if (S.follow && S.labels.length && !options.fromHistory) {
      // Keep the first box tracked so the annotator only nudges, never re-draws.
      S.selected = 0;
      drawBoxes();
    }
  } catch (error) {
    status(error.message, 'err');
  }
}

const recentLabels = new Map();
function trimRecent() {
  while (recentLabels.size > 80) {
    recentLabels.delete(recentLabels.keys().next().value);
  }
}

/* --------------------------------------------------------------- saving */

async function saveFrame(options = {}) {
  if (!S.seq || !S.loaded) return;
  if (!S.dirty && !options.force) return;
  try {
    await postJSON(`/api/seq/${encodeURIComponent(S.seq.seq_id)}/labels/${S.index}`, {
      boxes: S.labels,
      reviewed: options.markReviewed !== false,
    });
    S.dirty = false;
    if (options.markReviewed !== false && S.summary) {
      recentLabels.set(S.index, S.labels);
      S.summary.counts.reviewed += 1;
      if (S.summary.counts.boxes != null) S.summary.counts.boxes += S.labels.length;
      renderSequenceList();
    }
    if (options.advance && !options.silent) {
      await goToFrame(S.index + 1);
    }
    updateFrameState();
  } catch (error) {
    status(`保存失败：${error.message}`, 'err');
  }
}

async function saveBatch(entries, label) {
  if (!entries.length) return;
  try {
    await postJSON(`/api/seq/${encodeURIComponent(S.seq.seq_id)}/batch`, { entries });
    if (S.summary) {
      S.summary.counts.reviewed += entries.length;
      S.summary.counts.labelled += entries.length;
      if (S.summary.counts.boxes != null) {
        S.summary.counts.boxes += entries.reduce((total, entry) => total + (entry.boxes || []).length, 0);
      }
      renderSequenceList();
    }
    status(`${label}：写入 ${entries.length} 帧`, 'ok');
    drawTimeline();
  } catch (error) {
    status(`批量写入失败：${error.message}`, 'err');
  }
}

/* ------------------------------------------------------- edit operations */

function addBox(box) {
  pushUndo();
  S.labels.push(normalizeBox(box.x1, box.y1, box.x2, box.y2));
  S.selected = S.labels.length - 1;
  drawBoxes();
  updateBoxList();
}

function deleteSelected() {
  if (S.selected < 0 || !S.labels.length) return status('没有选中的框');
  pushUndo();
  S.labels.splice(S.selected, 1);
  S.selected = Math.min(S.selected, S.labels.length - 1);
  drawBoxes();
  updateBoxList();
  saveFrame({ markReviewed: true });
}

function acceptProposals() {
  if (!S.prelabels.length) {
    status('本帧没有模型候选', 'err');
    return goToFrame(S.index + 1);
  }
  pushUndo();
  S.labels = S.prelabels.map((b) => ({ ...b }));
  S.selected = S.labels.length ? 0 : -1;
  // pushUndo() already marked the frame dirty, but the save is forced anyway: this is the tool's
  // primary action and it must never be a no-op on a frame the annotator had not edited. Without the
  // flag, saveFrame() early-returns on a clean frame and the accepted proposals would live only in
  // memory — no file written, no advance.
  drawBoxes();
  updateBoxList();
  return saveFrame({ force: true, markReviewed: true, advance: true });
}

async function copyFromPrevious() {
  const index = S.index - 1;
  if (index < 0) return status('已经是第一帧');
  // recentLabels only holds frames visited in this session, so after a jump the previous frame is
  // usually absent even though it has annotations on disk. Fall back to the server rather than telling
  // the annotator "上一帧没有框" when the label is right there.
  let prev = recentLabels.get(index);
  if (!prev) {
    try {
      const payload = await api(`/api/seq/${encodeURIComponent(S.seq.seq_id)}/labels/${index}`);
      prev = payload.labels || [];
      recentLabels.set(index, prev);
    } catch (error) {
      status(`读取上一帧失败：${error.message}`, 'err');
      return;
    }
  }
  if (!prev.length) return status('上一帧没有框');
  pushUndo();
  S.labels = prev.map((b) => ({ ...b }));
  S.selected = 0;
  drawBoxes();
  updateBoxList();
  saveFrame({ markReviewed: true });
  status(`已复制上一帧的 ${S.labels.length} 个框`, 'ok');
}

/** Extrapolate the selected track forward, stopping where a real annotation already exists. */
function propagateForward(frames) {
  if (S.selected < 0 || !S.labels.length) return status('先选中一个框再传播');
  const box = S.labels[S.selected];
  const center = boxCenter(box);
  const w = box.x2 - box.x1, h = box.y2 - box.y1;

  // Velocity from the nearest annotated box in a recently visited earlier frame.
  let vx = 0, vy = 0, found = false;
  for (let back = 1; back <= 12 && !found; back++) {
    const prev = recentLabels.get(S.index - back);
    if (!prev || !prev.length) continue;
    const match = nearBox(prev, center.x, center.y, Math.max(60, w * 6));
    if (match) {
      const c = boxCenter(match);
      vx = (center.x - c.x) / back;
      vy = (center.y - c.y) / back;
      found = true;
    }
  }

  const entries = [];
  let stopped = 0;
  for (let k = 1; k <= frames; k++) {
    const idx = S.index + k;
    if (idx >= S.seq.frame_count) break;
    const px = center.x + vx * k;
    const py = center.y + vy * k;
    const existing = recentLabels.get(idx);
    if (existing && nearBox(existing, px, py, Math.max(w, h))) { stopped = k; break; }
    entries.push({ index: idx, boxes: [{ cls: 0, x1: px - w / 2, y1: py - h / 2, x2: px + w / 2, y2: py + h / 2 }] });
  }
  if (!entries.length) {
    return status(stopped ? `第 ${S.index + stopped} 帧已有标注，传播已停止` : '没有可写入的目标帧');
  }
  pushBatchUndo(entries);
  for (const entry of entries) recentLabels.set(entry.index, entry.boxes);
  const speed = Math.hypot(vx, vy);
  status(
    `传播 ${entries.length} 帧${stopped ? `，在第 ${S.index + stopped} 帧停止` : ''}` +
    `（速度 ${speed.toFixed(2)} px/帧${found ? '' : '，未找到历史，按静止处理'}）`,
    'ok'
  );
  drawTimeline();
  return saveBatch(entries, '传播');
}

/** Linearly interpolate this frame's boxes onto the next frame that already has annotations. */
function interpolateToNext() {
  if (!S.labels.length) return status('当前帧没有框');
  const current = S.labels;
  let targetIndex = -1;
  for (let i = S.index + 1; i < Math.min(S.index + 60, S.seq.frame_count); i++) {
    if (recentLabels.has(i) && recentLabels.get(i).length) { targetIndex = i; break; }
  }
  if (targetIndex < 0) return status('后续 60 帧内没有已标注帧');
  const target = recentLabels.get(targetIndex);
  const pair = nearBox(target, boxCenter(current[0]).x, boxCenter(current[0]).y, 1e6);
  if (!pair) return status('下一标注帧没有对应框');

  const a = boxCenter(current[S.selected >= 0 ? S.selected : 0]);
  const b = boxCenter(pair);
  const span = targetIndex - S.index;
  const entries = [];
  for (let i = 1; i < span; i++) {
    const t = i / span;
    const cx = a.x + (b.x - a.x) * t;
    const cy = a.y + (b.y - a.y) * t;
    const src = current[S.selected >= 0 ? S.selected : 0];
    const w = src.x2 - src.x1, h = src.y2 - src.y1;
    entries.push({ index: S.index + i, boxes: [{ cls: 0, x1: cx - w / 2, y1: cy - h / 2, x2: cx + w / 2, y2: cy + h / 2 }] });
  }
  pushBatchUndo(entries);
  for (const entry of entries) recentLabels.set(entry.index, entry.boxes);
  status(`插补 ${entries.length} 帧至第 ${targetIndex} 帧`, 'ok');
  drawTimeline();
  return saveBatch(entries, '插补');
}

function nextUnreviewed() {
  if (!S.summary || !S.seq) return;
  const last = S.seq.frame_count - 1;
  for (let i = S.index + 1; i <= last; i++) {
    if (!isReviewed(S.summary.reviewed, i)) return goToFrame(i);
  }
  for (let i = 0; i <= last; i++) {
    if (!isReviewed(S.summary.reviewed, i)) return goToFrame(i);
  }
  status('该序列已全部复核完成 🎉', 'ok');
}

/* ------------------------------------------------- run-length decode */

function isReviewed(runs, index) {
  if (!runs) return false;
  let cursor = 0;
  for (const [value, count] of runs) {
    if (index < cursor + count) return value === 1;
    cursor += count;
  }
  return false;
}

/* --------------------------------------------------------------- sidebar */

function renderSequenceList() {
  const filter = $('seq-search').value.trim().toLowerCase();
  const list = $('seq-list');
  list.innerHTML = '';
  for (const seq of S.sequences) {
    if (filter && !seq.name.toLowerCase().includes(filter) && !seq.seq_id.toLowerCase().includes(filter)) continue;
    const summary = seq._summary;
    const row = document.createElement('div');
    row.className = 'seq-item' + (S.seq && S.seq.seq_id === seq.seq_id ? ' active' : '');
    const counts = summary ? summary.counts : null;
    // frame_count lives on the summary root, not inside counts.
    const totalFrames = summary ? (summary.frame_count || seq.frame_count || 0) : 0;
    const pct = counts && totalFrames ? Math.round(100 * counts.reviewed / totalFrames) : 0;
    row.innerHTML = `
      <div class="name">${escapeHtml(seq.name)}</div>
      <div class="meta">
        <span>${seq.kind === 'video' ? '视频' : '帧目录'}</span>
        <span>${seq.frame_count} 帧</span>
        <span>${seq.width}×${seq.height}</span>
      </div>
      <div class="bar"><i style="width:${pct}%"></i></div>
      <div class="meta"><span>复核 ${pct}%${counts && counts.boxes != null ? ` · 框 ${counts.boxes}` : ''}</span></div>`;
    row.onclick = () => selectSequence(seq.seq_id);
    list.appendChild(row);
  }
}

function escapeHtml(text) {
  return String(text).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

function updateSidebars(payload) {
  const summary = S.summary;
  const boxes = summary && summary.counts.boxes != null ? summary.counts.boxes : '…';
  $('frame-stats').innerHTML = summary
    ? `<div>序列帧数 <b>${summary.frame_count}</b></div>
       <div>已标注 <b>${summary.counts.labelled}</b> · 已复核 <b>${summary.counts.reviewed}</b></div>
       <div>模型候选 <b>${summary.counts.prelabelled}</b> · 总框数 <b>${boxes}</b></div>`
    : '';
  $('frame-state').textContent = payload.reviewed ? '已复核' : (S.labels.length ? '已标注' : '未标注');
  updateBoxList();
}

function updateBoxList() {
  const list = $('box-list');
  const rows = [];
  S.labels.forEach((b, i) => {
    const w = Math.max(0, Math.round(b.x2 - b.x1)), h = Math.max(0, Math.round(b.y2 - b.y1));
    const c = boxCenter(b);
    rows.push(`<div class="box-row ${i === S.selected ? 'sel' : ''}" data-i="${i}">
      <span class="swatch" style="background:${i === S.selected ? '#ffd54a' : '#37c07a'}"></span>
      <span class="dims">${w}×${h} @ ${Math.round(c.x)},${Math.round(c.y)}</span>
    </div>`);
  });
  const proposals = visibleProposals();
  proposals.slice(0, 40).forEach((b, i) => {
    const w = Math.max(0, Math.round(b.x2 - b.x1)), h = Math.max(0, Math.round(b.y2 - b.y1));
    const c = boxCenter(b);
    // `i` indexes the filtered list, so the score must be looked up by the original prelabel index.
    const score = S.scores[proposalSourceIndex(i)];
    rows.push(`<div class="box-row" data-proposal="${i}">
      <span class="swatch" style="background:#7ac8ff"></span>
      <span class="dims">${w}×${h} @ ${Math.round(c.x)},${Math.round(c.y)}</span>
      <span class="src">${score == null ? '导入' : score.toFixed(2)}</span>
    </div>`);
  });
  list.innerHTML = rows.join('') || '<div class="stats">本帧没有框</div>';
  $('box-count').textContent = `${S.labels.length} / ${proposals.length}`;
  list.querySelectorAll('[data-i]').forEach((row) => {
    row.onclick = () => { S.selected = Number(row.dataset.i); drawBoxes(); updateBoxList(); };
  });
  list.querySelectorAll('[data-proposal]').forEach((row) => {
    row.onclick = () => addBox(visibleProposals()[Number(row.dataset.proposal)]);
  });
}

function updateFrameState() {
  $('frame-state').textContent = S.dirty ? '未保存' : (S.labels.length ? '已标注' : '空帧');
}

function updateCursor() {
  if (!S.seq || !S.seq.frame_count) return;
  const wrap = $('timeline-wrap');
  const pct = S.index / Math.max(1, S.seq.frame_count - 1);
  $('timeline-cursor').style.left = `${pct * wrap.clientWidth}px`;
}

/* -------------------------------------------------------------- timeline */

function drawTimeline() {
  const canvas = $('timeline');
  const wrap = $('timeline-wrap');
  const width = wrap.clientWidth, height = wrap.clientHeight;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = width * dpr;
  canvas.height = height * dpr;
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);
  if (!S.seq || !S.seq.frame_count) return;

  const total = S.seq.frame_count;
  const barTop = 4, barH = height - 8;
  ctx.fillStyle = '#0e1116';
  ctx.fillRect(0, barTop, width, barH);

  const summary = S.summary;
  if (summary) {
    const drawRuns = (runs, y, h, color) => {
      let cursor = 0;
      for (const [value, count] of runs) {
        if (value) {
          const x0 = (cursor / total) * width;
          const x1 = ((cursor + count) / total) * width;
          ctx.fillStyle = color;
          ctx.fillRect(x0, y, Math.max(1, x1 - x0), h);
        }
        cursor += count;
      }
    };
    drawRuns(summary.prelabelled, barTop, barH, 'rgba(74, 158, 226, 0.55)');
    drawRuns(summary.labelled, barTop, barH, 'rgba(55, 192, 122, 0.45)');
    drawRuns(summary.reviewed, barTop, barH, '#37c07a');
  }
  // Commit history is only known locally; show propagated/edited frames as they happen.
  let cursor = 0;
  for (const [value, count] of recentRuns(summary)) {
    if (value) {
      const x0 = (cursor / total) * width;
      ctx.fillStyle = '#e0a53a';
      ctx.fillRect(x0, barTop, Math.max(1, (count / total) * width), 3);
    }
    cursor += count;
  }
}

function recentRuns(summary) {
  const total = S.seq ? S.seq.frame_count : 0;
  if (!total) return [];
  const flags = new Uint8Array(total);
  for (const index of recentLabels.keys()) if (index < total) flags[index] = 1;
  const runs = [];
  for (let i = 0; i < total; i++) {
    if (runs.length && runs[runs.length - 1][0] === flags[i]) runs[runs.length - 1][1]++;
    else runs.push([flags[i], 1]);
  }
  return runs;
}

/* ---------------------------------------------------------- mouse input */

function bindMouse() {
  const viewport = $('viewport');

  viewport.addEventListener('mousedown', (event) => {
    if (!S.seq || event.button !== 0) return;
    const { x, y } = toImageCoords(event);

    if (S.drawMode) {
      S.drag = { type: 'create', x1: x, y1: y, x2: x, y2: y };
      return;
    }
    const hit = hitBox(x, y);
    if (hit) {
      S.selected = hit.index;
      pushUndo();
      const handle = hitHandle(hit.box, x, y);
      S.drag = handle
        ? { type: 'resize', index: hit.index, handle, origin: { ...hit.box } }
        : { type: 'move', index: hit.index, offsetX: x - hit.box.x1, offsetY: y - hit.box.y1 };
      drawBoxes();
      updateBoxList();
    } else {
      S.selected = -1;
      drawBoxes();
      updateBoxList();
    }
  });

  window.addEventListener('mousemove', (event) => {
    if (!S.drag) return;
    const { x, y } = toImageCoords(event);
    if (S.drag.type === 'create') {
      S.drag.x2 = x; S.drag.y2 = y;
      drawPreview(S.drag);
    } else if (S.drag.type === 'move') {
      const box = S.labels[S.drag.index];
      const w = box.x2 - box.x1, h = box.y2 - box.y1;
      box.x1 = clamp(x - S.drag.offsetX, 0, S.seq.width - w);
      box.y1 = clamp(y - S.drag.offsetY, 0, S.seq.height - h);
      box.x2 = box.x1 + w; box.y2 = box.y1 + h;
      drawBoxes();
    } else if (S.drag.type === 'resize') {
      applyResize(S.drag, x, y);
      drawBoxes();
    }
  });

  window.addEventListener('mouseup', () => {
    if (!S.drag) return;
    const drag = S.drag;
    S.drag = null;
    if (drag.type === 'create') {
      const box = normalizeBox(drag.x1, drag.y1, drag.x2, drag.y2);
      if (box.x2 - box.x1 < MIN_SIZE || box.y2 - box.y1 < MIN_SIZE) {
        drawBoxes();
        return;
      }
      // The create-drag path never pushed an undo entry, so addBox() is what records history here.
      addBox(box);
      S.drawMode = false;
    }
    S.dirty = true;
    updateFrameState();
    saveFrame({ markReviewed: true });
  });

  viewport.addEventListener('wheel', (event) => {
    if (!S.seq) return;
    event.preventDefault();
    if (event.ctrlKey || event.metaKey || event.shiftKey) {
      zoomAt(event.deltaY < 0 ? 1.15 : 1 / 1.15, event.clientX, event.clientY);
    } else {
      S.view.x -= event.deltaX;
      S.view.y -= event.deltaY;
      applyView();
    }
  }, { passive: false });

  viewport.addEventListener('dblclick', () => { S.drawMode = true; status('拖拽绘制新框（Esc 取消）'); });
}

function drawPreview(rect) {
  drawBoxes();
  const ctx = $('box-canvas').getContext('2d');
  const inv = 1 / S.view.z;
  ctx.lineWidth = 1.5 * inv;
  ctx.strokeStyle = '#ffd54a';
  ctx.setLineDash([5 * inv, 3 * inv]);
  ctx.strokeRect(rect.x1, rect.y1, rect.x2 - rect.x1, rect.y2 - rect.y1);
  ctx.setLineDash([]);
}

function hitBox(x, y) {
  // Prefer the smallest box under the cursor so a proposal inside a bigger box stays selectable.
  let best = null;
  S.labels.forEach((b, index) => {
    if (x >= b.x1 && x <= b.x2 && y >= b.y1 && y <= b.y2) {
      const area = (b.x2 - b.x1) * (b.y2 - b.y1);
      if (!best || area < best.area) best = { index, box: b, area };
    }
  });
  return best;
}

function hitHandle(box, x, y) {
  const inv = 1 / S.view.z;
  const tolerance = HANDLE * 1.6 * inv;
  const points = handlePoints(box);
  const names = ['nw', 'n', 'ne', 'w', 'e', 'sw', 's', 'se'];
  for (let i = 0; i < points.length; i++) {
    if (Math.abs(x - points[i][0]) <= tolerance && Math.abs(y - points[i][1]) <= tolerance) return names[i];
  }
  return null;
}

function applyResize(drag, x, y) {
  const box = S.labels[drag.index];
  const o = drag.origin;
  const w = o.x2 - o.x1, h = o.y2 - o.y1;
  const left = o.x1, top = o.y1, right = o.x2, bottom = o.y2;
  let nx1 = left, ny1 = top, nx2 = right, ny2 = bottom;
  if (drag.handle.includes('w')) nx1 = clamp(x, 0, right - MIN_SIZE);
  if (drag.handle.includes('e')) nx2 = clamp(x, nx1 + MIN_SIZE, S.seq.width);
  if (drag.handle.includes('n')) ny1 = clamp(y, 0, bottom - MIN_SIZE);
  if (drag.handle.includes('s')) ny2 = clamp(y, ny1 + MIN_SIZE, S.seq.height);
  box.x1 = nx1; box.y1 = ny1; box.x2 = nx2; box.y2 = ny2;
}

/* -------------------------------------------------------------- keyboard */

function bindKeys() {
  window.addEventListener('keydown', (event) => {
    if (event.target.tagName === 'INPUT' && event.target.type === 'number' && event.key !== 'Escape') {
      if (event.key === 'Enter') event.target.blur();
      return;
    }
    if (event.target.tagName === 'INPUT' && event.target.type === 'text') return;
    const ctrl = event.ctrlKey || event.metaKey;
    const step = Number($('fast-step').value) || 10;

    if (ctrl && event.key.toLowerCase() === 's') { event.preventDefault(); return saveFrame({ force: true, markReviewed: true }); }
    if (ctrl && event.key.toLowerCase() === 'z' && !event.shiftKey) { event.preventDefault(); return undo(); }
    if (ctrl && (event.key.toLowerCase() === 'y' || (event.key.toLowerCase() === 'z' && event.shiftKey))) { event.preventDefault(); return redo(); }

    switch (event.key) {
      case 'a': case 'A': case 'ArrowLeft':
        event.preventDefault();
        return goToFrame(S.index - (event.shiftKey ? step : 1));
      case 'd': case 'D': case 'ArrowRight':
        event.preventDefault();
        return goToFrame(S.index + (event.shiftKey ? step : 1));
      case 'Home': event.preventDefault(); return goToFrame(0);
      case 'End': event.preventDefault(); return goToFrame(S.seq.frame_count - 1);
      case ' ':
        event.preventDefault();
        return acceptProposals();
      case 'Enter':
        event.preventDefault();
        return saveFrame({ force: true, markReviewed: true, advance: true });
      case 'n': case 'N':
        S.drawMode = true;
        return status('拖拽绘制新框（Esc 取消）');
      case 'Escape':
        S.drawMode = false; S.drag = null;
        return drawBoxes();
      case 'Delete': case 'Backspace':
        event.preventDefault();
        return deleteSelected();
      case 'p': case 'P':
        return propagateForward(Number($('prop-frames').value) || 20);
      case 'i': case 'I':
        return interpolateToNext();
      case 'c': case 'C':
        return copyFromPrevious();
      case 'f': case 'F':
        return nextUnreviewed();
      case 'Tab':
        event.preventDefault();
        S.showProposals = !S.showProposals;
        $('chk-proposals').checked = S.showProposals;
        drawBoxes(); updateBoxList();
        return;
      case '[':
        $('min-score').value = clamp(S.minScore - 0.05, 0, 0.95);
        return applyThreshold();
      case ']':
        $('min-score').value = clamp(S.minScore + 0.05, 0, 0.95);
        return applyThreshold();
      case '+': case '=':
        return zoomAt(1.2);
      case '-': case '_':
        return zoomAt(1 / 1.2);
      case '0':
        return fitView();
      case 'R':
        if (event.shiftKey) return saveFrame({ force: true, markReviewed: true });
        return;
      default:
        return;
    }
  });
}

function applyThreshold() {
  S.minScore = Number($('min-score').value);
  $('min-score-val').textContent = S.minScore.toFixed(2);
  drawBoxes();
  updateBoxList();
}

/* --------------------------------------------------------- sequence ops */

async function selectSequence(seqId) {
  const seq = S.sequences.find((item) => item.seq_id === seqId);
  if (!seq) return;
  S.seq = seq;
  S.index = 0;
  S.loaded = false;
  S.labels = []; S.prelabels = []; S.scores = [];
  recentLabels.clear();
  imgCache.clear();
  try {
    const info = await api(`/api/seq/${encodeURIComponent(seqId)}/summary`);
    S.seq = { ...seq, ...info };
    S.summary = info.summary;
    seq._summary = info.summary;
    renderSequenceList();
    drawTimeline();
    loadBoxTotal(seqId);
  } catch (error) {
    status(error.message, 'err');
  }
  $('empty-hint').classList.add('hidden');
  renderSequenceList();
  fitView();
  await goToFrame(0, { force: true });
  refreshJobs();
}

/** Fetch the exact box total separately so a long sequence does not stall the timeline. */
async function loadBoxTotal(seqId) {
  try {
    const info = await api(`/api/seq/${encodeURIComponent(seqId)}/summary?boxes=1`);
    if (!S.seq || S.seq.seq_id !== seqId) return;
    S.summary.counts.boxes = info.summary.counts.boxes;
    S.summary.boxes_total_known = true;
    const seq = S.sequences.find((item) => item.seq_id === seqId);
    if (seq) seq._summary = S.summary;
    renderSequenceList();
    updateSidebars({ reviewed: false });
  } catch (error) {
    /* the total is cosmetic; ignore */
  }
}

async function loadSequences() {
  const state = await api('/api/state');
  S.sequences = state.sequences;
  renderSequenceList();
  if (state.scan_error) status(`扫描告警：${state.scan_error}`, 'err');
}

/* ------------------------------------------------------- pre-annotation */

function openPreannotate() {
  if (!S.seq) return status('先选择序列', 'err');
  const modal = $('modal');
  modal.classList.remove('hidden');
  $('modal-title').textContent = `预标注 · ${S.seq.name}`;
  const all = S.sequences.filter((s) => s.kind === 'video' || s.kind === 'frames');
  $('modal-body').innerHTML = `
    <div class="note">推理在训练服务器上执行（多尺度 Trial 0474 热图）。本机 torch 环境不可用，
    因此模型只在服务器加载；点击开始后不会阻塞标注界面。</div>
    <label class="chk" style="display:flex;gap:6px;align-items:center">
      <input id="pa-all" type="checkbox"> 预标注<b>全部 ${all.length}</b> 个序列（按 GPU 轮询分片）
    </label>
    <div id="pa-picker" style="max-height:190px;overflow:auto;border:1px solid var(--line);border-radius:6px;padding:6px">
      ${all.map((s) => `<label class="chk pa-item" data-id="${escapeHtml(s.seq_id)}" style="display:flex;gap:6px;padding:3px 4px">
        <input type="checkbox" data-seq="${escapeHtml(s.seq_id)}" checked> ${escapeHtml(s.name)}
        <span style="margin-left:auto;color:var(--fg-dim)">${s.frame_count}</span></label>`).join('')}
    </div>
    <label><span>GPU（逗号分隔，按序轮询分配）</span><input id="pa-gpus" value="0,1,2,3"></label>
    <label><span>额外尺度（原生 letterbox 之外）</span><input id="pa-scales" value="160,80"></label>
    <label><span>融合阈值 main-threshold</span><input id="pa-th" type="number" step="0.01" value="0.22"></label>
    <label><span>限帧（0 = 全部，用于试跑）</span><input id="pa-max" type="number" step="1" value="0"></label>
    <div id="modal-log">等待开始…</div>`;

  $('pa-all').onchange = (event) => {
    document.querySelectorAll('#pa-picker input[data-seq]').forEach((box) => { box.checked = event.target.checked; });
  };
  $('modal-ok').onclick = async () => {
    $('modal-ok').disabled = true;
    const picked = [...document.querySelectorAll('#pa-picker input[data-seq]:checked')]
      .map((box) => box.dataset.seq);
    if (!picked.length) {
      $('modal-log').textContent = '没有选中任何序列';
      $('modal-ok').disabled = false;
      return;
    }
    try {
      const result = await postJSON('/api/preannotate', {
        seq_ids: picked,
        devices: $('pa-gpus').value.split(',').map((s) => s.trim()).filter(Boolean),
        scales: $('pa-scales').value,
        main_threshold: $('pa-th').value,
        max_frames: $('pa-max').value,
      });
      status(`已提交 ${result.jobs.length} 个预标注任务，GPU ${result.gpus.join(',')}`, 'ok');
      refreshJobs();
    } catch (error) {
      $('modal-log').textContent = `提交失败：${error.message}`;
      $('modal-ok').disabled = false;
    }
  };
}

async function refreshJobs() {
  try {
    const { jobs } = await api('/api/jobs');
    const active = jobs.filter((j) => j.status === 'running' || j.status === 'queued');
    const summary = active.length
      ? `${active.length} 个进行中 · ${active.filter((j) => j.status === 'running').length} 运行中 · ` +
        active.slice(0, 3).map((j) => `${j.seq_id}[gpu${j.device}] ${j.done}/${j.total || '?'}`).join('  ')
      : jobs.length ? `最近：${jobs[0].seq_id} [${jobs[0].status}] ${jobs[0].message}` : '';
    $('job-text').textContent = summary;

    const log = $('modal-log');
    if (log && !$('modal').classList.contains('hidden')) {
      if (active.length) {
        const lines = active.map((j) => {
          const pct = j.total ? Math.round(100 * j.done / j.total) : 0;
          const bar = '█'.repeat(Math.round(pct / 5)).padEnd(20, '░');
          return `${j.seq_id} [gpu${j.device}] ${j.status.padEnd(7)} ${bar} ${j.done}/${j.total || '?'} 候选${j.detections}`;
        });
        const last = active.find((j) => j.log && j.log.length);
        if (last) lines.push('', ...last.log.slice(-12));
        log.textContent = lines.join('\n');
        log.scrollTop = log.scrollHeight;
      } else if (jobs.length) {
        const done = jobs.filter((j) => j.status === 'done').length;
        const failed = jobs.filter((j) => j.status === 'failed');
        log.textContent = `完成 ${done} / ${jobs.length}` +
          (failed.length ? `\n失败：\n${failed.map((j) => `${j.seq_id}: ${j.message}`).join('\n')}` : '');
      }
      if (!active.length) {
        $('modal-ok').disabled = false;
        const anyDone = jobs.some((j) => j.status === 'done');
        if (anyDone) {
          status('预标注完成，可刷新后加载候选', 'ok');
          if (S.seq) selectSequence(S.seq.seq_id);
        }
      }
    }
  } catch (error) {
    /* job polling is best-effort */
  }
}

/* ----------------------------------------------------------------- init */

function bindUI() {
  $('btn-prev').onclick = () => goToFrame(S.index - 1);
  $('btn-next').onclick = () => goToFrame(S.index + 1);
  $('btn-accept').onclick = acceptProposals;
  $('btn-clear').onclick = () => {
    if (!S.labels.length) return;
    pushUndo();
    S.labels = []; S.selected = -1;
    drawBoxes(); updateBoxList();
    saveFrame({ markReviewed: true });
  };
  $('btn-propagate').onclick = () => propagateForward(Number($('prop-frames').value) || 20);
  $('btn-interp').onclick = interpolateToNext;
  $('btn-preannot').onclick = openPreannotate;
  $('btn-fit').onclick = fitView;
  $('btn-zoom-in').onclick = () => zoomAt(1.2);
  $('btn-zoom-out').onclick = () => zoomAt(1 / 1.2);
  $('btn-rescan').onclick = async () => { await loadSequences(); status('已重新扫描目录'); };
  $('modal-cancel').onclick = () => $('modal').classList.add('hidden');

  $('frame-input').onchange = (event) => goToFrame(Number(event.target.value) || 0);
  $('seq-search').oninput = renderSequenceList;
  $('chk-proposals').onchange = (event) => { S.showProposals = event.target.checked; drawBoxes(); updateBoxList(); };
  $('chk-follow').onchange = (event) => { S.follow = event.target.checked; };
  $('min-score').oninput = applyThreshold;

  $('timeline').onclick = (event) => {
    const rect = event.currentTarget.getBoundingClientRect();
    if (!S.seq) return;
    const ratio = (event.clientX - rect.left) / rect.width;
    goToFrame(Math.round(ratio * (S.seq.frame_count - 1)));
  };

  window.addEventListener('resize', () => { fitView(); drawTimeline(); updateCursor(); });
}

async function init() {
  bindUI();
  bindMouse();
  bindKeys();
  try {
    await loadSequences();
    if (S.sequences.length) {
      status(`发现 ${S.sequences.length} 个序列`, 'ok');
    } else {
      status('未发现任何序列', 'err');
    }
  } catch (error) {
    status(`加载失败：${error.message}`, 'err');
  }
  // Deep link: ?seq=<seq_id>&f=<frame> opens straight onto a frame, so a specific disagreement can be
  // handed to a colleague as a URL instead of "go find it yourself".
  const params = new URLSearchParams(location.search);
  const wanted = params.get('seq');
  if (wanted && S.sequences.some((s) => s.seq_id === wanted)) {
    await selectSequence(wanted);
    const frame = Number(params.get('f'));
    if (Number.isFinite(frame) && frame > 0) await goToFrame(frame);
  }
  S.jobTimer = setInterval(refreshJobs, 2000);
}

init();

/* Debug/test handle. A top-level ``const`` lives in the global lexical scope and is NOT reachable as
 * ``window.S`` from another document, so the interactive harness (static/uitest.html) drives the app
 * through this object instead. It also makes the state inspectable from the browser console. */
window.__annotate = {
  state: S,
  goToFrame, saveFrame, saveBatch, addBox, deleteSelected, acceptProposals,
  copyFromPrevious, propagateForward, interpolateToNext, nextUnreviewed,
  applyThreshold, visibleProposals, fitView, zoomAt, pushUndo, undo, redo,
  hitBox, hitHandle, toImageCoords, updateBoxList, drawBoxes,
};
