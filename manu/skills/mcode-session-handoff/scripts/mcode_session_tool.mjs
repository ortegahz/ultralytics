#!/usr/bin/env node
/**
 * mcode 会话存储工具 —— 计算会话历史指纹 / 打包 / 校验
 *
 * 与 MCode 运行时 (0.6.12) 的算法逐位一致：
 *   activeRevision = sha256("[" + records.map(vwt).join(",") + "]")
 *   vwt(e)         = JSON.stringify(sortKeysDeep(e))   ← 默认分隔符，非紧凑
 *   sortKeysDeep   = 递归按 key 排序（仅 plain object）
 *
 * ⚠️ 三个曾经踩过的坑（详见 ../SKILL.md）：
 *   1. 用紧凑分隔符 (',',':') 算 ⇒ 哈希对不上。运行时用的是 JSON.stringify 默认分隔符。
 *   2. 用 ensure_ascii=True 的 Python 算 ⇒ 哈希对不上（非 ASCII 被转成 \uXXXX）。
 *      ⇒ 指纹计算一律用本文件（Node），不要用 Python 重写。
 *   3. 直接对文件字节做 sha256 ⇒ 对不上。指纹是「规范化后的记录数组」的哈希，
 *      不是文件字节的哈希。
 *
 * 用法：
 *   node mcode_session_tool.mjs fingerprint <sessionDir>      # 打印 activeRevision + 校验 catalog
 *   node mcode_session_tool.mjs verify <sessionDir>            # 只校验，退出码 0/1
 *   node mcode_session_tool.mjs pack <sessionDir> <outFile>    # 打包会话目录（含完整性清单）
 *   node mcode_session_tool.mjs unpack <packFile> <outDir>     # 解包并校验
 */
import { readFileSync, writeFileSync, existsSync, mkdirSync, readdirSync, statSync, rmSync } from "node:fs";
import { createHash } from "node:crypto";
import { join, relative, dirname } from "node:path";

/** 递归按 key 排序（复刻运行时 oSe） */
function sortKeysDeep(e) {
  if (Array.isArray(e)) return e.map(sortKeysDeep);
  if (e !== null && typeof e === "object") {
    const proto = Object.getPrototypeOf(e);
    if (proto === Object.prototype || proto === null) {
      return Object.fromEntries(Object.keys(e).sort().map((k) => [k, sortKeysDeep(e[k])]));
    }
  }
  return e;
}

/** 单条记录的规范序列化（复刻运行时 vwt） */
function vwt(e) {
  return JSON.stringify(sortKeysDeep(e));
}

/** 记录数组的指纹（复刻运行时 Cy） */
function computeRevision(records) {
  const h = createHash("sha256");
  h.update("[");
  for (let i = 0; i < records.length; i++) {
    if (i > 0) h.update(",");
    h.update(vwt(records[i]), "utf8");
  }
  h.update("]");
  return `sha256:${h.digest("hex")}`;
}

/** 读取会话目录的 messages.jsonl */
function readRecords(sessionDir) {
  const p = join(sessionDir, "messages.jsonl");
  if (!existsSync(p)) throw new Error(`messages.jsonl 不存在：${p}`);
  const raw = readFileSync(p, "utf8");
  const lines = raw.split("\n").filter((l) => l.trim().length > 0);
  const records = [];
  for (let i = 0; i < lines.length; i++) {
    try {
      records.push(JSON.parse(lines[i]));
    } catch (e) {
      throw new Error(`messages.jsonl 第 ${i + 1} 行不是合法 JSON：${e.message}`);
    }
  }
  return { records, raw, lineCount: lines.length };
}

/** 会话目录指纹 + 与 catalog 交叉校验 */
export function fingerprint(sessionDir) {
  const { records, lineCount } = readRecords(sessionDir);
  const revision = computeRevision(records);
  const out = {
    sessionDir,
    messageCount: records.length,
    lineCount,
    activeRevision: revision,
    fileSha256: createHash("sha256").update(readFileSync(join(sessionDir, "messages.jsonl"))).digest("hex"),
  };
  const catPath = join(sessionDir, "history-catalog.json");
  if (existsSync(catPath)) {
    try {
      const cat = JSON.parse(readFileSync(catPath, "utf8"));
      out.catalogRevision = cat.activeRevision;
      out.catalogMatches = cat.activeRevision === revision;
      out.catalogGeneration = cat.activeGeneration;
      out.catalogSessionId = cat.sessionId;
    } catch (e) {
      out.catalogError = String(e.message);
      out.catalogMatches = false;
    }
  } else {
    out.catalogMatches = null; // 缺 catalog ⇒ 运行时会自动重建，不是错误
  }
  return out;
}

/** 递归列出目录内所有文件（相对路径 + sha256） */
export function listFiles(dir, base = dir) {
  const out = [];
  for (const name of readdirSync(dir).sort()) {
    const full = join(dir, name);
    const st = statSync(full);
    if (st.isDirectory()) {
      out.push(...listFiles(full, base));
    } else if (st.isFile()) {
      out.push({
        path: relative(base, full),
        size: st.size,
        sha256: createHash("sha256").update(readFileSync(full)).digest("hex"),
      });
    }
  }
  return out;
}

/** 打包：会话目录 → tar 风格的 JSON 归档（含逐文件 sha256 清单） */
export function pack(sessionDir) {
  const fp = fingerprint(sessionDir);
  const files = listFiles(sessionDir);
  return {
    format: "mcode-session-pack/1",
    packedAtMs: Date.now(),
    sessionId: fp.catalogSessionId ?? null,
    activeRevision: fp.activeRevision,
    messageCount: fp.messageCount,
    files,
    payload: files.map((f) => ({
      path: f.path,
      content: readFileSync(join(sessionDir, f.path)).toString("base64"),
    })),
  };
}

/** 解包校验：逐文件比对 sha256 + 重算指纹 */
export function unpack(packObj, outDir) {
  rmSync(outDir, { recursive: true, force: true });
  mkdirSync(outDir, { recursive: true });
  for (const p of packObj.payload) {
    const full = join(outDir, p.path);
    mkdirSync(dirname(full), { recursive: true });
    writeFileSync(full, Buffer.from(p.content, "base64"));
  }
  const actual = listFiles(outDir);
  const expected = new Map(packObj.files.map((f) => [f.path, f.sha256]));
  const bad = [];
  for (const f of actual) {
    if (expected.get(f.path) !== f.sha256) bad.push(f.path);
  }
  const missing = [...expected.keys()].filter((p) => !actual.some((f) => f.path === p));
  const fp = fingerprint(outDir);
  return {
    filesChecked: actual.length,
    mismatched: bad,
    missing,
    revisionMatches: fp.activeRevision === packObj.activeRevision,
    computedRevision: fp.activeRevision,
    expectedRevision: packObj.activeRevision,
    messageCount: fp.messageCount,
  };
}

// ---------- CLI ----------
const [, , cmd, ...rest] = process.argv;
try {
  if (cmd === "fingerprint") {
    const fp = fingerprint(rest[0]);
    console.log(JSON.stringify(fp, null, 2));
  } else if (cmd === "verify") {
    const fp = fingerprint(rest[0]);
    const ok = fp.catalogMatches !== false;
    console.log(JSON.stringify({ ok, ...fp }, null, 2));
    process.exit(ok ? 0 : 1);
  } else if (cmd === "pack") {
    const p = pack(rest[0]);
    writeFileSync(rest[1], JSON.stringify(p));
    console.log(`已打包 ${p.files.length} 个文件 → ${rest[1]}`);
    console.log(`activeRevision = ${p.activeRevision}  messageCount = ${p.messageCount}`);
  } else if (cmd === "unpack") {
    const p = JSON.parse(readFileSync(rest[0], "utf8"));
    const r = unpack(p, rest[1]);
    console.log(JSON.stringify(r, null, 2));
    process.exit(r.mismatched.length === 0 && r.missing.length === 0 && r.revisionMatches ? 0 : 1);
  } else {
    console.error("用法: mcode_session_tool.mjs <fingerprint|verify|pack|unpack> ...");
    process.exit(2);
  }
} catch (e) {
  console.error(`错误：${e.message}`);
  process.exit(1);
}
