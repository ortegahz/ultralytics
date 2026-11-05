#!/usr/bin/env python3
"""
mcode 会话转移核心 —— 把一个会话从本机（host）搬到目标机（server）。

与 Codex 版 handoff_helper.py 的对应关系：
    adopt  → rewrite_session_row()   改写 DB 行（workspace_dir / project 绑定 / 状态复位）
    import → stage_session()          把会话目录 + DB 行落到目标机的 dataDir
    resume → verify_session()         无网只读验收（DB + 文件双侧对账）

设计要点（全部经实测验证，见 ../SKILL.md）：
  * 权威源是 messages.jsonl；history-catalog.json / user-message-locators.jsonl 是
    **派生索引**，运行时发现不一致会自动重建 ⇒ 不搬也行，搬了就必须自洽。
  * 会话目录名 = base64url(session_id)[:180]，不含时间戳语义 ⇒ 目标机沿用同相对目录。
  * manifest.json 里的绝对 paths.* 恢复时**不被读取**（运行时按 DB 的
    history_relative_dir 重算路径）⇒ 目录整体搬迁可行，无需改写。
  * local_runtime_session_locks 是执行权租约 ⇒ 接手时必须删除该会话的锁行。
  * local_runtime_thread_goals 与 session 是 UNIQUE 绑定 ⇒ 必须单独搬。
  * 可见性按 workspace_dir 精确匹配 ⇒ 必须改写成目标机上该 workspace 的真实路径。

用法：
    python3 mcode_handoff.py export  --session <id> --out <pack.json>
    python3 mcode_handoff.py import  --pack <pack.json> --data-dir <targetDataDir> \
                                     --workspace <targetWorkspaceDir> [--db <sqlite>]
    python3 mcode_handoff.py verify  --data-dir <dataDir> --session <id> [--db <sqlite>]
    python3 mcode_handoff.py list    [--db <sqlite>] [--workspace <dir>]
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

TOOL = Path(__file__).with_name("mcode_session_tool.mjs")


# --------------------------------------------------------------------------
# 通用
# --------------------------------------------------------------------------
def mcode_data_dir() -> Path:
    """本机 mcode 数据根（默认 ~/.minimax）。"""
    return Path(os.environ.get("MINIMAX_DATA_DIR", Path.home() / ".minimax")).expanduser()


def sessions_root(data_dir: Path) -> Path:
    return data_dir / "v2" / "sessions"


def db_path(data_dir: Path) -> Path:
    return data_dir / "v2" / "sqlite" / "runtime-state.sqlite"


def session_suffix(session_id: str) -> str:
    """会话目录后缀 = base64url(session_id)[:180]（复刻运行时 jdr()）。"""
    return base64.urlsafe_b64encode(session_id.encode()).decode().rstrip("=")[:180]


def rel_dir_for(session_id: str, created_at_ms: int) -> str:
    """会话相对目录 = YYYY/MM/DD/HH-MM-SS-mmm-session_<suffix>（UTC）。"""
    import datetime

    dt = datetime.datetime.fromtimestamp(created_at_ms / 1000, tz=datetime.timezone.utc)
    return (
        f"{dt:%Y/%m/%d}/{dt:%H-%M-%S}-{dt.microsecond // 1000:03d}"
        f"-session_{session_suffix(session_id)}"
    )


def node_fingerprint(session_dir: Path) -> dict:
    """调用 Node 工具算指纹（算法必须与运行时一致，Python 重写会因分隔符/转义出错）。"""
    out = subprocess.run(
        ["node", str(TOOL), "fingerprint", str(session_dir)],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(out.stdout)


def connect_db(db: Path) -> sqlite3.Connection:
    """打开 SQLite。必须把 WAL 一起带上，否则会漏读最近写入。"""
    con = sqlite3.connect(str(db))
    con.row_factory = sqlite3.Row
    return con


# --------------------------------------------------------------------------
# export —— 从本机抽出一个会话
# --------------------------------------------------------------------------
def export(session_id: str, out_path: Path, data_dir: Path | None = None) -> dict:
    data_dir = data_dir or mcode_data_dir()
    db = db_path(data_dir)
    con = connect_db(db)
    try:
        row = con.execute(
            "SELECT * FROM local_runtime_sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        if row is None:
            raise SystemExit(f"❌ 找不到会话 {session_id}（数据库：{db}）")

        rel = row["history_relative_dir"] or rel_dir_for(session_id, row["created_at_ms"])
        session_dir = sessions_root(data_dir) / rel
        if not (session_dir / "messages.jsonl").exists():
            raise SystemExit(f"❌ 会话目录缺失 messages.jsonl：{session_dir}")

        fp = node_fingerprint(session_dir)

        # 目标机要重建的 DB 行（剔除与机器相关的字段）
        target_row = {
            "session_id": row["session_id"],
            "record_json": row["record_json"],
            "updated_at_ms": row["updated_at_ms"],
            "columnar_version": row["columnar_version"],
            "agent_name": row["agent_name"],
            "runtime": row["runtime"],
            "session_type": row["session_type"],
            "status": row["status"],
            "archived": row["archived"],
            "visibility": row["visibility"],
            "session_kind": row["session_kind"],
            "purpose": row["purpose"],
            "purpose_kind": row["purpose_kind"],
            "origin_cron_id": row["origin_cron_id"],
            "parent_session_id": row["parent_session_id"],
            "workspace_dir": row["workspace_dir"],
            "project_workspace_dir": row["project_workspace_dir"],
            "is_default_workspace": row["is_default_workspace"],
            "title": row["title"],
            "created_at_ms": row["created_at_ms"],
            "error_message": row["error_message"],
            "error_code": row["error_code"],
            "extra_data_json": row["extra_data_json"],
            "history_relative_dir": rel,
        }

        goals = [
            dict(r)
            for r in con.execute(
                "SELECT * FROM local_runtime_thread_goals WHERE session_id = ?", (session_id,)
            )
        ]
        msg_rows = [
            dict(r)
            for r in con.execute(
                "SELECT * FROM local_runtime_message_rows WHERE session_id = ?", (session_id,)
            )
        ]
        row_migrations = [
            dict(r)
            for r in con.execute("SELECT * FROM local_runtime_message_row_migrations")
        ]
        # ★★★ 关键行：canonical history 的迁移 checkpoint。
        # 运行时 aY() 的判定顺序：
        #     if (!legacyHistory) → 用 messages.jsonl，正常
        #     if (checkpoints.getCheckpoint(sessionId)) → 用 messages.jsonl，正常
        #     否则 → 走 legacy 迁移路径（从 ledger/pi_history_rows/blob 重建物化结果），
        #           再与 messages.jsonl 的指纹比对；legacy 源为空 ⇒
        #           "Canonical history materialization is ambiguous" 报错。
        # ★ 漏掉这一行 ⇒ 目标机永远加载不了这个会话（2026-11-05 实测踩过）。
        file_migrations = [
            dict(r)
            for r in con.execute(
                "SELECT * FROM local_runtime_pi_history_file_migrations WHERE session_id = ?",
                (session_id,),
            )
        ]
        projects = [
            dict(r)
            for r in con.execute(
                "SELECT * FROM local_runtime_projects WHERE workspace_dir = ?",
                (row["workspace_dir"],),
            )
        ]
    finally:
        con.close()

    pack = node_pack(session_dir)
    pack["handoff"] = {
        "targetRow": target_row,
        "goals": goals,
        "messageRows": msg_rows,
        "messageRowMigrations": row_migrations,
        "piHistoryFileMigrations": file_migrations,
        "projects": projects,
        "sourceWorkspaceDir": row["workspace_dir"],
        "sourceRelativeDir": rel,
        "fingerprint": fp,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(pack, ensure_ascii=False), encoding="utf-8")
    return {
        "session_id": session_id,
        "relative_dir": rel,
        "files": len(pack["files"]),
        "message_count": fp["messageCount"],
        "active_revision": fp["activeRevision"],
        "goal_rows": len(goals),
        "message_rows": len(msg_rows),
        "out": str(out_path),
    }


def node_pack(session_dir: Path) -> dict:
    """调用 Node 工具打包（保证文件清单与指纹同源）。"""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        tmp = Path(tf.name)
    try:
        subprocess.run(
            ["node", str(TOOL), "pack", str(session_dir), str(tmp)],
            capture_output=True,
            text=True,
            check=True,
        )
        return json.loads(tmp.read_text(encoding="utf-8"))
    finally:
        tmp.unlink(missing_ok=True)


# --------------------------------------------------------------------------
# import —— 把会话落到目标机
# --------------------------------------------------------------------------
def import_pack(
    pack_path: Path,
    target_data_dir: Path,
    target_workspace: Path,
    db_override: Path | None = None,
    dry_run: bool = False,
) -> dict:
    pack = json.loads(pack_path.read_text(encoding="utf-8"))
    ho = pack["handoff"]
    row = dict(ho["targetRow"])
    session_id = row["session_id"]

    target_data_dir = Path(target_data_dir).expanduser()
    target_workspace = Path(target_workspace).expanduser()
    db = Path(db_override) if db_override else db_path(target_data_dir)

    # 1) 解包会话目录 → 目标机 dataDir/v2/sessions/<rel>
    rel = row["history_relative_dir"]
    dest_dir = sessions_root(target_data_dir) / rel
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    for p in pack["payload"]:
        full = dest_dir / p["path"]
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_bytes(base64.b64decode(p["content"]))

    # 2) manifest 的绝对路径改写成目标机
    #    ★ 运行时恢复时会用 manifest.paths.messages 定位历史文件
    #      （DB 列 history_relative_dir 是兜底），所以这里必须改成目标机路径。
    manifest_path = dest_dir / "manifest.json"
    if manifest_path.exists():
        m = json.loads(manifest_path.read_text(encoding="utf-8"))
        for k, v in list(m.get("paths", {}).items()):
            if not isinstance(v, str) or not v.startswith("/"):
                continue
            # 把「源机 sessions 根」前缀替换成「目标机 sessions 根」
            if "/v2/sessions/" in v:
                _, tail = v.split("/v2/sessions/", 1)
                m["paths"][k] = str(sessions_root(target_data_dir)) + "/" + tail
        manifest_path.write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")

    # 3) 校验落地后的指纹
    fp = node_fingerprint(dest_dir)
    revision_ok = fp["activeRevision"] == pack["activeRevision"]

    if dry_run:
        return {
            "dry_run": True,
            "dest_dir": str(dest_dir),
            "revision_ok": revision_ok,
            "computed": fp["activeRevision"],
            "expected": pack["activeRevision"],
        }

    con = connect_db(db)
    try:
        con.execute("PRAGMA foreign_keys=OFF")
        # 3a) workspace / project 绑定：目标机上必须指向真实 workspace
        proj = con.execute(
            "SELECT * FROM local_runtime_projects WHERE workspace_dir = ?",
            (str(target_workspace),),
        ).fetchone()
        if proj is None:
            # 建一个 project（project_kind=workspace）
            cols = [r[1] for r in con.execute("PRAGMA table_info(local_runtime_projects)")]
            vals = {
                "project_kind": "workspace",
                "workspace_dir": str(target_workspace),
                "pinned": 0,
                "hidden": 0,
                "order_index": 2147483647,
                "recent_at_ms": None,
                "latest_activity_at_ms": int(pack["packedAtMs"]),
                "session_count": 1,
                "extra_data_json": "{}",
                "created_at_ms": int(pack["packedAtMs"]),
                "updated_at_ms": int(pack["packedAtMs"]),
            }
            use = {k: v for k, v in vals.items() if k in cols}
            cur = con.execute(
                f"INSERT INTO local_runtime_projects ({','.join(use)}) "
                f"VALUES ({','.join('?' * len(use))})",
                list(use.values()),
            )
            project_id = cur.lastrowid
        else:
            project_id = proj["project_id"]

        row["workspace_dir"] = str(target_workspace)
        row["project_workspace_dir"] = str(target_workspace)
        row["project_id"] = project_id
        # ★ record_json 内也存了一份 workspaceDir —— 运行时会从它解析历史路径，
        #   只改独立列会导致「历史目录指向源机路径」⇒ 报
        #   "Canonical history materialization is ambiguous"。必须同步改写。
        if row.get("record_json"):
            try:
                rj = json.loads(row["record_json"])
                rj["workspaceDir"] = str(target_workspace)
                row["record_json"] = json.dumps(rj, ensure_ascii=False)
            except (TypeError, ValueError):
                pass
        # 接手态：不是运行中，且要可见
        row["status"] = "idle"
        row["archived"] = 0
        row["visibility"] = "visible"
        row["updated_at_ms"] = int(pack["packedAtMs"])

        cols = [r[1] for r in con.execute("PRAGMA table_info(local_runtime_sessions)")]
        use = {k: v for k, v in row.items() if k in cols}
        cur = con.execute(
            f"INSERT OR REPLACE INTO local_runtime_sessions ({','.join(use)}) "
            f"VALUES ({','.join('?' * len(use))})",
            list(use.values()),
        )

        # 3b) message_rows（UI 回看投影）
        con.execute("DELETE FROM local_runtime_message_rows WHERE session_id = ?", (session_id,))
        mr_cols = [r[1] for r in con.execute("PRAGMA table_info(local_runtime_message_rows)")]
        for m in ho["messageRows"]:
            m = dict(m)
            m.pop("id", None)  # id 自增，交给目标机分配
            use = {k: v for k, v in m.items() if k in mr_cols}
            con.execute(
                f"INSERT INTO local_runtime_message_rows ({','.join(use)}) "
                f"VALUES ({','.join('?' * len(use))})",
                list(use.values()),
            )

        # 3c) goals（与 session UNIQUE 绑定，必须单独搬）
        con.execute("DELETE FROM local_runtime_thread_goals WHERE session_id = ?", (session_id,))
        g_cols = [r[1] for r in con.execute("PRAGMA table_info(local_runtime_thread_goals)")]
        for g in ho["goals"]:
            g = dict(g)
            use = {k: v for k, v in g.items() if k in g_cols}
            con.execute(
                f"INSERT INTO local_runtime_thread_goals ({','.join(use)}) "
                f"VALUES ({','.join('?' * len(use))})",
                list(use.values()),
            )

        # 3d) 清执行权租约（否则目标机可能拒绝执行）
        con.execute("DELETE FROM local_runtime_session_locks WHERE session_id = ?", (session_id,))

        # 3e) display 行回填标记（有则搬，保证 UI 历史回看正常）
        try:
            mig_cols = [r[1] for r in con.execute("PRAGMA table_info(local_runtime_message_row_migrations)")]
            for m in ho.get("messageRowMigrations", []):
                m = dict(m)
                use = {k: v for k, v in m.items() if k in mig_cols}
                if not use:
                    continue
                exists = con.execute(
                    f"SELECT 1 FROM local_runtime_message_row_migrations WHERE "
                    f"{' AND '.join(f'{c}=?' for c in use)}",
                    list(use.values()),
                ).fetchone()
                if not exists:
                    con.execute(
                        f"INSERT INTO local_runtime_message_row_migrations ({','.join(use)}) "
                        f"VALUES ({','.join('?' * len(use))})",
                        list(use.values()),
                    )
        except sqlite3.Error:
            pass

        # 3f) ★ canonical history 迁移 checkpoint（**缺这行目标机就加载不了**）
        #     目标机若无此行，运行时会走 legacy 迁移路径重建历史并与
        #     messages.jsonl 指纹比对，legacy 源为空 ⇒ ambiguous 报错。
        fmr = ho.get("piHistoryFileMigrations") or []
        if fmr:
            con.execute(
                "DELETE FROM local_runtime_pi_history_file_migrations WHERE session_id = ?",
                (session_id,),
            )
            fm_cols = [
                r[1]
                for r in con.execute("PRAGMA table_info(local_runtime_pi_history_file_migrations)")
            ]
            for fm in fmr:
                fm = dict(fm)
                use = {k: v for k, v in fm.items() if k in fm_cols}
                con.execute(
                    f"INSERT INTO local_runtime_pi_history_file_migrations ({','.join(use)}) "
                    f"VALUES ({','.join('?' * len(use))})",
                    list(use.values()),
                )
        else:
            # 源机没有这一行（会话创建于非常早期）→ 目标机也需要
            # 造一条「source=empty / message_count=0」的等价 checkpoint，
            # 否则目标机同样会误入 legacy 迁移路径。
            con.execute(
                "INSERT OR REPLACE INTO local_runtime_pi_history_file_migrations "
                "(session_id, migrated_at_ms, source, message_count, target_revision) "
                "VALUES (?, ?, 'empty', 0, ?)",
                (
                    session_id,
                    int(pack["packedAtMs"]),
                    "sha256:4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
                ),
            )

        con.commit()
    finally:
        con.close()

    return {
        "session_id": session_id,
        "dest_dir": str(dest_dir),
        "project_id": project_id,
        "workspace": str(target_workspace),
        "revision_ok": revision_ok,
        "computed_revision": fp["activeRevision"],
        "expected_revision": pack["activeRevision"],
        "message_count": fp["messageCount"],
        "files": len(pack["files"]),
        "db": str(db),
    }


# --------------------------------------------------------------------------
# verify —— 无网只读验收
# --------------------------------------------------------------------------
def verify(session_id: str, data_dir: Path, db_override: Path | None = None) -> dict:
    data_dir = Path(data_dir).expanduser()
    db = Path(db_override) if db_override else db_path(data_dir)
    con = connect_db(db)
    try:
        row = con.execute(
            "SELECT * FROM local_runtime_sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        if row is None:
            return {"ok": False, "reason": "session-row-missing", "session_id": session_id}
        rel = row["history_relative_dir"]
        d = sessions_root(data_dir) / rel
        if not (d / "messages.jsonl").exists():
            return {"ok": False, "reason": "messages-jsonl-missing", "dir": str(d)}

        fp = node_fingerprint(d)
        goals = con.execute(
            "SELECT COUNT(*) c FROM local_runtime_thread_goals WHERE session_id = ?", (session_id,)
        ).fetchone()["c"]
        mrows = con.execute(
            "SELECT COUNT(*) c FROM local_runtime_message_rows WHERE session_id = ?", (session_id,)
        ).fetchone()["c"]
        locks = con.execute(
            "SELECT COUNT(*) c FROM local_runtime_session_locks WHERE session_id = ?", (session_id,)
        ).fetchone()["c"]
        # ★ checkpoint：缺这行 ⇒ 目标机运行时���误入 legacy 迁移路径 ⇒ 加载失败
        ck = con.execute(
            "SELECT * FROM local_runtime_pi_history_file_migrations WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        proj = con.execute(
            "SELECT * FROM local_runtime_projects WHERE project_id = ?", (row["project_id"],)
        ).fetchone()

        # 目录名是否与 suffix 规则一致
        expected_suffix = session_suffix(session_id)
        dir_ok = rel.endswith(f"-session_{expected_suffix}")

        return {
            "ok": bool(
                fp["catalogMatches"] is not False
                and dir_ok
                and locks == 0
                and ck is not None
            ),
            "session_id": session_id,
            "relative_dir": rel,
            "dir_name_matches_suffix_rule": dir_ok,
            "message_count": fp["messageCount"],
            "active_revision": fp["activeRevision"],
            "catalog_matches": fp["catalogMatches"],
            "catalog_revision": fp.get("catalogRevision"),
            "workspace_dir": row["workspace_dir"],
            "project_workspace_dir": row["project_workspace_dir"],
            "project_exists": proj is not None,
            "project_workspace_dir_ok": (proj is not None and proj["workspace_dir"] == row["workspace_dir"]),
            "status": row["status"],
            "visibility": row["visibility"],
            "archived": row["archived"],
            "session_kind": row["session_kind"],
            "goal_rows": goals,
            "message_rows": mrows,
            "lock_rows": locks,
            "history_checkpoint_present": ck is not None,
            "history_checkpoint": (dict(ck) if ck is not None else None),
        }
    finally:
        con.close()


# --------------------------------------------------------------------------
# list —— 列出目标机上某 workspace 的会话（对应 TUI 的会话列表）
# --------------------------------------------------------------------------
def list_sessions(data_dir: Path, workspace: str | None = None, db_override: Path | None = None) -> list:
    data_dir = Path(data_dir).expanduser()
    db = Path(db_override) if db_override else db_path(data_dir)
    con = connect_db(db)
    try:
        sql = (
            "SELECT session_id, session_kind, session_type, status, visibility, archived, "
            "title, workspace_dir, history_relative_dir, created_at_ms, updated_at_ms "
            "FROM local_runtime_sessions WHERE columnar_version = 3"
        )
        args: list = []
        if workspace:
            sql += " AND workspace_dir = ?"
            args.append(workspace)
        sql += " ORDER BY updated_at_ms DESC"
        return [dict(r) for r in con.execute(sql, args)]
    finally:
        con.close()


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="mcode 会话转移")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("export", help="从本机抽出一个会话")
    p.add_argument("--session", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--data-dir", default=None)

    p = sub.add_parser("import", help="把会话落到目标机")
    p.add_argument("--pack", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--workspace", required=True)
    p.add_argument("--db", default=None)
    p.add_argument("--dry-run", action="store_true")

    p = sub.add_parser("verify", help="无网只读验收")
    p.add_argument("--session", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--db", default=None)

    p = sub.add_parser("list", help="列出目标机会话")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--workspace", default=None)
    p.add_argument("--db", default=None)

    args = ap.parse_args()
    try:
        if args.cmd == "export":
            print(json.dumps(export(args.session, Path(args.out),
                                   Path(args.data_dir) if args.data_dir else None),
                             ensure_ascii=False, indent=2))
        elif args.cmd == "import":
            r = import_pack(Path(args.pack), Path(args.data_dir), Path(args.workspace),
                            Path(args.db) if args.db else None, args.dry_run)
            print(json.dumps(r, ensure_ascii=False, indent=2))
        elif args.cmd == "verify":
            r = verify(args.session, Path(args.data_dir), Path(args.db) if args.db else None)
            print(json.dumps(r, ensure_ascii=False, indent=2))
            sys.exit(0 if r.get("ok") else 1)
        elif args.cmd == "list":
            dd = Path(args.data_dir) if args.data_dir else mcode_data_dir()
            rows = list_sessions(dd, args.workspace, Path(args.db) if args.db else None)
            print(json.dumps(rows, ensure_ascii=False, indent=2))
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001
        print(f"❌ {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
