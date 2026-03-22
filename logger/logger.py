"""
Botroyale Log Collector & Dashboard Backend
- Receives logs from bots via POST /log
- Stores in SQLite
- Serves dashboard at /
- Streams new logs via SSE at /stream
"""
import os
import json
import sqlite3
import threading
import queue
import time
import re as _re
from datetime import datetime
from flask import Flask, request, jsonify, Response, send_from_directory

app = Flask(__name__, static_folder='static')

DB_PATH      = os.getenv("DB_PATH", "/data/logs.db")
MAX_LOGS     = int(os.getenv("MAX_LOGS", 50000))
PORT         = int(os.getenv("PORT", 5000))
API_BASE_URL = os.getenv("API_BASE_URL", "https://cdn.moltyroyale.com/api")

# Load all configured bot API keys from BOT{N}_API env vars (N = 1..100)
def _load_bot_configs():
    return [
        os.getenv(f"BOT{i}_API").strip()
        for i in range(1, 101)
        if os.getenv(f"BOT{i}_API", "").strip()
    ]

BOT_CONFIGS = _load_bot_configs()  # list of api_key strings

# Prune counter — only prune every 100 inserts
_insert_counter = [0]

# SSE subscriber queues — one per connected browser tab
subscribers      = []
subscribers_lock = threading.Lock()

# ─── DATABASE ─────────────────────────────────────
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS logs (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                ts        TEXT    NOT NULL,
                bot       TEXT    NOT NULL,
                game_id   TEXT,
                turn      INTEGER,
                msg       TEXT    NOT NULL,
                log_type  TEXT    NOT NULL DEFAULT 'move',
                channel   TEXT    NOT NULL DEFAULT 'game',
                created   REAL    NOT NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_bot     ON logs(bot)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_game_id ON logs(game_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_created ON logs(created)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_channel ON logs(channel)")

        # ── Persistent game outcome table — never pruned ──────────────────
        conn.execute("""
            CREATE TABLE IF NOT EXISTS game_results (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                bot            TEXT    NOT NULL,
                game_id        TEXT,
                outcome        TEXT    NOT NULL,   -- 'win'|'loss'|'unknown'
                kills          INTEGER NOT NULL DEFAULT 0,
                survival_turn  INTEGER NOT NULL DEFAULT 0,
                placement      INTEGER,            -- NULL if unknown
                joined_ts      TEXT,
                ended_ts       TEXT,
                join_date      TEXT    NOT NULL,   -- YYYY-MM-DD, indexed
                join_created   REAL    NOT NULL    -- unix timestamp for range queries
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_gr_bot      ON game_results(bot)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_gr_date     ON game_results(join_date)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_gr_created  ON game_results(join_created)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_gr_game_id  ON game_results(game_id)")
        conn.commit()

# ── Session tracker: bot -> open session dict ─────────────────────────────────
# Matches actual bot log messages from moltybot.py
_RE_JOIN    = _re.compile(
    r'(?:Joined game|Active game found.*game=)\s*([a-f0-9-]{36})'   # new join / accounts/me recovery
    r'|Recovered agent.*from game',                                   # register-fallback recovery
    _re.I
)
_RE_DEAD    = _re.compile(r'Agent dead', _re.I)
_RE_WIN     = _re.compile(r'🏆 WON!|placement=#1(?!\d)', _re.I)
_RE_FINISH  = _re.compile(r'Game finished', _re.I)
_RE_STANDBY = _re.compile(r'FULL RESET', _re.I)
_RE_KILL    = _re.compile(r'☠️\s+KILL!|💀.*KILL')

_open_sessions = {}   # bot_name -> { game_id, joined_ts, join_created, join_date, kills }

def _flush_session(conn, bot, sess, outcome, ended_ts, survival_turn, placement=None):
    """Write a completed game session to game_results."""
    conn.execute("""
        INSERT INTO game_results
            (bot, game_id, outcome, kills, survival_turn, placement,
             joined_ts, ended_ts, join_date, join_created)
        VALUES (?,?,?,?,?,?,?,?,?,?)
    """, (
        bot, sess["game_id"], outcome, sess["kills"], survival_turn,
        placement, sess["joined_ts"], ended_ts,
        sess["join_date"], sess["join_created"]
    ))

def update_session(conn, bot, msg, turn, game_id, ts, created):
    """
    Called for every incoming log line.  Maintains _open_sessions and writes
    to game_results whenever a session ends.  All DB writes share the caller's
    connection/transaction.
    """
    global _open_sessions
    sess = _open_sessions.get(bot)

    # ── JOIN ──
    jm = _RE_JOIN.search(msg)
    if jm:
        if sess:          # close dangling session as unknown
            _flush_session(conn, bot, sess, "unknown", ts, turn or 0)
        # Extract game_id from regex group 1 if present, else use POST body game_id
        extracted_gid = (jm.group(1) if jm.lastindex and jm.group(1) else None) or game_id
        _open_sessions[bot] = {
            "game_id":      extracted_gid,
            "joined_ts":    ts,
            "join_created": created,
            "join_date":    datetime.fromtimestamp(created).strftime("%Y-%m-%d"),
            "kills":        0,
        }
        return

    if not sess:
        return

    # ── KILLS ──
    if _RE_KILL.search(msg):
        sess["kills"] += 1

    # ── DEAD → loss ──
    if _RE_DEAD.search(msg):
        _flush_session(conn, bot, sess, "loss", ts, turn or 0)
        _open_sessions.pop(bot, None)
        return

    # ── WIN ──
    if _RE_WIN.search(msg):
        _flush_session(conn, bot, sess, "win", ts, turn or 0, placement=1)
        _open_sessions.pop(bot, None)
        return

    # ── GAME FINISHED (bot survived to end — outcome unknown without placement) ──
    if _RE_FINISH.search(msg):
        _flush_session(conn, bot, sess, "unknown", ts, turn or 0)
        _open_sessions.pop(bot, None)
        return

    # ── RESET (closes any dangling session) ──
    if _RE_STANDBY.search(msg):
        _flush_session(conn, bot, sess, "unknown", ts, turn or 0)
        _open_sessions.pop(bot, None)
        return

def classify(msg):
    import re
    if re.search(r'💀|KILL|☠️|eliminated', msg):                          return 'kill'
    if re.search(r'💊|[Hh]eal|potion|HP kritis|medkit|bandage|medical', msg): return 'heal'
    if re.search(r'⚠️|WATCHDOG|kabur|TERPAKSA|bahaya|Diserang', msg):     return 'warn'
    if re.search(r'💰|🎒|[Pp]ickup|[Ll]oot|Moltz', msg):                  return 'loot'
    return 'move'

def prune_old_logs():
    """Keep only MAX_LOGS most recent entries."""
    with get_db() as conn:
        count = conn.execute("SELECT COUNT(*) FROM logs").fetchone()[0]
        if count > MAX_LOGS:
            delete_count = count - MAX_LOGS
            conn.execute("""
                DELETE FROM logs WHERE id IN (
                    SELECT id FROM logs ORDER BY created ASC LIMIT ?
                )
            """, (delete_count,))
            conn.commit()

# ─── SSE ──────────────────────────────────────────
def broadcast(data: dict):
    """Send a log entry to all connected SSE clients."""
    payload = f"data: {json.dumps(data)}\n\n"
    with subscribers_lock:
        dead = []
        for q in subscribers:
            try:
                q.put_nowait(payload)
            except Exception:
                dead.append(q)
        for q in dead:
            subscribers.remove(q)

# ─── ROUTES ───────────────────────────────────────
@app.route("/")
def dashboard():
    resp = send_from_directory("static", "dashboard.html")
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
    resp.headers["Pragma"]        = "no-cache"
    resp.headers["Expires"]       = "0"
    return resp

@app.route("/log", methods=["POST"])
def receive_log():
    data    = request.get_json(silent=True) or {}
    bot     = data.get("bot", "unknown")
    msg     = data.get("msg", "")
    turn    = data.get("turn")
    game_id = data.get("game_id")
    ts      = data.get("ts") or datetime.utcnow().strftime("%H:%M:%S")
    created = time.time()
    ltype   = classify(msg)
    channel = data.get("channel", "game")

    row = {
        "bot": bot, "msg": msg, "turn": turn,
        "game_id": game_id, "ts": ts,
        "log_type": ltype, "channel": channel, "created": created
    }

    with get_db() as conn:
        cur = conn.execute("""
            INSERT INTO logs (ts, bot, game_id, turn, msg, log_type, channel, created)
            VALUES (:ts, :bot, :game_id, :turn, :msg, :log_type, :channel, :created)
        """, row)
        row["id"] = cur.lastrowid
        update_session(conn, bot, msg, turn, game_id, ts, created)
        conn.commit()

    _insert_counter[0] += 1
    if _insert_counter[0] % 100 == 0:
        prune_old_logs()
    broadcast(row)
    return jsonify({"ok": True}), 201

@app.route("/logs")
def get_logs():
    bot     = request.args.get("bot")
    game_id = request.args.get("game_id")
    ltype   = request.args.get("type")
    limit   = int(request.args.get("limit", 500))
    offset  = int(request.args.get("offset", 0))
    channel = request.args.get("channel")

    query  = "SELECT * FROM logs WHERE 1=1"
    params = []
    if bot:     query += " AND bot=?";     params.append(bot)
    if game_id: query += " AND game_id=?"; params.append(game_id)
    if ltype:   query += " AND log_type=?";params.append(ltype)
    if channel: query += " AND channel=?"; params.append(channel)
    query += " ORDER BY created DESC LIMIT ? OFFSET ?"
    params += [limit, offset]

    with get_db() as conn:
        rows = conn.execute(query, params).fetchall()
    return jsonify([dict(r) for r in reversed(rows)])

@app.route("/bots")
def get_bots():
    """Return distinct bots and their latest game_id + turn."""
    with get_db() as conn:
        rows = conn.execute("""
            SELECT bot,
                   MAX(game_id) as game_id,
                   MAX(turn)    as last_turn,
                   COUNT(*)     as log_count
            FROM logs
            GROUP BY bot
            ORDER BY bot
        """).fetchall()
    return jsonify([dict(r) for r in rows])

@app.route("/games")
def get_games():
    """Return distinct game_ids with bot list."""
    with get_db() as conn:
        rows = conn.execute("""
            SELECT game_id, GROUP_CONCAT(DISTINCT bot) as bots, COUNT(*) as log_count
            FROM logs
            WHERE game_id IS NOT NULL
            GROUP BY game_id
            ORDER BY MAX(created) DESC
        """).fetchall()
    result = []
    for r in rows:
        d = dict(r)
        d["bots"] = d["bots"].split(",") if d["bots"] else []
        result.append(d)
    return jsonify(result)

@app.route("/stream")
def stream():
    """SSE endpoint — push new logs to dashboard in real-time."""
    def event_gen():
        q = queue.Queue(maxsize=200)
        with subscribers_lock:
            subscribers.append(q)
        try:
            while True:
                try:
                    msg = q.get(timeout=15)
                    yield msg
                except queue.Empty:
                    yield ": keepalive\n\n"
        except GeneratorExit:
            with subscribers_lock:
                if q in subscribers:
                    subscribers.remove(q)

    return Response(event_gen(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

@app.route("/status")
def get_status():
    """
    Returns live bot status derived from recent log activity in DB.
    A bot is considered 'alive' if it logged within the last 5 minutes
    and has not logged FULL RESET / Agent dead as its most recent message.
    """
    result  = {}
    cutoff  = time.time() - 300  # 5-minute activity window

    with get_db() as conn:
        # Latest log per bot within the last 5 minutes
        rows = conn.execute("""
            SELECT bot,
                   game_id,
                   msg,
                   MAX(created) as last_seen
            FROM logs
            WHERE created > ?
            GROUP BY bot
        """, (cutoff,)).fetchall()

    for row in rows:
        bot_name = row["bot"]
        last_msg = row["msg"] or ""
        # Treat as inactive if last message indicates a reset or dead state
        is_inactive = bool(
            _RE_STANDBY.search(last_msg) or
            _RE_DEAD.search(last_msg)
        )
        result[bot_name] = {
            "status":    "inactive" if is_inactive else "alive",
            "game_id":   row["game_id"],
            "last_seen": row["last_seen"],
        }

    return jsonify(result)

@app.route("/history")
def get_history():
    """
    Returns per-day, per-bot game outcomes from the dedicated game_results table.
    Optional ?days=N filters to the last N days.
    Also merges any still-open in-memory sessions as 'ongoing'.

    Returns: {
      "YYYY-MM-DD": {
        "botName": {
          "games": int, "wins": int, "losses": int, "unknown": int,
          "total_kills": int, "avg_survival_turn": float,
          "game_list": [{ game_id, joined_ts, ended_ts, outcome,
                          kills, survival_turn, placement }]
        }
      }
    }
    """
    from collections import defaultdict

    days      = request.args.get("days", type=int, default=0)
    cutoff_ts = (time.time() - days * 86400) if days > 0 else 0

    with get_db() as conn:
        if cutoff_ts:
            rows = conn.execute(
                "SELECT * FROM game_results WHERE join_created >= ? ORDER BY join_created ASC",
                (cutoff_ts,)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM game_results ORDER BY join_created ASC"
            ).fetchall()

    by_date = defaultdict(lambda: defaultdict(list))
    for row in rows:
        r = dict(row)
        by_date[r["join_date"]][r["bot"]].append({
            "game_id":       r["game_id"],
            "joined_ts":     r["joined_ts"],
            "ended_ts":      r["ended_ts"],
            "outcome":       r["outcome"],
            "kills":         r["kills"],
            "survival_turn": r["survival_turn"],
            "placement":     r["placement"],
        })

    # Merge in-memory open sessions as "ongoing"
    for bot, sess in _open_sessions.items():
        if cutoff_ts and sess["join_created"] < cutoff_ts:
            continue
        by_date[sess["join_date"]][bot].append({
            "game_id":       sess["game_id"],
            "joined_ts":     sess["joined_ts"],
            "ended_ts":      None,
            "outcome":       "ongoing",
            "kills":         sess["kills"],
            "survival_turn": 0,
            "placement":     None,
        })

    result = {}
    for date_str in sorted(by_date.keys(), reverse=True):
        result[date_str] = {}
        for bot, games in by_date[date_str].items():
            wins   = sum(1 for g in games if g["outcome"] == "win")
            losses = sum(1 for g in games if g["outcome"] == "loss")
            kills  = sum(g["kills"] for g in games)
            turns  = [g["survival_turn"] for g in games if g["survival_turn"]]
            result[date_str][bot] = {
                "games":             len(games),
                "wins":              wins,
                "losses":            losses,
                "unknown":           len(games) - wins - losses,
                "total_kills":       kills,
                "avg_survival_turn": round(sum(turns) / len(turns), 1) if turns else 0,
                "game_list":         games,
            }
    return jsonify(result)


@app.route("/history/rebuild", methods=["POST"])
def rebuild_history():
    """
    One-time migration: replay the entire logs table into game_results.
    Safe to call multiple times — clears game_results first, then replays.
    """
    global _open_sessions
    _open_sessions = {}

    with get_db() as conn:
        conn.execute("DELETE FROM game_results")
        rows = conn.execute(
            "SELECT bot, msg, turn, game_id, ts, created FROM logs ORDER BY created ASC"
        ).fetchall()
        for row in rows:
            update_session(conn, row["bot"], row["msg"], row["turn"],
                           row["game_id"], row["ts"], row["created"])
        # Flush dangling open sessions as unknown
        for bot, sess in _open_sessions.items():
            _flush_session(conn, bot, sess, "unknown", None, 0)
        _open_sessions = {}
        conn.commit()

    return jsonify({"ok": True, "replayed": len(rows)})

@app.route("/accounts")
def get_accounts():
    """
    Fetch /accounts/me for every configured bot in parallel.
    Returns list of { name, api_key_hint, wallet, data, error } in config order.
    """
    import requests as ext_req
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if not BOT_CONFIGS:
        return jsonify({"success": False,
                        "error": "No bots configured. Set BOT1_API env vars."}), 500

    def fetch_one(api_key):
        try:
            res  = ext_req.get(
                f"{API_BASE_URL}/accounts/me",
                headers={"X-API-Key": api_key},
                timeout=8
            )
            body = res.json()
            return {
                "data":  body.get("data") if body.get("success") else None,
                "error": None if body.get("success") else body,
            }
        except Exception as e:
            return {"data": None, "error": str(e)}

    results = [None] * len(BOT_CONFIGS)
    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = {pool.submit(fetch_one, key): i for i, key in enumerate(BOT_CONFIGS)}
        for fut in as_completed(futures):
            results[futures[fut]] = fut.result()

    total_balance = sum((r["data"] or {}).get("balance", 0) for r in results if r)
    return jsonify({"success": True, "bots": results, "total_balance": total_balance})

@app.route("/clear", methods=["POST"])
def clear_logs():
    with get_db() as conn:
        conn.execute("DELETE FROM logs")
        conn.commit()
    return jsonify({"ok": True})

if __name__ == "__main__":
    init_db()
    app.run(host="0.0.0.0", port=PORT, threaded=True)
