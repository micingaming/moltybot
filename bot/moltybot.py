import requests
import time
import random
import os
import sys
import json
import threading
from dotenv import load_dotenv
load_dotenv()
from datetime import datetime, timedelta, timezone
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# =========================
# REALTIME LOG
# =========================
sys.stdout.reconfigure(line_buffering=True)

# =========================
# GLOBAL CONFIG
# =========================
BASE_URL      = os.getenv("BASE_URL", "")
AGENT_NAME    = os.getenv("AGENT_NAME")
API_KEY_ENV   = os.getenv("API_KEY", "").strip()
BOT_INDEX     = int(os.getenv("BOT_INDEX", ""))
LOGGER_URL    = os.getenv("LOGGER_URL", "").strip()

if not AGENT_NAME:
    raise EnvironmentError("AGENT_NAME must be set in environment")

# =========================
# OLLAMA CONFIG
# Only reply whispers from this sender name
# =========================
OLLAMA_URL   = os.getenv("OLLAMA_URL")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")
GUARDIAN_NAME = "Guardian"   # only whispers from this sender will be processed

# =========================
# GLOBAL SESSION
# =========================
session = requests.Session()

retry_strategy = Retry(
    total=8,
    backoff_factor=2,
    status_forcelist=[500, 502, 503, 504],
    allowed_methods=["GET", "POST"],
    raise_on_status=False
)

adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("https://", adapter)
session.mount("http://", adapter)

# =========================
# OLLAMA LLM HELPER
# Sends a question to local Ollama and returns the answer string.
# Falls back to a default message if Ollama is unreachable or errors out.
# =========================
def ask_ollama(question, timeout=120):
    """
    Send `question` to Ollama and return the generated answer.
    Returns a fallback string if the call fails for any reason.
    """
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model":  OLLAMA_MODEL,
                "system": "You are a concise assistant. Never restate question. Never show working, equations, or reasoning steps. Give only the final answer.",
                "prompt": question,
                "stream": False,        # get full response in one shot
                "options": {"num_ctx": 512},
            },
            timeout=timeout
        )
        resp.raise_for_status()
        answer = resp.json().get("response", "").strip()
        return answer if answer else "Yes"
    except requests.exceptions.ConnectionError:
        return "Yes"
    except requests.exceptions.Timeout:
        return "Yes"
    except Exception as e:
        return "Yes"

# =========================
# LOG
# =========================
def log(bot_id, msg, turn=0, game_id=None):
    tz = timezone(timedelta(hours=7))
    now = datetime.now(tz)
    turn_label = f" T{turn}" if turn else ""
    game_label = f":{game_id[24:]}" if game_id else ""
    print(f"[{now.strftime('%H:%M:%S')}]{turn_label} [{AGENT_NAME}{game_label}] {msg}", flush=True)

    if LOGGER_URL:
        def _send():
            try:
                requests.post(
                    LOGGER_URL,
                    json={
                        "bot":     AGENT_NAME,
                        "msg":     msg,
                        "turn":    turn,
                        "game_id": game_id,
                        "ts":      now.strftime("%H:%M:%S"),
                    },
                    timeout=3
                )
            except Exception:
                pass
        threading.Thread(target=_send, daemon=True).start()


# =========================
# SAFE REQUEST
# =========================
def safe_request(method, url, max_attempts=4, cooldown=20, **kwargs):
    for attempt in range(1, max_attempts + 1):
        try:
            res = getattr(session, method)(url, **kwargs)
            if res.status_code == 504:
                wait = cooldown * attempt
                log("sys", f"504 Timeout [{attempt}/{max_attempts}] -> waiting {wait}s | {url}")
                time.sleep(wait)
                continue
            return res
        except requests.exceptions.Timeout:
            wait = cooldown * attempt
            log("sys", f"Timeout [{attempt}/{max_attempts}] -> waiting {wait}s | {url}")
            time.sleep(wait)
        except requests.exceptions.ConnectionError as e:
            wait = cooldown * attempt
            log("sys", f"ConnectionError [{attempt}/{max_attempts}] -> waiting {wait}s | {e}")
            time.sleep(wait)

    log("sys", f"All {max_attempts} attempts failed -> returning None | {url}")
    return None


class NyrAgent:

    def __init__(self):
        self.index = BOT_INDEX
        self.name  = AGENT_NAME

        # Per-bot file isolation — all bots share /app volume so we add BOT_INDEX suffix
        self.key_file      = f"key_api_{BOT_INDEX}.txt"
        self.mem_file      = f"visited_{BOT_INDEX}.json"
        self.agent_file    = f"agent_id_{BOT_INDEX}.txt"
        self.game_file     = f"game_id_{BOT_INDEX}.txt"
        self.dzreg_file    = f"dz_regions_{BOT_INDEX}.json"
        self.whitelist_file = "bot_friendly.json"   # shared config, no index needed

        self.current_turn = 0
        self.game_id  = None
        self.agent_id = None

        self.picked_starter_weapon = False

        self.visited       = self.load_memory()
        self.api_key       = self.load_or_create_account()

        self.headers = {
            "X-API-Key": self.api_key,
            "Content-Type": "application/json"
        }

        self.game_id       = self.load_game_id()
        self.friendly_bots = self.load_friendly_bots()
        self.dz_regions    = self.load_dz_regions()
        self.agent_id      = self.load_agent_id()

        self.last_heartbeat      = time.time()
        self.move_counter        = 0
        self.last_activity_region = None
        self.heartbeat_timeout   = 600
        self.state_fail_count    = 0
        self.state_fail_limit    = 5
        self.replied_message_ids = set()  # track whisper IDs already replied to (avoid re-reply each turn)

    # =========================
    # MEMORY
    # =========================
    def load_memory(self):
        if os.path.exists(self.mem_file):
            with open(self.mem_file, "r") as f:
                return set(json.load(f))
        return set()

    def save_memory(self):
        with open(self.mem_file, "w") as f:
            json.dump(list(self.visited), f)

    # =========================
    # DZ REGIONS
    # =========================
    def load_dz_regions(self):
        if os.path.exists(self.dzreg_file):
            try:
                with open(self.dzreg_file) as f:
                    data = json.load(f)
                regions = set(data.get("dz_regions", []))
                log(self.index, f"Loaded {len(regions)} dz regions from file.", self.current_turn, self.game_id)
                return regions
            except Exception as e:
                log(self.index, f"Failed to load dz_regions.json: {e}", self.current_turn, self.game_id)
        return set()

    def save_dz_regions(self):
        try:
            output = {
                "description": "DZ regions - deathzones bot must never enter. Updated automatically each turn.",
                "dz_regions": sorted(list(self.dz_regions))
            }
            with open(self.dzreg_file, 'w') as f:
                json.dump(output, f, indent=2)
        except Exception as e:
            log(self.index, f"Failed to save dz_regions.json: {e}", self.current_turn, self.game_id)

    def load_friendly_bots(self):
        if os.path.exists(self.whitelist_file):
            try:
                with open(self.whitelist_file) as f:
                    data = json.load(f)
                bots = set(data.get("friendly_bots", []))
                log(self.index, f"Loaded {len(bots)} friendly bots: {bots}", self.current_turn, self.game_id)
                return bots
            except Exception as e:
                log(self.index, f"Failed to load bot_friendly.json: {e}", self.current_turn, self.game_id)
        return set()

    def get_pending_deathzone_ids(self, data):
        pending = set()
        for item in data.get("pendingDeathzones", []):
            if isinstance(item, dict) and item.get("id"):
                pending.add(item["id"])
            elif isinstance(item, str) and item:
                pending.add(item)
        for entry in data.get("recentLogs", []):
            if not isinstance(entry, dict):
                continue
            if entry.get("type") == "deathzone":
                for rid in entry.get("details", {}).get("regions", []):
                    if isinstance(rid, str) and rid:
                        pending.add(rid)
        return pending

    def update_dz_regions(self, data):
        new_ids = set()

        for r in data.get("visibleRegions", []):
            if isinstance(r, dict) and r.get("isDeathZone") and r.get("id"):
                new_ids.add(r["id"])

        for r in data.get("connectedRegions", []):
            if isinstance(r, dict) and r.get("isDeathZone") and r.get("id"):
                new_ids.add(r["id"])

        for entry in data.get("recentLogs", []):
            if isinstance(entry, dict) and entry.get("type") == "deathzone":
                for rid in entry.get("details", {}).get("regions", []):
                    if isinstance(rid, str) and rid:
                        new_ids.add(rid)

        pending_ids = self.get_pending_deathzone_ids(data)
        if pending_ids:
            new_pending = pending_ids - self.dz_regions
            if new_pending:
                log(self.index, f"Pending deathzones baru terdeteksi: {len(new_pending)} region(s)", self.current_turn, self.game_id)
        new_ids |= pending_ids

        added = new_ids - self.dz_regions
        if added:
            self.dz_regions |= new_ids
            self.save_dz_regions()
            log(self.index, f"DZ regions updated: +{len(added)} new (incl. pending). Total: {len(self.dz_regions)}", self.current_turn, self.game_id)

    # =========================
    # LOAD / SAVE IDS
    # =========================
    def load_game_id(self):
        if os.path.exists(self.game_file):
            gid = open(self.game_file).read().strip()
            if gid:
                log(self.index, f"Recovered GameID: {gid}", self.current_turn, self.game_id)
                return gid
        return None

    def load_agent_id(self):
        if os.path.exists(self.agent_file):
            aid = open(self.agent_file).read().strip()
            if aid:
                log(self.index, f"Recovered AgentID: {aid}", self.current_turn, self.game_id)
                return aid
        return None

    def save_agent_id(self):
        if self.agent_id:
            open(self.agent_file, "w").write(self.agent_id)

    def save_game_id(self):
        if self.game_id:
            open(self.game_file, "w").write(self.game_id)

    # =========================
    # ACCOUNT
    # =========================
    def load_or_create_account(self):
        # 1. Use API key from environment variable (set by docker-compose)
        if API_KEY_ENV:
            log(self.index, "Using API key from environment (API_KEY)", self.current_turn, self.game_id)
            return API_KEY_ENV

        # 2. Fall back to persisted key file
        if os.path.exists(self.key_file):
            log(self.index, "Using existing API key from file", self.current_turn, self.game_id)
            return open(self.key_file).read().strip()

        # 3. Create a new account
        log(self.index, "Creating new account...", self.current_turn, self.game_id)
        res = safe_request("post", f"{BASE_URL}/accounts", json={"name": self.name}, timeout=15)
        if res is None:
            raise RuntimeError("Failed to create account after retries")

        key = res.json()["data"]["apiKey"]
        open(self.key_file, "w").write(key)
        return key

    # =========================
    # RECOVER VIA /accounts/me
    # Replaces the old check_game_status + recover_agent_id which used
    # /games/{id}/state — an endpoint unavailable after game finishes.
    # =========================
    def recover_from_accounts_me(self):
        """
        Call /accounts/me to find an active game for this bot.
        Returns True and updates self.game_id / self.agent_id if found.
        """
        res = safe_request("get", f"{BASE_URL}/accounts/me", headers=self.headers, timeout=15)
        if res is None or res.status_code != 200:
            log(self.index, "accounts/me unavailable during recovery", self.current_turn, self.game_id)
            return False
        try:
            data       = res.json()["data"]
            curr_games = data.get("currentGames", [])
            if not curr_games:
                return False
            game       = curr_games[0]
            game_id    = game.get("gameId")
            agent_id   = game.get("agentId")
            if game_id and agent_id:
                self.game_id  = game_id
                self.agent_id = agent_id
                self.save_game_id()
                self.save_agent_id()
                log(self.index, f"Active game found via accounts/me: game={game_id}, agent={agent_id}", self.current_turn, self.game_id)
                return True
        except Exception as e:
            log(self.index, f"recover_from_accounts_me parse error: {e}", self.current_turn, self.game_id)
        return False

    # =========================
    # STARTUP FLOW
    # =========================
    def startup(self):
        """
        1. Query /accounts/me — if an active game exists, resume it.
        2. Otherwise full_reset -> find_and_join_game.
        """
        log(self.index, "Startup: checking for active game via accounts/me...", self.current_turn, self.game_id)
        if self.recover_from_accounts_me():
            log(self.index, f"Resuming active game {self.game_id} as agent {self.agent_id}", self.current_turn, self.game_id)
            return

        log(self.index, "No active game found -> joining new game", self.current_turn, self.game_id)
        self.full_reset()
        self.find_and_join_game()

    # =========================
    # JOIN GAME
    # =========================
    def find_and_join_game(self):
        while not self.agent_id:
            res = safe_request("get", f"{BASE_URL}/games?status=waiting", timeout=15)

            if res is None or res.status_code != 200:
                log(self.index, "Failed to fetch game list, retrying in 20s...", self.current_turn, self.game_id)
                time.sleep(20)
                continue

            try:
                games = res.json().get("data", [])
            except Exception as e:
                log(self.index, f"Failed to parse game list: {e}, retrying in 15s...", self.current_turn, self.game_id)
                time.sleep(15)
                continue

            free = [g for g in games if g.get("entryType") == "free"]

            if not free:
                log(self.index, "No free game available, retrying in 15s...", self.current_turn, self.game_id)
                time.sleep(15)
                continue

            self.game_id = free[0]["id"]
            log(self.index, f"Trying to join game {self.game_id}...", self.current_turn, self.game_id)

            reg = safe_request(
                "post", f"{BASE_URL}/games/{self.game_id}/agents/register",
                json={"name": self.name},
                headers=self.headers,
                timeout=15
            )

            if reg is None:
                log(self.index, "Register request failed entirely, retrying in 15s...", self.current_turn, self.game_id)
                self.game_id = None
                time.sleep(15)
                continue

            if reg.status_code == 201:
                self.agent_id = reg.json()["data"]["id"]
                self.save_game_id()
                self.save_agent_id()
                log(self.index, f"Joined game {self.game_id} as agent {self.agent_id}", self.current_turn, self.game_id)
                return

            # Already registered or other error — try accounts/me to recover
            log(self.index, f"Register returned {reg.status_code}, trying accounts/me recovery...", self.current_turn, self.game_id)
            if self.recover_from_accounts_me():
                log(self.index, f"Recovered agent {self.agent_id} from game {self.game_id}", self.current_turn, self.game_id)
                return

            log(self.index, "Could not recover from this game, retrying...", self.current_turn, self.game_id)
            self.game_id = None
            time.sleep(10)

    # =========================
    # WAIT FOR GAME TO FINISH
    # Called when bot dies — polls agent state until gameStatus == "finished"
    # before seeking a new game, so we don't race into the lobby immediately.
    # =========================
    def _wait_for_game_finish(self):
        log(self.index, "Agent dead — waiting for game to finish before joining next...", self.current_turn, self.game_id)
        while True:
            time.sleep(30)
            res = safe_request(
                "get",
                f"{BASE_URL}/games/{self.game_id}/agents/{self.agent_id}/state",
                headers=self.headers,
                timeout=20
            )
            if res is None or res.status_code != 200:
                log(self.index, "State unavailable while waiting — assuming game ended", self.current_turn, self.game_id)
                return
            try:
                status = res.json()["data"].get("gameStatus")
                if status == "finished":
                    log(self.index, "Game finished — ready to join next game", self.current_turn, self.game_id)
                    return
                log(self.index, f"Game still {status}, waiting...", self.current_turn, self.game_id)
            except Exception as e:
                log(self.index, f"_wait_for_game_finish parse error: {e}", self.current_turn, self.game_id)
                return

    # =========================
    # HEARTBEAT
    # =========================
    def heartbeat_watchdog(self):
        if time.time() - self.last_heartbeat > self.heartbeat_timeout:
            log(self.index, "WATCHDOG TRIGGERED -> resetting", self.current_turn, self.game_id)
            self.full_reset()
            self.find_and_join_game()

    # =========================
    # ACTION
    # =========================
    def send_action(self, action, thought="thinking"):
        payload = {
            "action": action,
            "thought": {
                "reasoning": thought,
                "plannedAction": action["type"]
            }
        }

        res = safe_request(
            "post",
            f"{BASE_URL}/games/{self.game_id}/agents/{self.agent_id}/action",
            json=payload,
            headers=self.headers,
            timeout=20
        )

        if res is None:
            log(self.index, "send_action failed after retries (server may be down)", self.current_turn, self.game_id)
        return res

    # =========================
    # AI HELPERS
    # =========================
    def has_weapon(self, inventory):
        return any(
            isinstance(i, dict) and i.get("category") == "weapon"
            for i in inventory
        )

    def build_danger_set(self, data, extra_ids=None):
        danger = set()

        for r in data.get("visibleRegions", []):
            if not isinstance(r, dict):
                continue
            if r.get("isDeathZone") and r.get("id"):
                danger.add(r["id"])

        for r in data.get("connectedRegions", []):
            if isinstance(r, dict) and r.get("isDeathZone") and r.get("id"):
                danger.add(r["id"])

        for item in data.get("pendingDeathzones", []):
            if isinstance(item, dict):
                rid = item.get("id", "")
                if rid:
                    danger.add(rid)
            elif isinstance(item, str) and item:
                danger.add(item)

        for rid in self.dz_regions:
            if isinstance(rid, str) and rid:
                danger.add(rid)

        if extra_ids:
            for x in extra_ids:
                if isinstance(x, str) and x:
                    danger.add(x)

        return danger

    def safe_move(self, data, curr_region, extra_danger_ids=None):
        try:
            connected_map = {}
            for r in data.get("connectedRegions", []):
                if isinstance(r, dict) and r.get("id"):
                    connected_map[r["id"]] = r

            all_connection_ids = []
            if isinstance(curr_region, dict):
                for cid in curr_region.get("connections", []):
                    if isinstance(cid, str) and cid:
                        all_connection_ids.append(cid)

            danger = self.build_danger_set(data, extra_ids=extra_danger_ids)

            safe = []
            for cid in all_connection_ids:
                if cid in connected_map:
                    if not connected_map[cid].get("isDeathZone", False):
                        safe.append(cid)
                else:
                    if cid not in danger:
                        safe.append(cid)

            unvisited = [c for c in safe if c not in self.visited]

            if unvisited:
                return random.choice(unvisited), "Tile aman"
            elif safe:
                return random.choice(safe), "Safe fallback"
            elif all_connection_ids:
                pending_ids = set()
                for item in data.get("pendingDeathzones", []):
                    if isinstance(item, dict):
                        rid = item.get("id", "")
                        if rid:
                            pending_ids.add(rid)
                    elif isinstance(item, str) and item:
                        pending_ids.add(item)
                not_pending = [c for c in all_connection_ids if c not in pending_ids]
                fallback = not_pending if not_pending else all_connection_ids
                if fallback:
                    return random.choice(fallback), "TERPAKSA"

            return None, "Terjebak total!"

        except Exception as e:
            log(self.index, f"safe_move error: {e}", self.current_turn, self.game_id)
            return None, f"safe_move exception: {e}"

    # =========================
    # AI DECISION
    # =========================
    def decide_action(self, data):
        if not isinstance(data, dict):
            return {"type": "rest"}, "Data tidak valid, istirahat."

        me = data.get("self", {})
        if not isinstance(me, dict):
            return {"type": "rest"}, "Self data tidak valid."

        curr_region = data.get("currentRegion", {})
        if not isinstance(curr_region, dict):
            return {"type": "rest"}, "currentRegion tidak valid."

        hp        = me.get("hp", 100)
        ep        = me.get("ep", 10)
        inventory = me.get("inventory", [])
        if not isinstance(inventory, list):
            inventory = []

        region_id_now = me.get("regionId", "")
        if region_id_now:
            self.visited.add(region_id_now)
            self.save_memory()

        self.update_dz_regions(data)

        # =============================================
        # 1. HINDARI DEATH ZONE
        # =============================================
        if curr_region.get("isDeathZone"):
            log(self.index, "⚠️  DEATHZONE — kabur sekarang!", self.current_turn, self.game_id)

            rid, reason = self.safe_move(data, curr_region)
            if rid:
                return {"type": "move", "regionId": rid}, f"KABUR DEATHZONE → {reason}"
            potion = next(
                (i for i in inventory if isinstance(i, dict) and i.get("category") == "recovery"),
                None
            )
            if potion and hp < me.get("maxHp", 100):
                return {"type": "use_item", "itemId": potion["id"]}, "Terjebak DZ, heal!"
            return {"type": "rest"}, "Terjebak DZ, tidak ada jalan keluar!"

        # =============================================
        # 1b. HINDARI PENDING DEATH ZONE
        # =============================================
        pending_ids = self.get_pending_deathzone_ids(data)
        if region_id_now in pending_ids:
            log(self.index, "⏰  DEATHZONE — pindah sebelum terlambat!", self.current_turn, self.game_id)

            rid, reason = self.safe_move(data, curr_region)
            if rid:
                return {"type": "move", "regionId": rid}, f"KABUR DZ → {reason}"

        # =============================================
        # 1c. CEK RECENT LOGS — KABUR JIKA DISERANG
        # =============================================
        my_equipped = me.get("equippedWeapon", {})
        my_bonus = my_equipped.get("atkBonus", 0) if isinstance(my_equipped, dict) else 0

        recent_logs = data.get("recentLogs", [])
        should_flee    = False
        attacker_region = None
        flee_reason    = ""

        attacker_ids_who_hit_us = set()
        for entry in recent_logs:
            if not isinstance(entry, dict):
                continue
            if entry.get("type") != "attack":
                continue
            details = entry.get("details", {})
            if not isinstance(details, dict):
                continue
            if details.get("defenderId") != self.agent_id:
                continue
            attacker_ids_who_hit_us.add(entry.get("agentId"))

        if len(attacker_ids_who_hit_us) >= 2:
            should_flee = True
            flee_reason = f"Diserang, kabur!"
            log(self.index, f"⚠️  {flee_reason}", self.current_turn, self.game_id)

        fight_back_target = None
        if not should_flee and len(attacker_ids_who_hit_us) == 1:
            for entry in recent_logs:
                if not isinstance(entry, dict):
                    continue
                if entry.get("type") != "attack":
                    continue
                details = entry.get("details", {})
                if not isinstance(details, dict):
                    continue
                if details.get("defenderId") != self.agent_id:
                    continue
                attacker_id = entry.get("agentId")
                for a in data.get("visibleAgents", []):
                    if not isinstance(a, dict) or a.get("id") != attacker_id:
                        continue
                    attacker_weapon = a.get("equippedWeapon", {})
                    attacker_bonus  = attacker_weapon.get("atkBonus", 0) if isinstance(attacker_weapon, dict) else 0
                    a_region = a.get("regionId")
                    if a_region and a_region != region_id_now:
                        should_flee    = True
                        attacker_region = a_region
                        flee_reason    = f"Diserang oleh {a.get('name','?')}, kabur!"
                        log(self.index, f"⚠️  {flee_reason}", self.current_turn, self.game_id)
                    elif attacker_bonus < my_bonus:
                        fight_back_target = a
                        log(self.index, f"💥 Diserang {a.get('name','?')} (ATK:{attacker_bonus} < mine:{my_bonus}) — balas serang!", self.current_turn, self.game_id)
                    elif attacker_bonus > my_bonus:
                        should_flee    = True
                        attacker_region = a_region
                        flee_reason    = f"Diserang {a.get('name','?')}, kabur!"
                        log(self.index, f"⚠️  {flee_reason}", self.current_turn, self.game_id)
                    break
                if should_flee or fight_back_target:
                    break

        if fight_back_target and not should_flee and ep >= 1:
            recovery_item = next(
                (i for i in inventory if isinstance(i, dict) and i.get("category") == "recovery"
                 and i.get("typeId") != "energy_drink"),
                None
            )
            if hp < 40 and not recovery_item:
                log(self.index, f"HP rendah ({hp}) dan tidak ada recovery — batalkan balas, kabur!", self.current_turn, self.game_id)
                rid, reason = self.safe_move(data, curr_region)
                if rid:
                    return {"type": "move", "regionId": rid}, f"HP kritis! ({reason})"
            elif hp < 40 and recovery_item:
                log(self.index, f"HP rendah ({hp}) saat fight back — heal dulu!", self.current_turn, self.game_id)
                return {"type": "use_item", "itemId": recovery_item["id"]}, f"Heal!"
            else:
                return (
                    {"type": "attack", "targetId": fight_back_target["id"], "targetType": "agent"},
                    f"Serang {fight_back_target.get('name','?')}!"
                )

        if should_flee:
            extra = {attacker_region} if attacker_region else None
            rid, reason = self.safe_move(data, curr_region, extra_danger_ids=extra)
            if rid:
                return {"type": "move", "regionId": rid}, f"Kabur! {flee_reason} ({reason})"

        # =============================================
        # 2. SURVIVAL: HEAL JIKA SEKARAT
        # =============================================
        if hp < 40:
            potion = next(
                (i for i in inventory if isinstance(i, dict) and i.get("category") == "recovery"),
                None
            )
            if potion:
                return {"type": "use_item", "itemId": potion["id"]}, f"HP kritis."
            enemies_here = [
                a for a in data.get("visibleAgents", [])
                if isinstance(a, dict)
                and a.get("regionId") == region_id_now
                and a.get("id") != self.agent_id
                and a.get("isAlive")
            ]
            if enemies_here:
                rid, reason = self.safe_move(data, curr_region)
                if rid:
                    return {"type": "move", "regionId": rid}, f"HP tipis! ({reason})"

        # =============================================
        # 3. ENERGY MANAGEMENT
        # =============================================
        if ep < 2:
            energy_drink = next(
                (i for i in inventory if isinstance(i, dict) and i.get("typeId") == "energy_drink"),
                None
            )
            if energy_drink:
                return {"type": "use_item", "itemId": energy_drink["id"]}, f"EP habis."
            return {"type": "rest"}, "EP habis, istirahat dulu."

        # =============================================
        # BUILD HELPER DATA
        # =============================================
        armed = self.has_weapon(inventory)
        danger = self.build_danger_set(data)
        connected_map = {
            r["id"]: r
            for r in data.get("connectedRegions", [])
            if isinstance(r, dict) and r.get("id")
        }

        visible_agents = [
            a for a in data.get("visibleAgents", [])
            if isinstance(a, dict)
            and a.get("id") != self.agent_id
            and a.get("isAlive")
            and a.get("name") not in self.friendly_bots
        ]
        visible_monsters = [
            m for m in data.get("visibleMonsters", [])
            if isinstance(m, dict)
        ]
        targets_here  = [
            a for a in visible_agents
            if a.get("regionId") == region_id_now
            and a.get("name") not in self.friendly_bots
        ]
        monsters_here = [m for m in visible_monsters if m.get("regionId") == region_id_now]

        # =============================================
        # 4. PHASE: BELUM PUNYA SENJATA
        # =============================================
        if not armed:
            if targets_here:
                enemy_ids = {a.get("regionId") for a in targets_here if a.get("regionId")}
                rid, reason = self.safe_move(data, curr_region, extra_danger_ids=enemy_ids)
                if rid:
                    return {"type": "move", "regionId": rid}, f"Kabur! ({reason})"

            attack_targets = [
                m for m in monsters_here
                if isinstance(m, dict) and m.get("name", "").lower() in ("bear", "bandit")
            ]
            if attack_targets and ep >= 2:
                target = attack_targets[0]
                return (
                    {"type": "attack", "targetId": target["id"], "targetType": "monster"},
                    f"Serang!"
                )

            if targets_here:
                enemy_ids = {a.get("regionId") for a in targets_here if a.get("regionId")}
                rid, reason = self.safe_move(data, curr_region, extra_danger_ids=enemy_ids)
                if rid:
                    return {"type": "move", "regionId": rid}, f"Kabur! ({reason})"

            ALLOWED_WEAPONS = ("sword", "pistol", "katana", "sniper")
            MAX_CARRY = {
                "bandage":        3,
                "emergency_food": 4,
                "map":            1,
                "binoculars":     1,
            }
            inv_count_now = {}
            for inv_item in inventory:
                if isinstance(inv_item, dict):
                    tid = inv_item.get("typeId", "")
                    inv_count_now[tid] = inv_count_now.get(tid, 0) + inv_item.get("quantity", 1)

            def worth_picking(item_dict):
                type_id  = item_dict.get("typeId", "")
                category = item_dict.get("category", "")
                if category == "weapon" and type_id not in ALLOWED_WEAPONS:
                    return False
                if type_id in MAX_CARRY:
                    return inv_count_now.get(type_id, 0) < MAX_CARRY[type_id]
                return True

            items_here = [
                i for i in data.get("visibleItems", [])
                if isinstance(i, dict)
                and i.get("regionId") == region_id_now
                and isinstance(i.get("item"), dict)
                and (
                    i["item"].get("category") in ("recovery", "currency")
                    or i["item"].get("category") == "weapon"
                )
                and worth_picking(i["item"])
            ]
            if items_here:
                return {"type": "explore"}, "Kumpulkan item di sini dulu."

            conn_ids = set(curr_region.get("connections", []))

            def is_reachable_safe(tile_id):
                if not isinstance(tile_id, str) or tile_id not in conn_ids:
                    return False
                if tile_id in connected_map:
                    return not connected_map[tile_id].get("isDeathZone", False)
                return tile_id not in danger

            visible_items = [
                i for i in data.get("visibleItems", [])
                if isinstance(i, dict) and isinstance(i.get("item"), dict)
            ]

            weapon_tiles = [
                i["regionId"] for i in visible_items
                if i["item"].get("category") == "weapon"
                and i["item"].get("typeId") in ALLOWED_WEAPONS
                and i.get("regionId") != region_id_now
                and is_reachable_safe(i.get("regionId", ""))
            ]
            recovery_tiles = [
                i["regionId"] for i in visible_items
                if i["item"].get("category") in ("recovery", "currency")
                and i.get("regionId") != region_id_now
                and is_reachable_safe(i.get("regionId", ""))
            ]

            for target_tiles, label in [(weapon_tiles, "senjata"), (recovery_tiles, "recovery/moltz")]:
                if target_tiles:
                    return {"type": "move", "regionId": random.choice(target_tiles)}, f"Menuju tile {label}"

            rid, reason = self.safe_move(data, curr_region)
            if rid:
                return {"type": "move", "regionId": rid}, f"Jelajah ({reason})"
            return {"type": "explore"}, "Tidak ada gerak, explore."

        # =============================================
        # 5. PHASE: SUDAH PUNYA SENJATA
        # =============================================
        my_equipped      = me.get("equippedWeapon", {})
        equipped_type_id = my_equipped.get("typeId", "") if isinstance(my_equipped, dict) else ""

        if equipped_type_id in ("knife", "bow"):
            bear_bandit_here = [
                m for m in monsters_here
                if isinstance(m, dict) and m.get("name", "").lower() in ("bear", "bandit")
            ]
            if bear_bandit_here and ep >= 2:
                target = bear_bandit_here[0]
                return (
                    {"type": "attack", "targetId": target["id"], "targetType": "monster"},
                    f"Serang!"
                )

        if targets_here:
            enemy_ids = {a.get("regionId") for a in targets_here if a.get("regionId")}
            rid, reason = self.safe_move(data, curr_region, extra_danger_ids=enemy_ids)
            if rid:
                return {"type": "move", "regionId": rid}, f"Musuh, kabur! ({reason})"

        rid, reason = self.safe_move(data, curr_region)
        if rid:
            return {"type": "move", "regionId": rid}, f"Pindah ke tile ({reason})"
        return {"type": "explore"}, "Tidak ada tile, explore."

    # =========================
    # FULL RESET
    # =========================
    def full_reset(self):
        log(self.index, "FULL RESET", self.current_turn, self.game_id)

        self.agent_id = None
        self.game_id  = None
        self.visited  = set()
        self.dz_regions = set()
        self.move_counter = 0
        self.last_activity_region = None
        self.picked_starter_weapon = False
        self.replied_message_ids = set()  # clear reply tracker for new game

        for f in [self.mem_file, self.agent_file, self.game_file, self.dzreg_file]:
            if os.path.exists(f):
                os.remove(f)

        log(self.index, "DZ regions cleared (new game).", self.current_turn, self.game_id)
        self.last_heartbeat = time.time()
        self.current_turn   = 0

    # =========================
    # MAIN LOOP
    # =========================
    def run_loop(self):
        self.startup()

        while True:
            try:
                res = safe_request(
                    "get",
                    f"{BASE_URL}/games/{self.game_id}/agents/{self.agent_id}/state",
                    headers=self.headers,
                    timeout=20
                )

                if res is None or res.status_code != 200:
                    self.state_fail_count += 8
                    status_code = res.status_code if res else "None"
                    log(self.index, f"State fetch failed ({status_code}) [{self.state_fail_count}/{self.state_fail_limit}] -> waiting 30s...", self.current_turn, self.game_id)
                    if self.state_fail_count >= self.state_fail_limit:
                        log(self.index, f"Too many failures -> resetting", self.current_turn, self.game_id)
                        self.state_fail_count = 0
                        self.full_reset()
                        self.find_and_join_game()
                    else:
                        time.sleep(30)
                    continue

                self.state_fail_count = 0
                data = res.json()["data"]
                me   = data["self"]
                self.last_heartbeat = time.time()

                # Pull turn counter from API — always increment locally first
                # as a reliable fallback, then try to sync with the server value.
                # The API field name may vary (currentTurn / turnNumber / turn).
                self.current_turn += 1
                for _fname in ("currentTurn", "turnNumber", "turn"):
                    _tval = data.get(_fname)
                    if isinstance(_tval, int) and _tval > 0:
                        self.current_turn = _tval
                        break

                # Track picked_starter_weapon from current inventory
                inv_now = me.get("inventory", [])
                if any(isinstance(i, dict) and i.get("typeId") in ("knife", "bow") for i in inv_now):
                    self.picked_starter_weapon = True

                # Update dz regions every turn
                self.update_dz_regions(data)

                # ── Whisper reply: respond ONLY to whispers from Guardian, answered by Ollama LLM ──
                my_region = me.get("regionId", "")
                for msg_entry in data.get("recentMessages", []):
                    if not isinstance(msg_entry, dict):
                        continue
                    msg_id      = msg_entry.get("id")
                    sender_id   = msg_entry.get("senderId")
                    sender_name = msg_entry.get("senderName", "")
                    msg_type    = msg_entry.get("type")         # "private" = whisper
                    target_id   = msg_entry.get("targetId")     # must be our own agent
                    msg_region  = msg_entry.get("regionId", "")
                    raw_content = msg_entry.get("content", "")
                    # Strip [Curse] prefix — question starts after it
                    if "[Curse]" in raw_content:
                        msg_text = raw_content.split("[Curse]", 1)[-1].strip()
                    else:
                        msg_text = raw_content.strip()
                    # Replace newlines with a period-space for clean single-line query
                    msg_text = msg_text.replace("\n", ". ").strip()

                    # Only process whispers directed at us that we haven't replied to yet
                    if not (
                        msg_type == "private"
                        and target_id == self.agent_id
                        and msg_id not in self.replied_message_ids
                    ):
                        continue

                    # ── GUARDIAN FILTER: ignore whispers from anyone else ──
                    if sender_name != GUARDIAN_NAME:
                        log(self.index, f"💬 Whisper from {sender_name} — ignored (not Guardian)", self.current_turn, self.game_id)
                        self.replied_message_ids.add(msg_id)
                        continue

                    # Only whisper back if sender is still in the same region
                    if msg_region and msg_region != my_region:
                        log(self.index, f"💬 Guardian whisper (different region) — skipping reply", self.current_turn, self.game_id)
                        self.replied_message_ids.add(msg_id)
                        continue

                    # ── Ask Ollama for an answer ──
                    log(self.index, f"💬 Guardian asked: {msg_text!r} — querying Ollama...", self.current_turn, self.game_id)
                    prompt = f"{msg_text}. Provide only the final answer in maximum 190 characters."
                    reply_msg = ask_ollama(prompt)
                    # If model repeated the question, strip it — take only text after last "="
                    if "=" in reply_msg:
                        reply_msg = reply_msg.split("=")[-1].strip()
                    # Remove all symbols except letters, digits, spaces, . and ,
                    import re
                    reply_msg = re.sub(r"[^\w\s.,+\-x:/=]", "", reply_msg)
                    reply_msg = " ".join(reply_msg.split())
                    # Truncate reply if too long (game API may have message length limits)
                    if len(reply_msg) > 500:
                        reply_msg = reply_msg[:190] + "..."
 
                    log(self.index, f"💬 Whisper reply to Guardian: {reply_msg}", self.current_turn, self.game_id)
                    self.send_action(
                        {"type": "whisper", "targetId": sender_id, "message": reply_msg},
                        "Guardian reply"
                    )
                    self.replied_message_ids.add(msg_id)

                # Game finished — bot survived to end
                if data["gameStatus"] == "finished":
                    log(self.index, "Game finished -> searching new game", self.current_turn, self.game_id)
                    self.full_reset()
                    self.find_and_join_game()
                    continue

                # Agent dead — wait for game to finish before joining next
                if not data["self"]["isAlive"]:
                    log(self.index, "Agent dead -> waiting for game to finish", self.current_turn, self.game_id)
                    self._wait_for_game_finish()
                    self.full_reset()
                    self.find_and_join_game()
                    continue

                # --- FREE ACTIONS ---

                # Inventory full (10 items): consume emergency rations
                inventory_list = [i for i in me.get("inventory", []) if isinstance(i, dict)]
                total_items = len(inventory_list)
                if total_items >= 10:
                    rations = next(
                        (i for i in inventory_list if i.get("typeId") == "emergency_food"),
                        None
                    )
                    if rations:
                        log(self.index, f"Inventory penuh ({total_items}/10) — gunakan Emergency rations", self.current_turn, self.game_id)
                        self.send_action({"type": "use_item", "itemId": rations["id"]}, "Pakai rations")

                # Pickup allowed items respecting max-carry limits
                ALLOWED_LOOT = (
                    "reward1",
                    "sword", "pistol", "katana", "sniper",
                    "emergency_food", "bandage", "medkit", "energy_drink",
                    "map", "binoculars",
                )
                MAX_CARRY = {
                    "bandage":        3,
                    "emergency_food": 4,
                    "map":            1,
                    "binoculars":     1,
                }
                inv_count = {}
                for i in me.get("inventory", []):
                    if isinstance(i, dict):
                        tid = i.get("typeId", "")
                        inv_count[tid] = inv_count.get(tid, 0) + i.get("quantity", 1)

                for item_entry in data.get("visibleItems", []):
                    if not isinstance(item_entry, dict):
                        continue
                    item_data = item_entry.get("item", {})
                    if not isinstance(item_data, dict):
                        continue
                    if item_entry.get("regionId") != me.get("regionId"):
                        continue
                    type_id = item_data.get("typeId")
                    if type_id not in ALLOWED_LOOT:
                        continue
                    if type_id in MAX_CARRY:
                        if inv_count.get(type_id, 0) >= MAX_CARRY[type_id]:
                            log(self.index, f"Skip {item_data.get('name','?')}: sudah max ({MAX_CARRY[type_id]})", self.current_turn, self.game_id)
                            continue
                    log(self.index, f"Pickup: {item_data.get('name','?')} [{type_id}]", self.current_turn, self.game_id)
                    self.send_action({"type": "pickup", "itemId": item_data["id"]}, "Ambil barang")
                    inv_count[type_id] = inv_count.get(type_id, 0) + 1
                    if type_id == "map":
                        log(self.index, "Auto-use Map: reveal full map!", self.current_turn, self.game_id)
                        self.send_action({"type": "use_item", "itemId": item_data["id"]}, "Map")

                # Equip best weapon by atkBonus
                inventory_items = [i for i in me.get("inventory", []) if isinstance(i, dict)]
                weapons = [i for i in inventory_items if i.get("category") == "weapon"]
                if weapons:
                    best_weapon = sorted(weapons, key=lambda x: x.get("atkBonus", 0), reverse=True)[0]
                    equipped    = me.get("equippedWeapon", {})
                    equipped_id = equipped.get("id") if isinstance(equipped, dict) else None
                    if best_weapon["id"] != equipped_id:
                        log(self.index, f"Equip senjata terbaik: {best_weapon.get('name','?')} (ATK+{best_weapon.get('atkBonus',0)})", self.current_turn, self.game_id)
                        self.send_action({"type": "equip", "itemId": best_weapon["id"]}, "Equip")

                result = self.decide_action(data)
                if not isinstance(result, tuple) or len(result) != 2:
                    log(self.index, f"decide_action returned invalid result: {result}, fallback to rest", self.current_turn, self.game_id)
                    action, reason = {"type": "rest"}, "decide_action error fallback"
                else:
                    action, reason = result
                self.send_action(action, reason)
                log(self.index, f"ACTION {action['type']} | reason: {reason}", self.current_turn, self.game_id)

                self.heartbeat_watchdog()
                time.sleep(60)

            except Exception as e:
                log(self.index, f"Loop error: {e}", self.current_turn, self.game_id)
                time.sleep(10)


if __name__ == "__main__":
    NyrAgent().run_loop()
