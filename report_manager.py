"""
Report Manager Module
Manages data sending to API - Periodic reports + Rate-limited alarms

Usage:
    from report_manager import ReportManager

    manager = ReportManager()

    # Define periodic report
    manager.add_periodic_report(
        name="table_status",
        interval_seconds=60,
        data_func=lambda: {"occupied": 5, "empty": 20}
    )

    # Define alarm
    manager.add_alarm(
        name="no_barista",
        cooldown_seconds=5  # Send at most once every 5s
    )

    # In main loop
    # 1. Check periodic reports
    manager.check_reports()

    # 2. Send alarm
    if violation:
        manager.send_alarm("no_barista", data={"message": "barista absent for 1 min"})
"""

import json

#####yeni######
import os
import time
import collections
import datetime
import threading
import typing

import requests

###############

MAX_QUEUE_SIZE = 500   # oldest item dropped when queue reaches this size
RETRY_DELAY   = 5      # retry interval when API is down (s)

######yeni######
LOG_FILE = "queue_log.json"
###############


class ReportManager:

    def __init__(self, gateway_base: str = None, api_key: str = None,
                 branch_id: str = None,
                 ######yeni######
                 log_file: str = LOG_FILE
                 ###############
                 ) -> None:
        self.gateway_base: str = gateway_base
        self.api_key: str = api_key
        self.branch_id: str = branch_id or "00000000-0000-0000-0000-000000000000"

        ######yeni######
        self._log_file: str = log_file
        ###############

        # Periodic reports and alarms
        self.periodic_reports: typing.Dict = {}
        self.alarms: typing.Dict = {}

        # ── Send queue ───────────────────────────────────────────
        # Each item: {"name": ..., "payload": {...}, "is_alarm": bool}
        self._queue: collections.deque = collections.deque()
        self._queue_lock = threading.Lock()

        ######yeni######
        # Last known server status. We assume True initially;
        # drops to False on first failed attempt and starts flushing to file.
        self._was_online = True
        self._load_queue()
        ###############

        # Background sender thread
        self._sender_thread = threading.Thread(target=self._sender_loop, daemon=True)
        self._sender_thread.start()

        print("✅ Report Manager ready")


    # ── Configuration ─────────────────────────────────────────────

    def set_api_credentials(self, gateway_base: str, api_key: str,
                            branch_id: str = None) -> None:
        self.gateway_base: str = gateway_base
        self.api_key: str = api_key
        if branch_id:
            self.branch_id: str = branch_id
        print(f"✅ API credentials updated: {gateway_base}")


    def add_periodic_report(self, name: str, interval_seconds: int,
                            data_func: typing.Callable = None,
                            camera_id: str = None,
                            module_id: str = None) -> None:
        self.periodic_reports[name] = {
            "interval":   interval_seconds,
            "data_func":  data_func,
            "last_sent":  datetime.datetime.now(),
            "camera_id":  camera_id,
            "module_id":  module_id,
        }
        print(f"✅ Periodic report added: {name} ({interval_seconds}s)")


    def add_alarm(self, name: str, cooldown_seconds: int = 5,
                  camera_id: str = None,
                  module_id: str = None,
                  once_per_day: bool = False,
                  data: typing.Dict = None) -> None:
        self.alarms[name] = {
            "cooldown":    cooldown_seconds,
            "last_sent":   None,
            "camera_id":   camera_id,
            "module_id":   module_id,
            "once_per_day": once_per_day,
            "sent_date":   None,
            "data":        data or {},
        }
        print(f"✅ Alarm added: {name} (cooldown: {cooldown_seconds}s)")


    # ── Public interface (used by main.py) ───────────────────────

    def check_reports(self) -> typing.Dict[str, bool]:
        """
        Check timing of periodic reports.
        Returns True for due reports — does NOT add to queue.
        Call send_report() to enqueue.
        Returns: {name: is_due}
        """
        results = {}
        now: datetime.datetime = datetime.datetime.now()

        for name, config in self.periodic_reports.items():
            if config["last_sent"] is None:
                should_send = True
            else:
                elapsed = (now - config["last_sent"]).total_seconds()
                should_send = elapsed >= config["interval"]

            if should_send:
                config["last_sent"] = now
                results[name] = True
            else:
                results[name] = False

        return results


    def send_report(self, name: str, data: typing.Dict = None,
                    media_path: str = None, message: str = None) -> bool:
        """
        Enqueue periodic report with data and media_path.
        Called after check_reports() returns True.
        """
        if name not in self.periodic_reports:
            print(f"❌ Report not defined: {name}")
            return False

        config = self.periodic_reports[name]
        self._enqueue(
            name=name,
            data=data,
            camera_id=config["camera_id"],
            module_id=config["module_id"],
            media_path=media_path,
            message=message,
            is_alarm=False,
        )
        return True


    def can_send_alarm(self, name: str) -> bool:
        """Has alarm cooldown elapsed and can it be sent? (check before MinIO upload)"""
        if name not in self.alarms:
            return False
        config = self.alarms[name]
        now: datetime.datetime = datetime.datetime.now()
        if config["once_per_day"] and config["sent_date"] == now.date():
            return False
        if config["last_sent"] is not None:
            elapsed = (now - config["last_sent"]).total_seconds()
            if elapsed < config["cooldown"]:
                return False
        return True


    def send_alarm(self, name: str, data: typing.Dict = None,
                   media_path: str = None, message: str = None) -> bool:
        """
        Enqueue alarm (rate limited).
        Returns: whether it was enqueued
        """
        if name not in self.alarms:
            print(f"❌ Alarm not defined: {name}")
            return False

        config = self.alarms[name]
        now: datetime.datetime = datetime.datetime.now()

        if config["once_per_day"]:
            if config["sent_date"] == now.date():
                return False

        if config["last_sent"] is not None:
            elapsed = (now - config["last_sent"]).total_seconds()
            if elapsed < config["cooldown"]:
                return False

        self._enqueue(
            name=name,
            data={**config["data"], **(data or {})},
            camera_id=config["camera_id"],
            module_id=config["module_id"],
            media_path=media_path,
            message=message,
            is_alarm=True,
        )

        config["last_sent"] = now
        if config["once_per_day"]:
            config["sent_date"] = now.date()

        return True


    # ── Queue management ─────────────────────────────────────────

    def _enqueue(self, name: str, data: typing.Dict = None,
                 camera_id: str = None, module_id: str = None,
                 media_path: str = None, message: str = None,
                 is_alarm: bool = False) -> None:
        """Enqueue data. Drops oldest if queue is full."""
        payload = {
            "cameraId":        camera_id,
            "moduleId":        module_id,
            "branchId":        self.branch_id,
            "triggeredAt":     datetime.datetime.utcnow().isoformat() + "Z",
            "mediaFolderPath": media_path,
            "input":           data or {},
            "data":            data or {},
            "message":         message,
        }
        ######yeni######
        # Note: 'sent' field removed — only pending items written to file
        item = {"name": name, "payload": payload, "is_alarm": is_alarm}
        ###############

        with self._queue_lock:
            if len(self._queue) >= MAX_QUEUE_SIZE:
                dropped = self._queue.popleft()
                print(f"⚠️  Queue full, dropped oldest item: {dropped['name']}")
            self._queue.append(item)
            ######yeni######
            # If server is known to be down, immediately flush new item to disk.
            # This ensures data is not lost even if the program crashes,
            # without waiting for the next sender_loop attempt.
            if not self._was_online:
                self._flush_queue_to_log()
            ###############

        qsize: int = self.queue_size()
        if qsize > 1:
            print(f"📥 Enqueued: {name} ({qsize} items in queue)")


    def _sender_loop(self) -> typing.NoReturn:
        """Background thread that drains the queue in order."""
        ######yeni######
        # self._was_online defined in __init__; transitions managed here.
        # - Server up → down: write queue to disk, was_online=False
        # - Server down → down: write to disk again (include newly added items)
        # - Server down → up: delete file, was_online=True
        ###############

        while True:
            with self._queue_lock:
                has_item: bool = len(self._queue) > 0
                item = self._queue[0] if has_item else None

            if item is None:
                time.sleep(0.5)
                continue

            success: bool = self._send_now(item)

            if success:
                with self._queue_lock:
                    # Remove same item only if no other thread has replaced it
                    if self._queue and self._queue[0] is item:
                        self._queue.popleft()

                    ######yeni######
                    # Did server come back?
                    if not self._was_online:
                        print("✅ Server connection restored")
                        self._was_online = True

                    if len(self._queue) == 0:
                        # Queue empty, no need for file
                        self._delete_log()
                    else:
                        # Still pending → refresh file with current queue
                        # (sent item should no longer be in file)
                        self._flush_queue_to_log()
                    ###############

            else:
                ######yeni######
                # Server down: write queue to disk on EVERY failed attempt.
                # Ensures newly added items are also saved while server is down.
                if self._was_online:
                    print("⚠️  Server down, writing queue to disk...")
                    self._was_online = False
                with self._queue_lock:
                    self._flush_queue_to_log()
                ###############

                # No connection; wait before retrying
                time.sleep(RETRY_DELAY)


    def _send_now(self, item: typing.Dict, timeout: int = 10) -> bool:
        """Actually send a single queued item to the API."""
        if self.gateway_base is None or self.api_key is None:
            return False

        name      = item["name"]
        payload   = item["payload"]
        is_alarm  = item["is_alarm"]

        try:
            url: str = f"{self.gateway_base.rstrip('/')}/AiInput"
            headers: dict[str, str] = {
                "Content-Type": "application/json",
                "X-API-KEY":    self.api_key,
            }

            response: requests.Response = requests.post(url, json=payload, headers=headers,
                                     timeout=timeout)

            if response.status_code == 200:
                report_type: str = "🚨 Alarm" if is_alarm else "📊 Report"
                qsize: int = self.queue_size()
                suffix: str = f" ({qsize} items remaining in queue)" if qsize else ""
                print(f"{report_type} sent: {name}{suffix}")
                return True
            else:
                try:
                    body = response.text[:300]
                except Exception:
                    body = "(unreadable)"
                print(f"❌ API error ({name}): {response.status_code} — {body}")
                return False

        except requests.exceptions.Timeout:
            print(f"❌ Timeout ({name}): {timeout}s")
            return False

        except requests.exceptions.ConnectionError:
            print(f"❌ Connection error ({name}): {self.gateway_base}")
            return False

        except Exception as e:
            print(f"❌ Send error ({name}): {e}")
            return False


    # ── Status queries ───────────────────────────────────────────

    ######yeni######
    def _read_log(self) -> list:
        if not os.path.exists(self._log_file):
            return []
        try:
            with open(self._log_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️  Log read error: {e}")
            return []

    def _write_log(self, logs: list) -> None:
        try:
            with open(self._log_file, "w", encoding="utf-8") as f:
                json.dump(logs, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️  Log write error: {e}")

    def _delete_log(self) -> None:
        """Delete the file when server comes back and queue is empty."""
        if os.path.exists(self._log_file):
            try:
                os.remove(self._log_file)
            except Exception as e:
                print(f"⚠️  Log delete error: {e}")

    def _flush_queue_to_log(self) -> None:
        """
        Write the ENTIRE current queue to file (snapshot).
        IMPORTANT: Must be called while holding _queue_lock.
        """
        snapshot = list(self._queue)
        self._write_log(snapshot)

    def _load_queue(self) -> None:
        """
        Load any pending notifications from file into queue on startup.
        Does NOT delete the file — we haven't sent yet, keep it safe.
        """
        logs = self._read_log()
        for item in logs:
            # Backward compatibility: skip if 'sent' field is present
            # (only pending should be in file, but kept for safety with old log files)
            if item.get("sent") is True:
                continue
            # Clean up old format
            item.pop("sent", None)
            self._queue.append(item)
        if logs:
            print(f"♻️  Pending notifications loaded: {len(self._queue)} items")
            # Data remaining from previous run → server was probably down.
            # Start with this assumption, will switch to True on first successful send.
            self._was_online = False
    ###############


    def queue_size(self) -> int:
        with self._queue_lock:
            return len(self._queue)


    def get_status(self) -> typing.Dict:
        now: datetime.datetime = datetime.datetime.now()

        reports_status = {}
        for name, config in self.periodic_reports.items():
            if config["last_sent"]:
                elapsed  = (now - config["last_sent"]).total_seconds()
                next_in: int  = max(0, config["interval"] - elapsed)
            else:
                next_in = 0
            reports_status[name] = {
                "interval":        config["interval"],
                "last_sent":       config["last_sent"],
                "next_in_seconds": next_in,
            }

        alarms_status = {}
        for name, config in self.alarms.items():
            if config["last_sent"]:
                elapsed  = (now - config["last_sent"]).total_seconds()
                ready_in: int = max(0, config["cooldown"] - elapsed)
            else:
                ready_in = 0
            alarms_status[name] = {
                "cooldown":         config["cooldown"],
                "last_sent":        config["last_sent"],
                "ready_in_seconds": ready_in,
                "once_per_day":     config["once_per_day"],
                "sent_today":       config["sent_date"] == now.date()
                                    if config["once_per_day"] else None,
            }

        return {
            "pending_queue": self.queue_size(),
            "reports":       reports_status,
            "alarms":        alarms_status,
        }


    def reset_daily(self) -> None:
        for name, config in self.alarms.items():
            if config["once_per_day"]:
                config["sent_date"] = None
        print("✅ Daily alarms reset")
