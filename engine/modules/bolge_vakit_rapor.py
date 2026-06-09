"""
Bolge Vakit Rapor (Aggregator)
------------------------------
No-stream / virtual camera module. Reads bolge_vakit_analizi state files
from other camera processes and aggregates them into a single report.

Type A (passive) — update() is a no-op. get_data() reads state files live.

Config example:
    {
        "type":    "bolge_vakit_rapor",
        "name":    "bolge_vakit_rapor_main",
        "sources": [
            {"state_name": "bolge_vakit_manav",       "zone": "Zone 1", "key": "Time Spent in the Greengrocer's"},
            {"state_name": "bolge_vakit_kasap",        "zone": "Zone 2", "key": "Time Spent in the Butcher"},
            {"state_name": "bolge_vakit_bakliyat",     "zone": "Zone 4", "key": "Time Spent in the Pulses Section"},
            {"state_name": "bolge_vakit_hijyen",       "zone": "Zone 5", "key": "Time Spent in the Hygiene Section"},
            {"state_name": "bolge_soguk_dolap",        "zone": "Zone 6", "key": "Time Spent in the Chilled & Dairy Section"},
            {"state_name": "bolge_vakit_atistirmalik", "zone": "Zone 1", "key": "Time Spent in the Snacks Section"}
        ]
    }

get_data() output:
    {
        "Time Spent in the Greengrocer's":          float  (minutes),
        "Time Spent in the Butcher":                float,
        "Time Spent in the Pulses Section":         float,
        "Time Spent in the Hygiene Section":        float,
        "Time Spent in the Chilled & Dairy Section": float,
        "Time Spent in the Snacks Section":         float
    }
"""

import datetime
import json
import os

import cv2
import numpy as np

from .base import BaseModule

STATE_DIR = "state"


class BolgeVakitRaporModule(BaseModule):

    def __init__(self, name: str, sources: list = None, **_kwargs):
        self.name    = name
        self.sources = sources or []
        print(f"✅ BolgeVakitRaporModule ready [{name}] — {len(self.sources)} sources")

    # ==================== UPDATE ====================

    def update(self, bboxes, class_ids, scores, object_ids, frame, class_names: dict):
        pass

    # ==================== DATA ====================

    def get_data(self) -> dict:
        today  = datetime.datetime.now().date().isoformat()
        result = {}

        for src in self.sources:
            key  = src["key"]
            path = os.path.join(STATE_DIR, f"bolge_vakit_{src['state_name']}.json")
            try:
                with open(path, encoding="utf-8") as f:
                    state = json.load(f)
                if state.get("date") != today:
                    result[key] = 0.0
                    continue
                seconds       = float(state.get("zone_dwell", {}).get(src["zone"], 0.0))
                result[key]   = round(seconds / 60.0, 1)
            except FileNotFoundError:
                result[key] = 0.0
            except Exception as e:
                print(f"[{self.name}] State read error ({src['state_name']}): {e}")
                result[key] = 0.0

        return result

    # ==================== COMPOSITE FRAME ====================

    def compose_frame(self, minio) -> np.ndarray:
        """
        Downloads snapshot.jpg from MinIO for each source camera,
        composes a 3-column grid and returns it as a single cv2 frame.
        Sources must include a 'minio_folder' field.
        """
        CELL_W, CELL_H = 640, 360
        COLS           = 3
        PLACEHOLDER    = np.zeros((CELL_H, CELL_W, 3), dtype=np.uint8)

        cells = []
        for src in self.sources:
            folder = src.get("minio_folder")
            if folder:
                img = minio.download_snapshot(folder)
            else:
                img = None

            if img is None:
                cell = PLACEHOLDER.copy()
                cv2.putText(cell, folder or "N/A", (8, CELL_H // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 80), 1, cv2.LINE_AA)
            else:
                cell = cv2.resize(img, (CELL_W, CELL_H))

            label = src.get("key", folder or "")
            cv2.rectangle(cell, (0, 0), (CELL_W, 28), (0, 0, 0), -1)
            cv2.putText(cell, label, (6, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
            cells.append(cell)

        while len(cells) % COLS != 0:
            cells.append(PLACEHOLDER.copy())

        rows = [np.hstack(cells[i:i + COLS]) for i in range(0, len(cells), COLS)]
        grid = np.vstack(rows)

        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(grid, ts, (grid.shape[1] - 240, grid.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
        return grid

    # ==================== DRAW ====================

    def draw(self, frame):
        return frame

    # ==================== SHUTDOWN / RESET ====================

    def shutdown(self):
        pass

    def reset(self):
        pass
