"""RTX Control Center — web UI cho hệ thống multi mini model.

Stdlib thuần (http.server) — không thêm dependency; chạy:
    uv run rtx-dashboard            # http://127.0.0.1:8420

API:
  GET  /api/matrix          matrix model (quét logs mới mỗi lần gọi)
  GET  /api/live            GPU + tiến trình train + queue (JSON)
  GET  /api/twin/<key>      GIF digital twin mới nhất của model
  GET  /api/trainlog        40 dòng cuối log train đang chạy / gần nhất
  POST /api/action          {action: smoke|train|export|demo|queue|scan, key?}
                            — chạy nền qua `rtx-pipeline`, một hành động
                            chiếm GPU (train/smoke/queue) mỗi lúc.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import pipeline
from .registry import FAMILIES, RTX_STATE_DIR, TWIN_DIR

STATIC_DIR = Path(__file__).parent / "static"

# Các hành động cần GPU — chỉ một mỗi lúc (RTX 4060 8 GB).
GPU_ACTIONS = {"smoke", "train", "queue"}


def _spawn_pipeline(args: list[str], log_name: str) -> Path:
    RTX_STATE_DIR.mkdir(parents=True, exist_ok=True)
    log_path = RTX_STATE_DIR / log_name
    env_pipeline = pipeline._base_env()
    with open(log_path, "ab") as lf:
        subprocess.Popen(
            ["uv", "run", "rtx-pipeline", *args],
            stdout=lf, stderr=subprocess.STDOUT,
            start_new_session=True, env=env_pipeline, cwd=str(pipeline.REPO_ROOT),
        )
    return log_path


def make_handler() -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *a):  # gọn log server
            pass

        # ------------------------------------------------------------- GET
        def do_GET(self):
            if self.path in ("/", "/index.html"):
                return self._file(STATIC_DIR / "index.html", "text/html; charset=utf-8")
            if self.path.startswith("/static/"):
                name = self.path[len("/static/"):]
                ctype = "text/css" if name.endswith(".css") else \
                        "application/javascript" if name.endswith(".js") else "application/octet-stream"
                return self._file(STATIC_DIR / name, ctype)
            if self.path == "/api/matrix":
                from .matrix import scan
                return self._json(scan())
            if self.path == "/api/live":
                return self._json(pipeline.live_status())
            if self.path.startswith("/api/twin/"):
                key = self.path[len("/api/twin/"):]
                if key not in FAMILIES:
                    return self._json({"error": "unknown family"}, 404)
                gif = TWIN_DIR / key / "latest.gif"
                return self._file(gif, "image/gif") if gif.exists() else self._json({"error": "chưa có"}, 404)
            if self.path == "/api/trainlog":
                return self._json(self._trainlog())
            return self._json({"error": "not found"}, 404)

        # ------------------------------------------------------------ POST
        def do_POST(self):
            if self.path != "/api/action":
                return self._json({"error": "not found"}, 404)
            try:
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                return self._json({"error": "bad json"}, 400)
            action = body.get("action", "")
            key = body.get("key", "")

            if action not in {"smoke", "train", "export", "demo", "queue", "scan"}:
                return self._json({"error": f"hành động lạ: {action}"}, 400)
            if action in {"smoke", "train", "export", "demo"} and key not in FAMILIES:
                return self._json({"error": f"family lạ: {key}"}, 400)

            live = pipeline.live_status()
            if action in GPU_ACTIONS and live["training"]:
                pids = [t["pid"] for t in live["training"]]
                return self._json({"error": f"GPU đang bận (train PID {pids}) — dùng Queue để xếp hàng",
                                   "busy": True}, 409)

            ts = time.strftime("%Y%m%d_%H%M%S")
            args = {"smoke": ["smoke", key], "train": ["train", key, "--chain"],
                    "export": ["export", key], "demo": ["demo", key],
                    "queue": ["queue"], "scan": ["scan"]}[action]
            log = _spawn_pipeline(args, f"action_{action}_{key or 'all'}_{ts}.log")
            return self._json({"ok": True, "action": action, "key": key, "log": str(log)})

        # ---------------------------------------------------------- helpers
        def _headers(self, code=200, ctype="application/json; charset=utf-8", size=0):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(size))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()

        def _json(self, obj, code=200):
            data = json.dumps(obj, ensure_ascii=False).encode()
            self._headers(code, size=len(data))
            self.wfile.write(data)

        def _file(self, path: Path, ctype: str):
            if not path.is_file():
                data = b'{"error": "not found"}'
                self._headers(404, size=len(data))
                self.wfile.write(data)
                return
            data = path.read_bytes()
            self._headers(200, ctype, size=len(data))
            self.wfile.write(data)

        def _trainlog(self) -> dict:
            """Log train mới nhất: của family đang chạy, hoặc file mới nhất."""
            live = pipeline.live_status()
            fam = live["training"][0].get("family") if live.get("training") else None
            tlog_dir = RTX_STATE_DIR / "train_logs"
            logs = sorted(tlog_dir.glob("train_*.log"),
                          key=lambda p: p.stat().st_mtime) if tlog_dir.exists() else []
            chosen = None
            if fam:
                cand = [p for p in logs if f"train_{fam}_" in p.name]
                chosen = cand[-1] if cand else None
            chosen = chosen or (logs[-1] if logs else None)
            if not chosen:
                return {"lines": [], "file": None}
            lines = chosen.read_text(errors="replace").splitlines()[-40:]
            return {"lines": lines, "file": chosen.name}

    return Handler


def main():
    ap = argparse.ArgumentParser(prog="rtx-dashboard",
                                 description="RTX Control Center — dashboard multi mini model")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8420)
    args = ap.parse_args()

    httpd = ThreadingHTTPServer((args.host, args.port), make_handler())
    print(f"🦆 RTX Control Center: http://{args.host}:{args.port}")
    print("   Ctrl+C để dừng")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nĐã dừng.")


if __name__ == "__main__":
    main()
