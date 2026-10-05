"""Loopback-only HTTP API; authenticated, same-origin mutations."""
import argparse
import json
import secrets
import os
import socket
import signal
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit
from .registry import WorkError, catalog, strict
from .runtime import Runtime

STATIC = Path(__file__).parent / "static"


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False

    def server_bind(self):
        if os.name == "nt":
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()

    def __init__(self, port=8765, runtime=None):
        super().__init__(("127.0.0.1", port), Handler)
        self.runtime = runtime or Runtime()
        self.token = secrets.token_urlsafe(32)
        self.origin = "http://127.0.0.1:" + str(self.server_port)

    def server_close(self):
        if hasattr(self, "runtime"):
            self.runtime.close()
        super().server_close()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        # Do not log tokens, file contents or arguments.
        pass

    def send(self, status, value, content_type="application/json; charset=utf-8"):
        data = json.dumps(value, ensure_ascii=False).encode("utf-8") if content_type.startswith("application/json") else value
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; frame-ancestors 'none'")
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def authenticate(self, mutation=False):
        if self.headers.get("Host") != urlsplit(self.server.origin).netloc:
            raise WorkError("INVALID_HOST", "仅支持本机 127.0.0.1 服务地址")
        if self.headers.get("Authorization") != "Bearer " + self.server.token:
            raise WorkError("UNAUTHORIZED", "访问令牌无效，请使用启动时的地址打开")
        origin = self.headers.get("Origin")
        if mutation and origin != self.server.origin:
            raise WorkError("INVALID_ORIGIN", "修改 / 执行接口必须来自本服务来源")
        if origin and origin != self.server.origin:
            raise WorkError("INVALID_ORIGIN", "不接受跨来源请求")

    def do_GET(self):
        parsed = urlsplit(self.path)
        path = parsed.path
        try:
            if path.startswith("/api/"):
                self.authenticate()
                r = self.server.runtime
                if path == "/api/config":
                    value = r.settings()
                elif path == "/api/tools":
                    value = {"tools": catalog()}
                elif path == "/api/tasks":
                    value = {"tasks": r.recent()}
                elif path == "/api/events":
                    query = parse_qs(parsed.query)
                    after = int(query.get("after", ["0"])[0])
                    if after < 0:
                        raise WorkError("INVALID_SEQUENCE", "after 必须非负")
                    value = r.events(query.get("task_id", [""])[0], after)
                else:
                    raise WorkError("NOT_FOUND", "接口不存在")
                self.send(200, value)
            else:
                files = {"/": ("index.html", "text/html; charset=utf-8"),
                         "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                         "/style.css": ("style.css", "text/css; charset=utf-8")}
                if path not in files:
                    self.send(404, {"error": "Not found"})
                    return
                name, mime = files[path]
                self.send(200, (STATIC / name).read_bytes(), mime)
        except WorkError as exc:
            self.send(403 if exc.code in ("UNAUTHORIZED", "INVALID_ORIGIN", "INVALID_HOST") else 400, {"error": exc.to_dict()})
        except (ValueError, OSError) as exc:
            self.send(400, {"error": {"code": "BAD_REQUEST", "message": str(exc)}})

    def do_POST(self):
        try:
            # Drain a bounded request body even on auth rejection. Closing a Windows
            # socket with unread bytes can reset it before the error reaches clients.
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 2 * 1024 * 1024:
                raise WorkError("BODY_LIMIT", "请求体必须在 1 字节至 2 MiB 之间")
            self.connection.settimeout(15)
            raw = self.rfile.read(length)
            self.authenticate(mutation=True)
            if self.headers.get_content_type() != "application/json":
                raise WorkError("CONTENT_TYPE", "必须提交 application/json")
            body = json.loads(raw.decode("utf-8"))
            r, path = self.server.runtime, urlsplit(self.path).path
            if path == "/api/config":
                value = r.configure(body)
            elif path == "/api/context":
                strict(body, {"text", "extract"})
                if not isinstance(body.get("extract", True), bool):
                    raise WorkError("INVALID_TYPE", "extract 必须是布尔值")
                value = r.context(body.get("text", ""), body.get("extract", True))
            elif path == "/api/preview":
                strict(body, {"plan", "selections"}, {"plan"})
                value = r.preview(body["plan"], body.get("selections"))
            elif path == "/api/execute":
                strict(body, {"preview_id", "digest"}, {"preview_id", "digest"})
                value = r.execute(body["preview_id"], body["digest"])
            elif path == "/api/cancel":
                strict(body, {"task_id"}, {"task_id"})
                value = r.cancel(body["task_id"])
            else:
                raise WorkError("NOT_FOUND", "接口不存在")
            self.send(200, value)
        except WorkError as exc:
            self.send(403 if exc.code in ("UNAUTHORIZED", "INVALID_ORIGIN", "INVALID_HOST") else 400, {"error": exc.to_dict()})
        except (ValueError, TypeError, KeyError, OSError) as exc:
            self.send(400, {"error": {"code": "BAD_REQUEST", "message": str(exc)}})
        except Exception as exc:
            self.send(500, {"error": {"code": "SERVER_ERROR", "message": str(exc)}})


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Dave Work 本地 Agent Harness")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    try:
        server = Server(args.port)
    except OSError:
        if args.port != 8765:
            raise
        server = Server(0)
    url = server.origin + "/#token=" + server.token
    print("Dave Work：" + url, flush=True)
    print("filter / translator 按配置在首次请求加载，模型计划需要预览后点击执行。终端使用当前用户权限。", flush=True)

    def stop(*_):
        server.runtime.close()
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever(poll_interval=0.2)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
