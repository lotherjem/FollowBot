from __future__ import annotations

import argparse
import hmac
import json
import os
import re
import secrets
import time
import mimetypes
import struct
import sys
import threading
import webbrowser
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from tools.simui.session import SimSession

STATIC = Path(__file__).resolve().parent / "static"


# --------------------------------------------------------------------------- #
# PNG encoding
# --------------------------------------------------------------------------- #


def encode_png(rgb: np.ndarray) -> bytes:
    """Minimal RGB PNG encoder.

    Pillow would do this in one line, but it is not a dependency anywhere else
    in the project and this is thirty lines. Filter type 0 on every row: the
    maps are large flat regions, so zlib gets them down to a few kB regardless
    and a smarter filter would buy nothing.
    """
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    height, width, _ = rgb.shape

    rows = np.zeros((height, width * 3 + 1), dtype=np.uint8)
    rows[:, 1:] = rgb.reshape(height, width * 3)
    compressed = zlib.compress(rows.tobytes(), 6)

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", compressed)
        + chunk(b"IEND", b"")
    )


# --------------------------------------------------------------------------- #
# Login
# --------------------------------------------------------------------------- #


class Auth:
    """A password for the interface, and the sessions it opens.

    Deliberately small: one shared password from ``carrybot.toml``, a random
    session token in an HttpOnly, SameSite=Strict cookie, and a lock-out after
    repeated failures. That keeps other people on the same network from
    driving the robot. It is not transport security -- the password crosses
    the LAN in the clear at login -- which is why the port must never be
    exposed to the internet.
    """

    COOKIE = "carrybot_session"
    MAX_FAILURES = 5

    def __init__(self, password: str = "", idle: float = 12 * 3600.0,
                 clock=time.monotonic):
        self.password = password or ""
        self.idle = idle
        self.clock = clock
        self._sessions: Dict[str, float] = {}
        self._failures: Dict[str, Tuple[int, float]] = {}
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return bool(self.password)

    def login(self, address: str, attempt: str) -> Tuple[Optional[str], str]:
        """``(token, "")`` on success, ``(None, reason)`` otherwise."""
        now = self.clock()
        with self._lock:
            count, locked_until = self._failures.get(address, (0, 0.0))
            if now < locked_until:
                return None, (f"too many wrong passwords; try again in "
                              f"{locked_until - now:.0f} s")
            if hmac.compare_digest(attempt.encode(), self.password.encode()):
                self._failures.pop(address, None)
                token = secrets.token_urlsafe(32)
                self._sessions[token] = now
                return token, ""
            count += 1
            lock = 0.0
            if count >= self.MAX_FAILURES:
                # 30 s, doubling per further failure, capped at 5 minutes.
                lock = min(300.0, 30.0 * 2 ** (count - self.MAX_FAILURES))
            self._failures[address] = (count, now + lock)
            return None, "wrong password" + (f"; locked for {lock:.0f} s" if lock else "")

    def check(self, token: Optional[str]) -> bool:
        if not self.enabled:
            return True
        if not token:
            return False
        now = self.clock()
        with self._lock:
            seen = self._sessions.get(token)
            if seen is None or now - seen > self.idle:
                self._sessions.pop(token, None)
                return False
            self._sessions[token] = now
            return True

    def logout(self, token: Optional[str]) -> None:
        with self._lock:
            self._sessions.pop(token or "", None)


_CLIENT_ID = re.compile(r"^[A-Za-z0-9_-]{8,64}$")
LOOPBACK = {"127.0.0.1", "localhost", "::1"}


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #


class Handler(BaseHTTPRequestHandler):
    session: SimSession = None  # type: ignore[assignment]
    auth: Auth = Auth()
    protocol_version = "HTTP/1.1"

    #: Reachable without logging in: the login page and what it needs.
    PUBLIC = ("/login", "/api/login", "/static/style.css")

    def log_message(self, *args) -> None:  # quiet; the console is for the sim
        pass

    # -- helpers -------------------------------------------------------- #

    def _send(self, status: int, body: bytes, content_type: str,
              cache: str = "no-store") -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache)
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload, status: int = 200, headers=()) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for name, value in headers:
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)

    def _token(self) -> Optional[str]:
        for part in self.headers.get("Cookie", "").split(";"):
            name, _, value = part.strip().partition("=")
            if name == Auth.COOKIE:
                return value
        return None

    def _client(self) -> Optional[str]:
        """The id the page sends to say which open page it is."""
        value = self.headers.get("X-CarryBot-Client", "")
        return value if _CLIENT_ID.match(value) else None

    def _allowed(self, path: str) -> bool:
        """Logged in, or asking for something public. Answers for itself if not."""
        if path in self.PUBLIC or self.auth.check(self._token()):
            return True
        if path.startswith("/api/"):
            self._send_json({"ok": False, "error": "login required", "login": True}, 401)
        else:
            self.send_response(303)
            self.send_header("Location", "/login")
            self.send_header("Content-Length", "0")
            self.end_headers()
        return False

    def _same_origin(self) -> bool:
        """Refuse commands that another website's page tried to send.

        The SameSite cookie already stops a logged-in browser from carrying its
        session to a request another site makes; this also covers the case with
        no password, where there is no cookie to withhold.
        """
        origin = self.headers.get("Origin")
        if not origin:
            return True
        host = self.headers.get("Host", "")
        return origin.split("://", 1)[-1] == host

    # -- GET ------------------------------------------------------------ #

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        try:
            if not self._allowed(path):
                return
            live = getattr(self.session, "live", False)
            if path == "/login":
                self._send_file(STATIC / "login.html")
            elif path in ("/", "/index.html"):
                self._send_file(STATIC / ("live.html" if live else "index.html"))
            elif path.startswith("/static/"):
                name = Path(path).name
                candidate = STATIC / name
                if candidate.is_file() and candidate.parent == STATIC:
                    self._send_file(candidate)
                else:
                    self._send(404, b"not found", "text/plain")
            elif path == "/api/state":
                # Polling is the heartbeat: an interface that stops asking is
                # an interface that has gone, and live mode disarms on that.
                if live:
                    client = self._client()
                    self.session.touch(client)
                    data = self.session.telemetry(client)
                    data["auth"] = self.auth.enabled
                    self._send_json(data)
                else:
                    self._send_json(self.session.telemetry())
            elif path in ("/api/camera.jpg", "/api/depth.jpg"):
                grab = getattr(self.session,
                               "camera_jpeg" if "camera" in path else "depth_jpeg", None)
                image = grab() if grab is not None else None
                if image is None:
                    self._send(204, b"", "image/jpeg")
                else:
                    self._send(200, image, "image/jpeg")
            elif path == "/api/world.png":
                self._send(200, encode_png(self.session.world_rgb()), "image/png")
            elif path == "/api/costmap.png":
                self._send(200, encode_png(self.session.costmap_rgb()), "image/png")
            else:
                self._send(404, b"not found", "text/plain")
        except BrokenPipeError:
            pass
        except Exception as exc:  # never take the server down for one request
            self._send(500, str(exc).encode(), "text/plain")

    def _send_file(self, path: Path) -> None:
        if not path.is_file():
            self._send(404, b"not found", "text/plain")
            return
        kind = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self._send(200, path.read_bytes(), kind)

    # -- POST ----------------------------------------------------------- #

    def do_POST(self) -> None:
        path = self.path.split("?", 1)[0]
        if not self._same_origin():
            self._send_json({"ok": False, "error": "cross-site request refused"}, 403)
            return
        if path == "/api/login":
            self._login()
            return
        if path == "/api/logout":
            self.auth.logout(self._token())
            self._send_json({"ok": True}, headers=[(
                "Set-Cookie", f"{Auth.COOKIE}=; HttpOnly; SameSite=Strict; Path=/; Max-Age=0")])
            return
        if path != "/api/cmd":
            self._send(404, b"not found", "text/plain")
            return
        if not self._allowed(path):
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length) or b"{}")
            action = payload.pop("action", "")
            if getattr(self.session, "live", False):
                self._send_json(self.session.command(action, payload, self._client()))
            else:
                self._send_json(self.session.command(action, payload))
        except BrokenPipeError:
            pass
        except Exception as exc:
            self._send_json({"ok": False, "error": str(exc)})

    def _login(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", 0))
            attempt = str(json.loads(self.rfile.read(min(length, 4096)) or b"{}")
                          .get("password", ""))
        except (ValueError, AttributeError):
            attempt = ""
        if not self.auth.enabled:
            self._send_json({"ok": True})
            return
        token, error = self.auth.login(self.client_address[0], attempt)
        if token is None:
            self._send_json({"ok": False, "error": error}, 401)
            return
        cookie = (f"{Auth.COOKIE}={token}; HttpOnly; SameSite=Strict; Path=/; "
                  f"Max-Age={int(self.auth.idle)}")
        self._send_json({"ok": True}, headers=[("Set-Cookie", cookie)])


# --------------------------------------------------------------------------- #


def _lan_addresses() -> list:
    """This machine's LAN addresses, for printing a URL a phone can open."""
    import socket

    found = []
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("10.255.255.255", 1))      # no packet is sent
            found.append(probe.getsockname()[0])
    except OSError:
        pass
    return [a for a in found if not a.startswith("127.")]


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1",
                        help="0.0.0.0 to reach it from another machine")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true")
    live = parser.add_argument_group("live mode (the real camera, and optionally the robot)")
    live.add_argument("--live", action="store_true",
                      help="use the stereo camera instead of the simulated world")
    live.add_argument("--drive", choices=("dry", "sim", "hardware"), default="dry",
                      help="dry: start with the motors off -- they can be switched "
                           "on from the interface (default); hardware: start with "
                           "them on; sim: simulated motors, no switch")
    live.add_argument("--servo", action="store_true", default=None,
                      help="a pan servo is fitted (hardware drive only); "
                           "or set [live] servo_fitted in carrybot.toml")
    live.add_argument("--device", default=None,
                      help="camera override: a path, an index, or vid:pid")
    live.add_argument("--weights", default=None,
                      help="YOLO pose weights (default: [live] weights in carrybot.toml)")
    live.add_argument("--no-tracker", action="store_true",
                      help="skip YOLO: mapping, click-to-go and manual still work")
    live.add_argument("--no-password", action="store_true",
                      help="allow the interface on the network with no password. "
                           "Anyone who can reach the page can then drive the robot")
    live.add_argument("--config", metavar="PATH", default=None,
                      help="robot config file (default: carrybot.toml)")
    live.add_argument("--replay", metavar="FILE.npz", default=None,
                      help="use a recording instead of the camera (probe --record)")
    args = parser.parse_args(argv)

    if args.live:
        from carrybot_config import ConfigError, load_config
        from tools.simui.live import build_live_session

        try:
            config = load_config(args.config)
        except ConfigError as exc:
            print(f"error: {exc}")
            return 2
        print(config.describe_source())
        for warning in config.warnings:
            print(f"warning: {warning}")
        password = os.environ.get("CARRYBOT_PASSWORD") or config.live.password
        on_network = args.host not in LOOPBACK
        if on_network and not password and not args.no_password:
            print("refusing to open the live interface to the network with no "
                  "password: anyone who can reach the page could drive the robot.\n"
                  "Set [live] password in carrybot.toml (or CARRYBOT_PASSWORD), or "
                  "pass --no-password to accept that.")
            return 2
        Handler.auth = Auth(password)
        if on_network and not password:
            print("WARNING: no password -- anyone on this network can drive the robot")
        try:
            Handler.session = build_live_session(
                drive=args.drive, servo=args.servo, device=args.device,
                weights=args.weights, no_tracker=args.no_tracker,
                replay=args.replay, config=config)
        except Exception as exc:
            print(f"could not start live mode: {exc}")
            return 1
        session = Handler.session
        print(f"camera: {session.camera_label}")
        if session.hardware_active:
            print("motors: ON -- the robot will move when armed")
        elif session.hardware is None:
            print(f"motors: off, and cannot be switched on: {session.hardware_note}")
        else:
            blockers, _ = session.hardware.preflight()
            print("motors: off -- switch them on from the interface"
                  + ("" if not blockers else
                     "\n        (blocked for now: " + " ".join(blockers) + ")"))
        if not session.follow_available:
            print(f"note:   {session.follow_note}")
    else:
        Handler.session = SimSession()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    shown = "localhost" if args.host in ("127.0.0.1", "0.0.0.0") else args.host
    url = f"http://{shown}:{args.port}/"

    print(f"CarryBot {'live' if args.live else 'simulator'} UI on {url}")
    if args.host == "0.0.0.0":
        for address in _lan_addresses():
            print(f"  from another machine: http://{address}:{args.port}/")
    print("Ctrl-C to stop.")
    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        Handler.session.close()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
