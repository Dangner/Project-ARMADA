#!/usr/bin/env python3
"""
ARMADA-Immune: a small immune-system-inspired SIEM + SOAR demo (stdlib only).

  python armada_demo.py server            # SIEM + SOAR brain + dashboard (laptop)
  python armada_demo.py agent             # laptop endpoint: watches ./test_zone
  python armada_demo.py attack ransom     # SAFE simulated ransomware burst + C2 beacon
  python armada_demo.py attack known      # drops a file whose hash is on the blocklist
  python armada_demo.py phone             # scans Samsung phone over adb (real telemetry)
  python armada_demo.py phone --demo      # no adb: simulated risky app
  python armada_demo.py phone --enforce   # also disable apps judged MALWARE
  python armada_demo.py phone --watch     # re-scan every 20 s (phone stays 'online')
  (open http://LAPTOP-IP:8000/?phone=1 on ANY phone/laptop browser to register it as a device)

Immune mapping:  Skin = signature/hash/extension | Innate = behaviour rules
                 Network = domain reputation     | Memory = learned fingerprints
                 Response = SOAR playbook (alert / quarantine / block / isolate)
Risk = 0.25*S + 0.35*B + 0.25*N + 0.15*A   (SAFE <30, SUSPICIOUS 30-60, MALWARE >=60)
Events from the same entity within 30 s are correlated (SIEM) before scoring.
"""
import sys, os, re, csv, io, json, time, math, shutil, hashlib, threading, subprocess, socket, platform, getpass
import urllib.request, urllib.parse
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = 8000
SERVER = os.environ.get("ARMADA_SERVER", f"http://127.0.0.1:{PORT}")
HOST = socket.gethostname()
MEM_FILE = "armada_memory.json"
WINDOW = 30
KNOWN_BAD = b"ARMADA-DEMO-KNOWN-BAD-SAMPLE"
BLOCK = {hashlib.sha256(KNOWN_BAD).hexdigest()}
BAD_DOMAINS = {"c2.evil-demo.test"}
BAD_EXT = {".locked", ".encrypted", ".crypt"}
DANG = ["SEND_SMS", "READ_SMS", "RECEIVE_SMS", "READ_CONTACTS", "READ_CALL_LOG",
        "RECORD_AUDIO", "SYSTEM_ALERT_WINDOW", "REQUEST_INSTALL_PACKAGES"]
TRUSTED_STORES = ("com.android.vending", "com.sec.android.app.samsungapps")

LOCK = threading.Lock()
EVENTS, ACTIONS, RECENT, QUAR = [], [], {}, set()
DEVICES = {}
SUSP_PROCS = {"mimikatz.exe", "nc.exe", "ncat.exe", "psexec.exe", "procdump.exe", "meterpreter", "netcat"}
LAYERS = Counter()
MEM = {"hash": {}, "sig": {}}


def load_mem():
    global MEM
    if os.path.exists(MEM_FILE):
        MEM = json.load(open(MEM_FILE))


def save_mem():
    json.dump(MEM, open(MEM_FILE, "w"), indent=1)


# ----------------------------------------------------------------- engine
def sig(ev):
    t = ev["type"]
    if t == "file":
        return f"file|{ev.get('ext','')}|ent{int(ev.get('entropy',0) > 7.2)}|burst{int(ev.get('burst',0) >= 15)}"
    if t == "app":
        return f"app|perms{min(len(ev.get('perms', [])), 4)}|side{int(bool(ev.get('sideloaded')))}"
    return f"net|{ev.get('domain')}"


def score(ev):
    S = B = N = A = 0
    why = []
    if ev.get("sha256") in MEM["hash"] or sig(ev) in MEM["sig"]:
        A = 100
        why.append(("Memory", "seen before (hash or behaviour signature) - secondary response"))
    if ev.get("sha256") in BLOCK:
        S = 100
        why.append(("Skin", "hash on blocklist"))
    if ev.get("ext") in BAD_EXT:
        S = max(S, 60)
        why.append(("Skin", f"ransomware-style extension {ev['ext']}"))
    if ev.get("sideloaded"):
        S = max(S, 40)
        why.append(("Skin", "app not from a trusted store"))
    if ev.get("entropy", 0) > 7.2:
        B += 35
        why.append(("Innate", f"high entropy {ev['entropy']:.2f} (encrypted/packed)"))
    if ev.get("burst", 0) >= 15:
        B += 50
        why.append(("Innate", f"{ev['burst']} new files in 5 s (mass-write)"))
    if ev.get("perms"):
        B += min(100, 25 * len(ev["perms"]))
        why.append(("Innate", "dangerous permissions: " + ",".join(ev["perms"])))
    if ev.get("domain") in BAD_DOMAINS:
        N = 100
        why.append(("Network", f"contact with bad domain {ev['domain']}"))
    return S, min(B, 100), N, A, why


STG = {}
TRACE = {"n": 0, "last": None, "alert": None}
SRC = {"file": "fs", "app": "proc", "net": "net"}
OUT = {"SAFE": "safe", "SUSPICIOUS": "susp", "MALWARE": "mal"}


def build_trace(ev, why, S, B, N, A, nh, risk, verdict, acts, learned):
    by = lambda L: [b for a, b in why if a == L]
    t = ev["type"]
    if t == "file":
        d = f"{ev.get('name')} | entropy {ev.get('entropy', 0):.2f} | {ev.get('burst', 0)} new files/5s"
    elif t == "app":
        d = f"{ev.get('name')} | {len(ev.get('perms', []))} dangerous permission(s)"
    else:
        d = f"DNS lookup {ev.get('domain')}"
    steps = [{"s": SRC[t], "st": "hit" if by("Network") else "info", "n": d}]
    sk, inn, mem = by("Skin"), by("Innate"), by("Memory")
    steps.append({"s": "sig", "st": "hit" if sk else "pass", "n": "; ".join(sk) or "no known pattern"})
    steps.append({"s": "beh", "st": "hit" if inn else "pass", "n": "; ".join(inn) or "behaviour normal"})
    steps.append({"s": "prof", "st": "info", "n": f"profile S={S:.0f} B={B:.0f} N={N:.0f} A={A:.0f} from {nh} correlated event(s)"})
    steps.append({"s": "ml", "st": "off", "n": "DANN / CDAN / TTT run in the offline research pipeline, not in this live demo"})
    steps.append({"s": "adv", "st": "off", "n": "FGSM / PGD adversarial shield is not active in the live demo"})
    steps.append({"s": "dec", "st": "info", "n": f"risk {risk:.1f} -> {verdict}"})
    default = {"SAFE": "allow execution", "SUSPICIOUS": "sandbox execute", "MALWARE": "quarantine + alert"}[verdict]
    steps.append({"s": OUT[verdict], "st": "hit", "n": "; ".join(x for x in acts if not x.startswith("MEMORY")) or default})
    if mem:
        steps.append({"s": "mem", "st": "hit", "n": "recall: " + mem[0]})
    elif learned:
        steps.append({"s": "mem", "st": "hit", "n": f"learned {learned} new fingerprint(s)"})
    else:
        steps.append({"s": "mem", "st": "pass", "n": "nothing new to store"})
    return steps


def process(ev):
    now = time.time()
    corr = ev.get("corr") or ev.get("host", "?")
    S, B, N, A, why = score(ev)
    with LOCK:
        hist = [h for h in RECENT.get(corr, []) if now - h["t"] < WINDOW]
        hist.append({"t": now, "S": S, "B": B, "N": N, "A": A, "ev": ev})
        RECENT[corr] = hist
        S, B, N, A = (max(h[k] for h in hist) for k in "SBNA")
        risk = 0.25 * S + 0.35 * B + 0.25 * N + 0.15 * A
        if S == 100:
            risk = 100
        if A == 100:
            risk = max(risk, 80)
        verdict = "MALWARE" if risk >= 60 else "SUSPICIOUS" if risk >= 30 else "SAFE"
        acts, quarantine, learned = [], [], 0
        if verdict == "SUSPICIOUS":
            acts.append("ALERT analyst + send to sandbox")
        if verdict == "MALWARE":
            for h in hist:
                p = h["ev"].get("path")
                if p and p not in QUAR:
                    QUAR.add(p)
                    quarantine.append(p)
            if quarantine:
                acts.append(f"QUARANTINE {len(quarantine)} file(s)")
            dom = next((h["ev"]["domain"] for h in hist if h["ev"].get("domain") in BAD_DOMAINS), None)
            if dom:
                acts.append(f"BLOCK domain {dom} at firewall (simulated)")
            app = next((h["ev"]["name"] for h in hist if h["ev"]["type"] == "app"), None)
            if app:
                acts.append(f"ISOLATE app {app} (disable on phone)")
            for h in hist:
                e = h["ev"]
                if e.get("sha256") and e["sha256"] not in MEM["hash"]:
                    MEM["hash"][e["sha256"]] = {"name": e["name"], "t": now}
                    learned += 1
                if sig(e) not in MEM["sig"]:
                    MEM["sig"][sig(e)] = {"t": now}
                    learned += 1
            if learned:
                acts.append(f"MEMORY learned {learned} new fingerprint(s)")
                save_mem()
        for layer, _ in why:
            LAYERS[layer] += 1
        LAYERS["Response"] += len(acts)
        rec = {"t": time.strftime("%H:%M:%S"), "host": ev.get("host"), "name": ev.get("name"),
               "S": S, "B": B, "N": N, "A": A, "risk": round(risk, 1), "verdict": verdict,
               "why": [f"{a}: {b}" for a, b in why], "actions": acts}
        EVENTS.append(rec)
        ACTIONS.extend(f"{rec['t']}  [{ev.get('host')}] {a}" for a in acts)
        steps = build_trace(ev, why, S, B, N, A, len(hist), risk, verdict, acts, learned)
        for x in steps:
            if x["st"] != "off":
                d = STG.setdefault(x["s"], {"seen": 0, "hit": 0})
                d["seen"] += 1
                d["hit"] += (x["st"] in ("hit", "info")) if x["s"] in ("prof", "dec") else (x["st"] == "hit")
        TRACE["n"] += 1
        tr = {"id": TRACE["n"], "t": rec["t"], "name": ev.get("name"), "host": ev.get("host"),
              "verdict": verdict, "risk": rec["risk"], "steps": steps}
        TRACE["last"] = tr
        if verdict != "SAFE":
            TRACE["alert"] = tr
    return {"verdict": verdict, "risk": rec["risk"], "actions": acts, "quarantine": quarantine, "trace": tr}


# ----------------------------------------------------------------- server
class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, body, ctype="application/json"):
        b = body.encode() if isinstance(body, str) else body
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        if u.path == "/api/state":
            now = time.time()
            with LOCK:
                devs = []
                for d in DEVICES.values():
                    evs = [e for e in EVENTS if e["host"] == d["id"]]
                    worst = "MALWARE" if any(e["verdict"] == "MALWARE" for e in evs) else \
                        "SUSPICIOUS" if any(e["verdict"] == "SUSPICIOUS" for e in evs) else "SAFE"
                    devs.append({"id": d["id"], "name": d.get("name", d["id"]), "kind": d.get("kind", "device"),
                                 "source": d.get("source", ""), "ip": d.get("ip", ""),
                                 "age": round(now - d["last_seen"]), "online": now - d["last_seen"] < 45,
                                 "worst": worst, "events": len(evs),
                                 "alerts": sum(e["verdict"] != "SAFE" for e in evs)})
                s = {"events": EVENTS[-40:][::-1], "actions": ACTIONS[-25:][::-1], "layers": dict(LAYERS),
                     "memory": len(MEM["hash"]) + len(MEM["sig"]), "total": len(EVENTS),
                     "malware": sum(e["verdict"] == "MALWARE" for e in EVENTS), "devices": devs,
                     "verdicts": {k: sum(e["verdict"] == k for e in EVENTS) for k in ("SAFE", "SUSPICIOUS", "MALWARE")},
                     "stages": STG, "trace": TRACE["alert"] or TRACE["last"]}
            self.send(json.dumps(s))
        elif u.path == "/api/device":
            did = urllib.parse.parse_qs(u.query).get("id", [""])[0]
            with LOCK:
                d = dict(DEVICES.get(did) or {"error": "unknown"})
                d["age"] = round(time.time() - d.get("last_seen", 0))
                d["events"] = [e for e in EVENTS if e["host"] == did][-60:][::-1]
            self.send(json.dumps(d))
        elif u.path == "/immunity":
            here = os.path.dirname(os.path.abspath(__file__))
            f = os.path.join(here, "armada_adaptive_immunity.html")
            if not os.path.exists(f):
                return self.send("Put armada_adaptive_immunity.html in the same folder as armada_demo.py", "text/plain")
            page = open(f, encoding="utf-8").read()
            if os.path.exists(os.path.join(here, "p5.min.js")):   # optional offline copy of p5.js
                page = re.sub(r'https://cdnjs[^"]*p5\.min\.js', "/p5.min.js", page)
            self.send(page, "text/html; charset=utf-8")
        elif u.path == "/p5.min.js":
            f = os.path.join(os.path.dirname(os.path.abspath(__file__)), "p5.min.js")
            self.send(open(f, "rb").read() if os.path.exists(f) else b"", "application/javascript")
        elif u.path == "/api/reset":
            with LOCK:
                EVENTS.clear(); ACTIONS.clear(); RECENT.clear(); QUAR.clear(); LAYERS.clear(); STG.clear()
                TRACE["last"] = TRACE["alert"] = None
                MEM["hash"].clear(); MEM["sig"].clear(); save_mem()
            self.send("{}")
        else:
            self.send(DASH, "text/html; charset=utf-8")

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/api/device":
            now = time.time()
            body["ip"] = body.get("ip") or self.client_address[0]
            with LOCK:
                old = DEVICES.get(body["id"], {})
                new = {**old, **body}
                new["first_seen"] = old.get("first_seen", now)
                new["last_seen"] = now
                DEVICES[body["id"]] = new
            self.send("{}")
        else:
            self.send(json.dumps(process(body)))


def server():
    load_mem()
    ip = "localhost"
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.connect(("8.8.8.8", 80)); ip = s.getsockname()[0]
    except Exception:
        pass
    print(f"Dashboard: http://localhost:{PORT}   |  on phone (same Wi-Fi): http://{ip}:{PORT}/?phone=1")
    ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()


# ------------------------------------------------------------------ agent
def post(ev, path="/api/event"):
    req = urllib.request.Request(SERVER + path, json.dumps(ev).encode(),
                                 {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=10))


def entropy(b):
    if not b:
        return 0.0
    n = len(b)
    return -sum(v / n * math.log2(v / n) for v in Counter(b).values())


def run(*a, t=25):
    try:
        return subprocess.run(list(a), capture_output=True, text=True, errors="replace", timeout=t).stdout
    except Exception:
        return ""


def local_ip():
    try:
        x = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        x.connect(("8.8.8.8", 80))
        return x.getsockname()[0]
    except Exception:
        return ""


def laptop_inventory():
    info = {"Hostname": HOST, "User": getpass.getuser(), "OS": platform.platform(), "Architecture": platform.machine(),
            "Processor": platform.processor() or "n/a", "CPU cores": os.cpu_count(), "Local IP": local_ip(),
            "Python": platform.python_version()}
    try:
        du = shutil.disk_usage(os.path.splitdrive(os.getcwd())[0] + os.sep)
        info["Disk"] = f"{du.used / 2**30:.0f} / {du.total / 2**30:.0f} GB used"
    except Exception:
        pass
    procs, conns, startup = [], [], []
    try:
        import psutil
        vm = psutil.virtual_memory()
        info["RAM"] = f"{vm.used / 2**30:.1f} / {vm.total / 2**30:.1f} GB ({vm.percent}%)"
        info["CPU load"] = f"{psutil.cpu_percent(0.3)}%"
        info["Boot time"] = time.strftime("%Y-%m-%d %H:%M", time.localtime(psutil.boot_time()))
        bt = psutil.sensors_battery()
        if bt:
            info["Battery"] = f"{bt.percent:.0f}% {'charging' if bt.power_plugged else 'on battery'}"
        names = {}
        for p in psutil.process_iter(["pid", "name", "username", "memory_info"]):
            i = p.info
            names[i["pid"]] = i["name"]
            procs.append({"name": i["name"], "pid": i["pid"], "user": i["username"] or "",
                          "mem_mb": round((i["memory_info"].rss if i["memory_info"] else 0) / 2**20, 1)})
        for c in psutil.net_connections("inet"):
            if c.status in ("ESTABLISHED", "LISTEN"):
                conns.append({"proto": "TCP" if c.type == socket.SOCK_STREAM else "UDP",
                              "local": f"{c.laddr.ip}:{c.laddr.port}",
                              "remote": f"{c.raddr.ip}:{c.raddr.port}" if c.raddr else "",
                              "state": c.status, "proc": names.get(c.pid, c.pid)})
    except ImportError:
        names = {}
        if os.name == "nt":
            import ctypes

            class MS(ctypes.Structure):
                _fields_ = [("l", ctypes.c_ulong), ("load", ctypes.c_ulong), ("tp", ctypes.c_ulonglong),
                            ("ap", ctypes.c_ulonglong), ("a", ctypes.c_ulonglong), ("b", ctypes.c_ulonglong),
                            ("c", ctypes.c_ulonglong), ("d", ctypes.c_ulonglong), ("e", ctypes.c_ulonglong)]
            m = MS(); m.l = ctypes.sizeof(m)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            info["RAM"] = f"{m.tp / 2**30:.1f} GB total, {m.load}% in use"
            for row in csv.reader(io.StringIO(run("tasklist", "/fo", "csv", "/nh"))):
                if len(row) >= 5:
                    try:
                        mem = float(re.sub(r"[^\d]", "", row[4]) or 0) / 1024
                    except ValueError:
                        mem = 0
                    names[int(row[1])] = row[0]
                    procs.append({"name": row[0], "pid": int(row[1]), "user": row[2], "mem_mb": round(mem, 1)})
            for ln in run("netstat", "-ano", "-p", "tcp").splitlines():
                p = ln.split()
                if len(p) >= 5 and p[0] == "TCP" and p[3] in ("ESTABLISHED", "LISTENING"):
                    conns.append({"proto": "TCP", "local": p[1], "remote": "" if p[3] == "LISTENING" else p[2],
                                  "state": p[3], "proc": names.get(int(p[4]), p[4])})
            for hive in ("HKCU", "HKLM"):
                for ln in run("reg", "query", hive + r"\Software\Microsoft\Windows\CurrentVersion\Run").splitlines():
                    m2 = re.match(r"\s+(\S.*?)\s+REG_\w+\s+(.*)", ln)
                    if m2:
                        startup.append({"hive": hive, "name": m2.group(1), "command": m2.group(2)})
        else:
            for ln in run("ps", "-eo", "pid,user,rss,comm", "--no-headers").splitlines():
                p = ln.split(None, 3)
                if len(p) == 4:
                    procs.append({"name": p[3], "pid": int(p[0]), "user": p[1], "mem_mb": round(int(p[2]) / 1024, 1)})
            for ln in run("ss", "-tn").splitlines()[1:]:
                p = ln.split()
                if len(p) >= 5:
                    conns.append({"proto": "TCP", "local": p[3], "remote": p[4], "state": p[0], "proc": ""})
    for p in procs:
        p["flag"] = p["name"].lower() in SUSP_PROCS
    procs.sort(key=lambda p: -p["mem_mb"])
    q = len(os.listdir("quarantine")) if os.path.isdir("quarantine") else 0
    info["ARMADA quarantined files"] = q
    info["Suspicious process names"] = ", ".join(p["name"] for p in procs if p["flag"]) or "none"
    return {"id": HOST, "name": HOST, "kind": "laptop", "source": "ARMADA agent", "info": info,
            "procs": procs[:300], "conns": conns[:300], "startup": startup}


def heartbeat():
    while True:
        try:
            post(laptop_inventory(), "/api/device")
        except Exception as e:
            print("heartbeat error:", e)
        time.sleep(15)


def agent(zone="test_zone"):
    os.makedirs(zone, exist_ok=True)
    os.makedirs("quarantine", exist_ok=True)
    seen, times = set(os.listdir(zone)), []
    threading.Thread(target=heartbeat, daemon=True).start()
    print(f"Agent watching ./{zone}  (host={HOST})  -> {SERVER}")
    while True:
        for f in sorted(set(os.listdir(zone)) - seen):
            seen.add(f)
            p = os.path.join(zone, f)
            try:
                data = open(p, "rb").read(65536)
            except OSError:
                continue
            now = time.time()
            times = [x for x in times if now - x < 5] + [now]
            r = post({"type": "file", "host": HOST, "corr": HOST, "name": f, "path": os.path.abspath(p),
                      "ext": os.path.splitext(f)[1].lower(), "sha256": hashlib.sha256(data).hexdigest(),
                      "entropy": entropy(data), "burst": len(times)})
            for q in r["quarantine"]:
                if os.path.exists(q):
                    shutil.move(q, os.path.join("quarantine", os.path.basename(q) + ".quarantined"))
                    print("  QUARANTINED", os.path.basename(q))
        time.sleep(0.4)


def attack(mode):
    zone = "test_zone"
    os.makedirs(zone, exist_ok=True)
    tag = os.urandom(3).hex()
    if mode == "known":
        open(os.path.join(zone, f"invoice_{tag}.exe"), "wb").write(KNOWN_BAD)
        return print("dropped known-bad sample")
    for i in range(25):  # harmless random bytes; polymorphic: different every run
        open(os.path.join(zone, f"report_{tag}_{i}.docx.locked"), "wb").write(os.urandom(4096))
        time.sleep(0.05)
    time.sleep(1.5)
    post({"type": "net", "host": HOST, "corr": HOST, "name": "DNS beacon (SIMULATED)", "domain": "c2.evil-demo.test"})
    print("simulated ransomware burst + C2 beacon sent")


# ------------------------------------------------------------------ phone
def adb(*a):
    return run("adb", *a, t=40)


def phone_inventory():
    P = lambda k: adb("shell", "getprop", k).strip()
    bat = adb("shell", "dumpsys", "battery")
    g = lambda pat, txt: (re.search(pat, txt) or [None, ""])[1]
    serial = adb("get-serialno").strip()
    model = P("ro.product.model")
    info = {"Manufacturer": P("ro.product.manufacturer"), "Model": model, "Android": P("ro.build.version.release"),
            "SDK": P("ro.build.version.sdk"), "Security patch": P("ro.build.version.security_patch"),
            "Build": P("ro.build.display.id"), "Serial": serial,
            "Battery": g(r"level: (\d+)", bat) + "%", "Battery temp": str(int(g(r"temperature: (\d+)", bat) or 0) / 10) + " C",
            "Wi-Fi IP": g(r"inet (\d+\.\d+\.\d+\.\d+)", adb("shell", "ip", "-f", "inet", "addr", "show", "wlan0")),
            "USB debugging": "ON" if adb("shell", "settings", "get", "global", "adb_enabled").strip() == "1" else "off",
            "Rooted (su found)": "YES" if adb("shell", "which", "su").strip() else "no"}
    df = adb("shell", "df", "/data").strip().splitlines()
    if len(df) > 1:
        x = df[-1].split()
        try:
            info["Storage /data"] = f"{int(x[2]) / 2**20:.0f} / {int(x[1]) / 2**20:.0f} GB used"
        except (ValueError, IndexError):
            pass
    procs = []
    for ln in adb("shell", "ps", "-A").splitlines()[1:]:
        p = ln.split()
        if len(p) >= 9:
            procs.append({"name": p[-1], "pid": p[1], "user": p[0], "mem_mb": round(int(p[4]) / 1024, 1) if p[4].isdigit() else 0})
    conns = []
    states = {"01": "ESTABLISHED", "0A": "LISTEN"}
    for ln in adb("shell", "cat", "/proc/net/tcp").splitlines()[1:]:
        p = ln.split()
        if len(p) > 3 and p[3] in states:
            def hx(v):
                ip, port = v.split(":")
                return ".".join(str(int(ip[i:i + 2], 16)) for i in (6, 4, 2, 0)) + ":" + str(int(port, 16))
            conns.append({"proto": "TCP", "local": hx(p[1]), "remote": "" if p[3] == "0A" else hx(p[2]), "state": states[p[3]], "proc": ""})
    return model, serial, {"info": info, "procs": procs[:300], "conns": conns[:300]}


def phone(enforce=False, demo=False, watch=False):
    while True:
        if demo:
            devid, name = "phone-DEMO", "Demo phone (SIMULATED)"
            inv = {"info": {"Model": "SM-DEMO (simulated)", "Android": "13", "Note": "simulated data - run without --demo for the real phone"},
                   "procs": [], "conns": []}
            apps = [("com.demo.flashlight", "1.0", "com.android.vending", [], False),
                    ("com.demo.freegamebooster", "2.3", "null", ["SEND_SMS", "READ_SMS", "READ_CONTACTS", "SYSTEM_ALERT_WINDOW"], True)]
        else:
            if "\tdevice" not in adb("devices"):
                return print("No authorised phone found. Plug in USB, enable USB debugging, accept the prompt on the phone, check `adb devices`.")
            model, serial, inv = phone_inventory()
            devid, name = f"phone-{model}-{serial[-4:]}", f"{model} (adb)"
            apps = []
            for line in adb("shell", "pm", "list", "packages", "-3").splitlines():
                if not line.startswith("package:"):
                    continue
                pk = line[8:].strip()
                d = adb("shell", "dumpsys", "package", pk)
                perms = [x for x in DANG if f"android.permission.{x}: granted=true" in d]
                m = re.search(r"installerPackageName=(\S+)", d)
                inst = m.group(1) if m else "null"
                v = re.search(r"versionName=(\S+)", d)
                apps.append((pk, v.group(1) if v else "", inst, perms, inst not in TRUSTED_STORES))
        rows = []
        for pk, ver, inst, perms, side in apps:
            r = post({"type": "app", "host": devid, "corr": pk, "name": pk, "perms": perms, "sideloaded": side})
            if demo and pk.endswith("freegamebooster"):
                r = post({"type": "net", "host": devid, "corr": pk, "name": pk + " beacon (SIMULATED)", "domain": "c2.evil-demo.test"})
            rows.append({"name": pk, "version": ver, "installer": inst, "perms": perms, "sideloaded": side,
                         "verdict": r["verdict"], "risk": r["risk"]})
            print(f"{pk:45s} {r['verdict']:10s} risk={r['risk']}")
            if enforce and r["verdict"] == "MALWARE" and not demo:
                adb("shell", "pm", "disable-user", "--user", "0", pk)
                print("   -> app disabled")
        post({"id": devid, "name": name, "kind": "phone", "source": "adb" if not demo else "simulated",
              "apps": rows, **inv}, "/api/device")
        if not watch:
            return
        time.sleep(20)


# -------------------------------------------------------------- dashboard
DASH = r"""<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>ARMADA-Immune</title>
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600&family=Lora:wght@400;500&display=swap" rel="stylesheet">
<style>
:root{--dark:#141413;--light:#faf9f5;--mid:#b0aea5;--lgray:#e8e6dc;--orange:#d97757;--blue:#6a9bcc;--green:#788c5d;--red:#c0392b;--amber:#f39c12}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Poppins',system-ui,sans-serif;background:linear-gradient(135deg,var(--light) 0%,#f5f3ee 100%);min-height:100vh;color:var(--dark);font-size:13px}
.container{display:flex;min-height:100vh;padding:16px;gap:16px}
.sidebar{width:262px;flex-shrink:0;background:rgba(255,255,255,.95);padding:20px;border-radius:12px;box-shadow:0 10px 30px rgba(20,20,19,.1);align-self:flex-start;position:sticky;top:16px;max-height:calc(100vh - 32px);overflow-y:auto}
.main{flex:1;min-width:0}
@media(max-width:900px){.container{flex-direction:column}.sidebar{width:auto;position:static;max-height:none;align-self:stretch}}
h1{font-family:'Lora',serif;font-size:18px;font-weight:500;margin-bottom:4px}
.subtitle{color:var(--mid);font-size:12px;margin-bottom:20px;line-height:1.4}
.cs{margin-bottom:20px}.cs h3,.card h3{font-size:13px;font-weight:600;margin-bottom:10px;display:flex;align-items:center;gap:6px}
.cs h3::before,.card h3::before{content:'\2022';color:var(--orange);font-weight:bold}
button{background:var(--dark);color:var(--light);border:none;padding:8px 12px;border-radius:6px;font-family:inherit;font-size:12px;cursor:pointer;transition:all .2s}
button:hover{background:#2a2a28}button.sec{background:var(--lgray);color:var(--dark)}button.sec:hover{background:#d4d2c8}
button.on{background:var(--orange);color:#fff}
.nav{display:grid;gap:6px}.nav button{text-align:left}
.stat{background:var(--light);border-radius:8px;padding:10px;font-family:'Courier New',monospace;font-size:11px;line-height:1.8;margin-top:6px}
.lrow{display:flex;justify-content:space-between;align-items:center;font-size:12px;padding:3px 0}.lrow b{font-family:'Courier New',monospace}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:7px;flex-shrink:0}.dot.y{background:var(--green)}.dot.n{background:var(--mid)}
.card{background:rgba(255,255,255,.95);border-radius:12px;padding:16px;box-shadow:0 10px 30px rgba(20,20,19,.08);margin-bottom:16px;min-width:0}
.two{display:grid;grid-template-columns:2fr 1fr;gap:16px}@media(max-width:1100px){.two{grid-template-columns:1fr}}
.w{color:#8d8b82;font-size:12px}.a{color:#a8553a;font-size:12px}
.SAFE{color:var(--green)}.SUSPICIOUS{color:#c77d0a}.MALWARE{color:var(--red)}
#devs{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:10px}
.dv{background:var(--light);border-left:5px solid var(--mid);border-radius:8px;padding:10px;cursor:pointer;line-height:1.5}
.dv.sel{outline:2px solid var(--orange)}.dv.SAFE{border-color:var(--green)}.dv.SUSPICIOUS{border-color:var(--amber)}.dv.MALWARE{border-color:var(--red)}
.tb{background:var(--lgray);color:var(--dark);margin:0 4px 8px 0}.tb:hover{background:#d4d2c8}.tb.on{background:var(--orange);color:#fff}
table{width:100%;border-collapse:collapse;font-size:12.5px}th{text-align:left;color:#8d8b82;font-weight:500;padding:5px 6px;position:sticky;top:0;background:#fff}
td{padding:5px 6px;border-top:1px solid var(--lgray);word-break:break-word;vertical-align:top}tr.flag td{background:#fbe9e7}
input{background:var(--light);border:1px solid var(--lgray);color:var(--dark);border-radius:8px;padding:7px 10px;font-family:inherit;font-size:12px;width:220px}input:focus{outline:none;border-color:var(--orange)}
.e{border-left:4px solid var(--mid);padding:7px 10px;margin:7px 0;background:var(--light);border-radius:0 8px 8px 0}
.e.SAFE{border-color:var(--green)}.e.SUSPICIOUS{border-color:var(--amber)}.e.MALWARE{border-color:var(--red)}.v{font-weight:600}
.scroll{max-height:52vh;overflow:auto}
/* pipeline */
.cap{font-size:10px;letter-spacing:.08em;text-transform:uppercase;color:#8d8b82;margin-bottom:2px}
.legend{display:flex;flex-wrap:wrap;gap:14px;margin:4px 0 14px;font-size:11px;color:#6f6d64}.legend i{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:5px}
.prow{display:flex;align-items:stretch;gap:8px;flex-wrap:wrap;margin-bottom:12px}
.col{display:flex;flex-direction:column;justify-content:center;gap:6px;flex:1;min-width:150px}
.arrow{align-self:center;color:var(--mid);font-size:20px}
.node{position:relative;background:#fff;border:1.5px solid var(--lgray);border-left:5px solid var(--k);border-radius:10px;padding:9px 10px;cursor:pointer;transition:all .25s}
.node:hover{box-shadow:0 4px 14px rgba(20,20,19,.12)}.node b{font-size:12px;display:block;padding-right:34px}.node small{color:#7a786f;font-size:11px;line-height:1.3;display:block}
.node .bd{position:absolute;right:8px;top:7px;font:10px 'Courier New',monospace;color:#8d8b82}
.node.lit{background:#fff3ed;border-color:var(--orange);box-shadow:0 0 0 3px rgba(217,119,87,.25)}
.node.sel{outline:2px solid var(--dark)}.node.off{border-style:dashed;opacity:.82}
.chips{display:flex;gap:4px;margin-top:6px;flex-wrap:wrap}.chip{background:var(--light);border-radius:5px;padding:3px 6px;font-size:10px;line-height:1.3}.chip span{color:#8d8b82;display:block}
.badge{display:inline-block;color:#fff;border-radius:10px;padding:1px 8px;font-size:10px;margin-left:6px;vertical-align:middle}
.three{display:grid;grid-template-columns:1.2fr 1fr 1.4fr;gap:16px}@media(max-width:1100px){.three{grid-template-columns:1fr}}
.ps{display:grid;grid-template-columns:repeat(auto-fit,minmax(78px,1fr));gap:8px}
.pc{background:var(--light);border-radius:8px;padding:8px;border-top:3px solid var(--k)}.pc b{font:600 20px 'Courier New',monospace;display:block}.pc small{color:#8d8b82;font-size:10px}
.ts{padding:4px 0;border-top:1px solid var(--lgray);font-size:12px;line-height:1.45}.ts:first-child{border-top:0}
#insp p{margin-bottom:6px;line-height:1.5}
</style>
<div class=container>
<div class=sidebar>
 <h1>ARMADA-Immune</h1>
 <div class=subtitle>Immune-system-inspired malware defense &mdash; SIEM + SOAR</div>
 <div class=cs><h3>Views</h3><div class=nav>
  <button id=vsoc class=on>SOC view &middot; devices</button>
  <button id=vpipe class=sec>System pipeline</button>
  <button id=vimm class=sec>Adaptive immunity (live)</button></div></div>
 <div class=cs><h3>Immune layers</h3><div id=lay></div></div>
 <div class=cs><h3>Status</h3><div class=stat id=side>connecting...</div></div>
 <div class=cs><h3>Actions</h3><div class=nav><button id=arm class=sec>Arm phone alerts</button><button id=rst class=sec>Reset demo</button></div></div>
 <div class=cs><h3>Legend</h3>
  <div class=lrow><span><span class=dot style="background:var(--green)"></span>Safe</span></div>
  <div class=lrow><span><span class=dot style="background:var(--amber)"></span>Suspicious</span></div>
  <div class=lrow><span><span class=dot style="background:var(--red)"></span>Malware</span></div></div>
</div>
<div class=main>
<div id=soc>
 <div class=card><h3>Connected devices</h3><div id=devs></div></div>
 <div class=card><h3 id=dtitle>Select a device</h3><div id=tabs></div>
  <input id=flt placeholder="filter..." style="margin:0 0 10px"><div id=dbody class=scroll></div></div>
 <div class=two>
  <div class=card><h3>SIEM event stream (correlated)</h3><div id=ev class=scroll></div></div>
  <div class=card><h3>SOAR playbook actions</h3><div id=ac class="a scroll"></div></div>
 </div>
</div>
<div id=pipe style="display:none">
 <div class=card>
  <h3>System pipeline</h3>
  <div class=legend>
   <span><i style="background:#788c5d"></i>barrier</span><span><i style="background:#6a9bcc"></i>innate</span>
   <span><i style="background:#d97757"></i>&#9733; novel bridge</span><span><i style="background:#9b7bb8"></i>adaptive ML</span>
   <span><i style="background:#c0392b"></i>defense</span><span><i style="background:#3f9d9a"></i>memory</span>
   <span><i style="background:#788c5d"></i>safe outcome</span></div>
  <div style="margin-bottom:12px"><span class=w>Run a sample through the live engine:</span>
   <button class=sec onclick="runSample('benign')">Benign file</button>
   <button class=sec onclick="runSample('app')">Suspicious app</button>
   <button class=sec onclick="runSample('ransom')">Ransomware + C2</button>
   <button class=sec onclick="runSample('known')">Known-bad hash</button>
   <span class=w>&nbsp;(run Ransomware twice to see Memory recall)</span></div>
  <div class=prow id=p1></div><div class=prow id=p2></div>
  <div class=cap>Threat memory</div><div id=pm></div>
 </div>
 <div class=card><h3>Live model trace</h3>
  <div class=three>
   <div><div class=cap>Selected stage</div><div id=insp></div></div>
   <div><div class=cap>Pipeline state</div><div class=ps id=ps></div></div>
   <div><div class=cap>Execution trace</div><div id=trh class=w style="margin-bottom:4px"></div><div id=trace class=scroll style="max-height:40vh"></div></div>
  </div>
 </div>
</div>
<div id=imm style="display:none"><iframe id=ifr style="width:100%;height:calc(100vh - 32px);border:0;border-radius:12px"></iframe></div>
</div></div>
<script>
const $=id=>document.getElementById(id);
const esc=s=>String(s==null?"":s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const sleep=ms=>new Promise(r=>setTimeout(r,ms));
const LAY=[["Skin","signature","#788c5d"],["Innate","behaviour","#6a9bcc"],["Network","reputation","#d9a441"],["Memory","learned","#3f9d9a"],["Response","SOAR","#d97757"]];
const KC={src:"#8d8b82",barrier:"#788c5d",innate:"#6a9bcc",novel:"#d97757",adaptive:"#9b7bb8",defense:"#c0392b",decision:"#141413",memory:"#3f9d9a",safe:"#788c5d",susp:"#f39c12",mal:"#c0392b"};
const ST={
 fs:{t:"File system",d:"entropy, hash, size",k:"src",mode:"live",about:"Collects static file features for every new file: SHA-256, size and byte entropy.",demo:"Real: the agent watches the test_zone folder on the laptop."},
 proc:{t:"Process",d:"API calls, injection",k:"src",mode:"partial",about:"Process behaviour: API calls, child processes, injection attempts.",demo:"Partial: on the phone, installed apps and their dangerous permissions are scored (via adb). Process lists are shown in the device inventory but not scored."},
 net:{t:"Network",d:"DNS, IP rep, traffic",k:"src",mode:"partial",about:"DNS requests, IP/domain reputation and traffic volume.",demo:"Partial: domains are checked against a small bad-domain list. The C2 beacon in the attack simulator is simulated. Live connection lists are shown in the inventory."},
 reg:{t:"Registry",d:"startup, persistence",k:"src",mode:"partial",about:"Registry modification, startup persistence, service creation.",demo:"Partial: Windows Run keys are read into the laptop inventory (Startup tab) but not scored yet."},
 sig:{t:"Signature filter",d:"Known-pattern block · skin barrier",k:"barrier",mode:"live",about:"The skin barrier: blocks what is already known before deeper analysis.",demo:"Real: SHA-256 blocklist, ransomware-style extensions, apps from untrusted sources. A known-bad hash goes straight to MALWARE."},
 beh:{t:"Behavioral engine",d:"Class anomaly · innate immunity",k:"innate",mode:"live",about:"Innate immunity: generic anomaly rules that need no prior knowledge of the sample.",demo:"Real: entropy above 7.2, 15+ new files in 5 s, dangerous permission combinations."},
 prof:{t:"Threat profiler ★ novel",d:"Structures signal · dendritic cell",k:"novel",mode:"live",about:"The dendritic-cell bridge: turns raw behavioural signals into one labelled profile before the decision.",demo:"Implemented here as correlation: events for the same host or app within 30 s are merged into one S / B / N / A profile."},
 ml:{t:"ML engine · adaptive immunity",d:"",k:"adaptive",mode:"off",chips:[["DANN / GRL","Marginal align"],["CDAN","Conditional align"],["TTT","Concept drift adapt"]],about:"Grouped self-attention encoder over EMBER 2018 features with dual-discriminator domain adaptation (DANN/GRL marginal, CDAN conditional) and test-time training for concept drift.",demo:"Offline: this is the research pipeline (armada_train_eval.py). It is not executed inside the live demo. Your 20K-sample run gave accuracy ~88.8%, F1 ~0.887, AUC ~0.953, with a negligible TTT gain at that scale."},
 adv:{t:"Adversarial shield",d:"FGSM · PGD · evasion defense",k:"defense",mode:"off",about:"Adversarial training and evasion detection against FGSM and PGD style perturbations.",demo:"Offline: not active in the live demo. In the live risk formula the A term is filled by Memory recall."},
 dec:{t:"Decision engine",d:"Weighted risk score · three thresholds",k:"decision",mode:"live",about:"Risk = 0.25 S + 0.35 B + 0.25 N + 0.15 A. Below 30 SAFE, 30 to 60 SUSPICIOUS, 60 and above MALWARE.",demo:"Real. A known-bad hash forces risk 100, and a memory hit forces at least 80."},
 safe:{t:"Safe",d:"Allow execution",k:"safe",mode:"live",about:"Risk below 30: execution is allowed.",demo:"Real: the event is logged and nothing else happens."},
 susp:{t:"Suspicious",d:"Sandbox execute",k:"susp",mode:"live",about:"Risk 30 to 60: send to a sandbox and alert the analyst.",demo:"The alert and playbook step are real; there is no sandbox VM in this demo."},
 mal:{t:"Malware",d:"Quarantine + alert",k:"mal",mode:"live",about:"Risk 60 and above: quarantine, block and alert.",demo:"Real quarantine of files on the laptop agent, optional app disable on the phone. Domain blocking is simulated."},
 mem:{t:"Threat memory DB",d:"Stores fingerprints · memory cells",k:"memory",mode:"live",about:"Memory cells: fingerprints of confirmed malware so the next sighting gets a faster, stronger response.",demo:"Real: file hashes and behaviour signatures are stored in armada_memory.json. A changed variant with the same behaviour is still recalled."}
};
const nodeH=id=>{const n=ST[id];return `<div class="node ${n.mode=="off"?"off":""}" id="n-${id}" style="--k:${KC[n.k]}" onclick="pickStage('${id}')"><span class=bd id="bd-${id}"></span><b>${n.t}</b><small>${n.d}</small>${n.chips?`<div class=chips>${n.chips.map(c=>`<div class=chip>${c[0]}<span>${c[1]}</span></div>`).join("")}</div>`:""}</div>`};
const col=(h,cap)=>`<div class=col>${cap?`<div class=cap>${cap}</div>`:""}${h}</div>`;
const arr='<div class=arrow>&rarr;</div>';
$("p1").innerHTML=col(["fs","proc","net","reg"].map(nodeH).join(""),"Telemetry sources")+arr+col(nodeH("sig"))+arr+col(nodeH("beh"))+arr+col(nodeH("prof"));
$("p2").innerHTML=col(nodeH("ml"))+arr+col(nodeH("adv"))+arr+col(nodeH("dec"))+arr+col(["safe","susp","mal"].map(nodeH).join(""),"Outcome");
$("pm").innerHTML=nodeH("mem");
let sel=null,tab="Overview",armed=false,lastMal=0,filt="",D=null,selStage=null,TR=null,lastTr=null,playing=false,busy=false,STAGES={};
function show(v){["soc","pipe","imm"].forEach(k=>{$(k).style.display=k==v?"":"none";$("v"+k).className=k==v?"on":"sec"});if(v=="imm"&&!$("ifr").getAttribute("src"))$("ifr").src="/immunity"}
$("vsoc").onclick=()=>show("soc");$("vpipe").onclick=()=>show("pipe");$("vimm").onclick=()=>show("imm");
$("arm").onclick=()=>{armed=true;navigator.vibrate&&navigator.vibrate(100);$("arm").textContent="Phone alerts armed"};
$("rst").onclick=()=>fetch("/api/reset");
$("flt").oninput=e=>{filt=e.target.value.toLowerCase();renderDetail()};
const ago=a=>a<60?a+"s ago":a<3600?Math.round(a/60)+"m ago":Math.round(a/3600)+"h ago";
const ICON={phone:"&#128241;",laptop:"&#128187;"};
function pick(id){sel=id;tab="Overview";D=null;dtick()}
function setTab(t){tab=t;renderDetail()}
function tbl(h,rows){return "<table><tr>"+h.map(x=>"<th>"+x+"</th>").join("")+"</tr>"+rows.join("")+"</table>"}
function renderDetail(){
 const d=D;if(!d||d.error){$("dbody").innerHTML="";$("tabs").innerHTML="";return}
 const T=[["Overview",0]];
 if(d.procs&&d.procs.length)T.push(["Processes",d.procs.length]);
 if(d.conns&&d.conns.length)T.push(["Network",d.conns.length]);
 if(d.apps&&d.apps.length)T.push(["Apps",d.apps.length]);
 if(d.startup&&d.startup.length)T.push(["Startup",d.startup.length]);
 T.push(["Alerts",(d.events||[]).length]);
 if(!T.some(x=>x[0]==tab))tab="Overview";
 $("tabs").innerHTML=T.map(([t,n])=>`<button class="tb ${t==tab?"on":""}" onclick="setTab('${t}')">${t}${n?" ("+n+")":""}</button>`).join("");
 $("dtitle").innerHTML=(ICON[d.kind]||"&#128421;")+" "+esc(d.name)+` <span class=w>${esc(d.source||"")} &middot; ${esc(d.ip||"")} &middot; seen ${ago(d.age)}</span>`;
 const f=x=>!filt||JSON.stringify(x).toLowerCase().includes(filt);let h="";
 if(tab=="Overview"){const i=d.info||{};h=tbl(["Property","Value"],Object.keys(i).map(k=>`<tr><td class=w>${esc(k)}</td><td>${esc(Array.isArray(i[k])?i[k].join(", "):typeof i[k]=="object"?JSON.stringify(i[k]):i[k])}</td></tr>`))}
 if(tab=="Processes")h=tbl(["Name","PID","User","Mem MB",""],d.procs.filter(f).map(p=>`<tr class="${p.flag?"flag":""}"><td>${esc(p.name)}</td><td>${esc(p.pid)}</td><td>${esc(p.user)}</td><td>${p.mem_mb}</td><td>${p.flag?"&#9888; suspicious":""}</td></tr>`));
 if(tab=="Network")h=tbl(["Proto","Local","Remote","State","Process"],d.conns.filter(f).map(c=>`<tr><td>${c.proto}</td><td>${esc(c.local)}</td><td>${esc(c.remote)}</td><td>${esc(c.state)}</td><td>${esc(c.proc)}</td></tr>`));
 if(tab=="Apps")h=tbl(["App","Version","Installer","Dangerous permissions","Verdict"],d.apps.slice().sort((a,b)=>b.risk-a.risk).filter(f).map(a=>`<tr><td>${esc(a.name)}</td><td>${esc(a.version)}</td><td>${esc(a.installer)}${a.sideloaded?" &#9888;":""}</td><td>${esc((a.perms||[]).join(", "))}</td><td class=${a.verdict}>${a.verdict} (${a.risk})</td></tr>`));
 if(tab=="Startup")h=tbl(["Hive","Name","Command"],d.startup.filter(f).map(s=>`<tr><td>${esc(s.hive)}</td><td>${esc(s.name)}</td><td>${esc(s.command)}</td></tr>`));
 if(tab=="Alerts")h=(d.events||[]).filter(f).map(evHtml).join("")||"<span class=w>no events for this device</span>";
 $("dbody").innerHTML=h}
const evHtml=e=>`<div class="e ${e.verdict}"><span class="v ${e.verdict}">${e.verdict}</span> risk ${e.risk} &middot; ${esc(e.host)} &middot; ${esc(e.name)} <span class=w>${e.t}<br>S=${e.S} B=${e.B} N=${e.N} A=${e.A}<br>${e.why.map(esc).join("<br>")}</span>${e.actions.map(a=>`<div class=a>&#9654; ${esc(a)}</div>`).join("")}</div>`;
/* ---- pipeline ---- */
const IC={hit:["&#9679;","#d97757"],pass:["&#9675;","#788c5d"],off:["&#9676;","#8d8b82"],info:["&middot;","#6a9bcc"]};
const stepH=x=>`<div class=ts><span style="color:${IC[x.st][1]}">${IC[x.st][0]}</span> <b style="color:${KC[ST[x.s].k]}">${ST[x.s].t.replace(" ★ novel","")}</b> <span class=w>${esc(x.n)}</span></div>`;
function markSel(){document.querySelectorAll(".node").forEach(n=>n.classList.toggle("sel",n.id=="n-"+selStage))}
function pickStage(id){selStage=id;markSel();renderInsp()}
function renderInsp(){
 const n=ST[selStage];
 if(!n){$("insp").innerHTML="<span class=w>Run the pipeline or click any node to inspect it.</span>";return}
 const c=STAGES[selStage]||{hit:0,seen:0},st=TR&&TR.steps.find(x=>x.s==selStage);
 const M={live:["LIVE","#788c5d"],partial:["PARTIAL","#c77d0a"],off:["OFFLINE in demo","#8d8b82"]}[n.mode];
 $("insp").innerHTML=`<div><b style="color:${KC[n.k]}">${n.t}</b><span class=badge style="background:${M[1]}">${M[0]}</span></div><div class=w style="margin:3px 0 8px">${n.d}</div><p>${n.about}</p><p class=w><b>In this demo:</b> ${n.demo}</p>`+(n.mode=="off"?"":`<div class=stat>triggered ${c.hit} / inspected ${c.seen}</div>`)+(st?`<div class=stat>last trace: ${esc(st.n)}</div>`:"")}
function renderState(){
 const g=id=>(STAGES[id]||{hit:0}).hit;
 $("ps").innerHTML=[["barrier",g("sig"),"#788c5d",""],["innate",g("beh"),"#6a9bcc",""],["adaptive",0,"#9b7bb8","offline"],["defense",0,"#c0392b","offline"],["memory",g("mem"),"#3f9d9a",""]]
  .map(([n,v,c,t])=>`<div class=pc style="--k:${c}"><small>${n}</small><b>${v}</b><small>${t}</small></div>`).join("");
 for(const id in ST){const b=$("bd-"+id);if(!b)continue;if(ST[id].mode=="off")b.textContent="offline";else{const c=STAGES[id];b.textContent=c?c.hit+"/"+c.seen:"0/0"}}
}
function traceHead(tr){$("trh").innerHTML=`${tr.t} &middot; ${esc(tr.name)} &rarr; <b class=${tr.verdict}>${tr.verdict}</b> (risk ${tr.risk})`}
function showTrace(tr){TR=tr;traceHead(tr);$("trace").innerHTML=tr.steps.map(stepH).join("");renderInsp()}
async function play(list,delay){
 playing=true;
 for(const tr of list){
  document.querySelectorAll(".node.lit").forEach(n=>n.classList.remove("lit"));
  TR=tr;traceHead(tr);$("trace").innerHTML="";
  for(const x of tr.steps){const n=$("n-"+x.s);n&&n.classList.add("lit");$("trace").insertAdjacentHTML("beforeend",stepH(x));selStage=x.s;markSel();renderInsp();await sleep(delay)}
  await sleep(600)}
 setTimeout(()=>document.querySelectorAll(".node.lit").forEach(n=>n.classList.remove("lit")),6000);
 playing=false}
const rid=()=>Math.random().toString(16).slice(2).padEnd(16,"0");
const sha=()=>rid()+rid()+rid()+rid();
async function ev(o){const r=await fetch("/api/event",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(o)});return r.json()}
async function runSample(k){
 if(playing||busy)return;busy=true;
 try{const c="pt-"+rid().slice(0,6),H="pipeline-test",rs=[];
  if(k=="benign")rs.push(await ev({type:"file",host:H,corr:c,name:"quarterly_report.docx",ext:".docx",sha256:sha(),entropy:4.3,burst:1}));
  if(k=="app")rs.push(await ev({type:"app",host:H,corr:c,name:"com.test.flashlight-pro (TEST)",perms:["SEND_SMS","READ_CONTACTS","SYSTEM_ALERT_WINDOW"],sideloaded:true}));
  if(k=="ransom"){for(let i=0;i<3;i++)rs.push(await ev({type:"file",host:H,corr:c,name:"report_"+i+".docx.locked",ext:".locked",sha256:sha(),entropy:7.95,burst:20}));
   rs.push(await ev({type:"net",host:H,corr:c,name:"DNS beacon (SIMULATED)",domain:"c2.evil-demo.test"}))}
  if(k=="known")rs.push(await ev({type:"file",host:H,corr:c,name:"invoice.exe",ext:".exe",sha256:"__KNOWN__",entropy:5.1,burst:1}));
  const t=rs.map(r=>r.trace);lastTr=Math.max(lastTr||0,...t.map(x=>x.id));
  await play(t.length>1?[t[0],t[t.length-1]]:t,320)}
 finally{busy=false}}
/* ---- polling ---- */
async function tick(){try{const s=await (await fetch("/api/state")).json();
 STAGES=s.stages||{};renderState();
 $("lay").innerHTML=LAY.map(([k,d,c])=>`<div class=lrow><span><span class=dot style="background:${c}"></span>${k} <span class=w>${d}</span></span><b>${s.layers[k]||0}</b></div>`).join("");
 const v=s.verdicts||{},on=s.devices.filter(d=>d.online).length;
 $("side").innerHTML=`Devices online: ${on}/${s.devices.length}<br>Events: ${s.total}<br>Safe: ${v.SAFE||0}<br>Suspicious: ${v.SUSPICIOUS||0}<br>Malware: ${v.MALWARE||0}<br>Memory cells: ${s.memory}`;
 if(!sel&&s.devices.length){sel=s.devices[0].id;dtick()}
 $("devs").innerHTML=s.devices.map(d=>`<div class="dv ${d.worst} ${d.id==sel?"sel":""}" data-id="${esc(d.id)}" onclick="pick(this.dataset.id)"><b>${ICON[d.kind]||"&#128421;"} ${esc(d.name)}</b><br><span class=w><span class="dot ${d.online?"y":"n"}"></span>${d.online?"online":"offline"} &middot; ${esc(d.ip)} &middot; ${esc(d.source)}<br>seen ${ago(d.age)} &middot; <span class=${d.worst}>${d.worst}</span> &middot; ${d.alerts} alerts / ${d.events} events</span></div>`).join("")||"<span class=w>No devices yet. Start the agent, run the phone scan, or open this page on a phone with ?phone=1</span>";
 $("ev").innerHTML=s.events.map(evHtml).join("");$("ac").innerHTML=s.actions.map(esc).join("<br>");
 if(s.malware>lastMal&&armed&&navigator.vibrate)navigator.vibrate([300,100,300]);lastMal=s.malware;
 const tr=s.trace;
 if(tr&&!busy&&!playing){
  if(lastTr===null){lastTr=tr.id;showTrace(tr)}
  else if(tr.id!==lastTr){lastTr=tr.id;if($("pipe").style.display!="none")play([tr],240);else showTrace(tr)}}
 else if(!tr&&lastTr===null){renderInsp()}
 }catch(e){}}
async function dtick(){if(!sel)return;try{D=await (await fetch("/api/device?id="+encodeURIComponent(sel))).json();renderDetail()}catch(e){}}
async function reg(){
 if(!/phone|device/.test(location.search))return;
 let id;try{id=localStorage.getItem("aid");if(!id){id="browser-"+Math.random().toString(36).slice(2,8);localStorage.setItem("aid",id)}}catch(e){id="browser-"+Math.random().toString(36).slice(2,8)}
 const ua=navigator.userAgent,c=navigator.connection||{};let b=null;try{b=await navigator.getBattery()}catch(e){}
 const m=(ua.match(/Android [\d.]+; ([^)]+)\)/)||[])[1];
 const kind=/Android|iPhone|iPad|Mobile/.test(ua)?"phone":"laptop";
 const info={"Device model":m||"n/a","Browser UA":ua,"Platform":navigator.platform,"Screen":screen.width+"x"+screen.height+" @"+(window.devicePixelRatio||1)+"x","Language":navigator.language,"Time zone":Intl.DateTimeFormat().resolvedOptions().timeZone,"CPU cores":navigator.hardwareConcurrency||"n/a","Memory (approx GB)":navigator.deviceMemory||"n/a","Touch points":navigator.maxTouchPoints,"Network type":c.effectiveType||"n/a","Downlink Mbps":c.downlink||"n/a","RTT ms":c.rtt||"n/a","Battery":b?Math.round(b.level*100)+"% "+(b.charging?"charging":"discharging"):"n/a"};
 fetch("/api/device",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({id:id,name:(m||navigator.platform)+" (browser)",kind:kind,source:"browser",info:info})})}
renderState();renderInsp();
setInterval(tick,1000);setInterval(dtick,3000);setInterval(reg,10000);tick();reg();
</script>"""
DASH = DASH.replace("__KNOWN__", sorted(BLOCK)[0])

if __name__ == "__main__":
    c = sys.argv[1] if len(sys.argv) > 1 else "server"
    if c == "server": server()
    elif c == "agent": agent()
    elif c == "attack": attack(sys.argv[2] if len(sys.argv) > 2 else "ransom")
    elif c == "phone": phone("--enforce" in sys.argv, "--demo" in sys.argv, "--watch" in sys.argv)
    else: print(__doc__)
