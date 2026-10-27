"""Execute actual Python subprocesses; no agent SDK or Docker daemon required."""
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

import pytest

RUNTIME = Path(__file__).resolve().parents[1] / "container" / "script-runtime"


def invoke(tmp_path, source, **context_changes):
    script = tmp_path / "script.py"
    script.write_text(source)
    context = {"companyId": "qa", "executionId": "command-1", "scriptKey": "reviewed",
               "scriptSha256": hashlib.sha256(script.read_bytes()).hexdigest(), "inputFiles": [], "config": {}, **context_changes}
    (tmp_path / "context.json").write_text(json.dumps(context))
    env = {**os.environ, "PYTHONPATH": str(RUNTIME), "DEV_MODE": "REMOTE", "FIREBASE_PROJECT_ID": "qa-runtime-test"}
    process = subprocess.run([sys.executable, str(RUNTIME / "run_script.py"), "--script", "script.py", "--context", "context.json", "--result", "result.json"],
                             cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30)
    return process, json.loads((tmp_path / "result.json").read_text())


def test_runs_exact_script_once_and_discards_raw_sdk_output(tmp_path):
    process, receipt = invoke(tmp_path, '''
from pathlib import Path
def run(context):
    print("counto_session=must-never-be-logged")
    marker = Path("executions.txt")
    marker.write_text(marker.read_text() + "run\\n" if marker.exists() else "run\\n")
    return {"status": "completed", "summary": "verified", "effects": [], "verification": {"passed": True}}
''')
    assert process.returncode == 0
    assert not process.stdout and not process.stderr
    assert (tmp_path / "executions.txt").read_text() == "run\n"
    assert receipt["companyId"] == "qa" and receipt["scriptKey"] == "reviewed"


@pytest.mark.parametrize("source,changes", [
    ('def run(c):\n return {"status":"completed","summary":"no validation","effects":[]}', {}),
    ('def run(c):\n return {"status":"completed","summary":"wrong company","effects":[],"verification":{"passed":True},"companyId":"other"}', {}),
    ('def run(c):\n raise RuntimeError("counto_session=private")', {}),
    ('def run(c):\n raise AssertionError("must not execute")', {"scriptSha256": "0" * 64}),
])
def test_invalid_or_failed_script_never_repairs_or_reports_success(tmp_path, source, changes):
    process, receipt = invoke(tmp_path, source, **changes)
    assert process.returncode == 1
    assert receipt["status"] == "uncertain"
    assert "private" not in json.dumps(receipt) + process.stdout + process.stderr


def test_declared_owner_failure_retains_partial_effects(tmp_path):
    process, receipt = invoke(tmp_path, '''
def run(c):
 return {"status":"failed","summary":"Owner verification denied: HTTP 403","effects":[{"service":"vault","documentId":123}]}
''')
    assert process.returncode == 1
    assert receipt["effects"] == [{"service": "vault", "documentId": 123}]
    assert "403" in receipt["summary"]


def test_script_can_explicitly_extract_then_write_and_verify_using_shared_client(tmp_path):
    # Contract fixture only: this tests the real Counto HTTP client in the script
    # process, not production/dev authentication or real Vault persistence.
    events = []
    class Owner(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_POST(self):
            events.append(self.path)
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path == "/extract":
                assert body == {"text": "USD 100 fee 3"}
                result = {"currency": "USD", "gross": 100, "fee": -3, "net": 97}
            else:
                assert self.path == "/save" and body["net"] == 97
                result = {"documentId": 123}
            encoded = json.dumps(result).encode()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(encoded)
        def do_GET(self):
            events.append(self.path)
            assert self.path == "/documents/123"
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"currency":"USD","net":97}')
    server = ThreadingHTTPServer(("127.0.0.1", 0), Owner)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        process, receipt = invoke(tmp_path, '''
from appdefs import sync_requests
def run(context):
    base = context["config"]["fixtureBase"]
    response = sync_requests.post(base + "/extract", json={"text":"USD 100 fee 3"}, timeout=5)
    response.raise_for_status()
    data = response.json()
    assert data["gross"] + data["fee"] == data["net"]
    response = sync_requests.post(base + "/save", json=data, timeout=5)
    response.raise_for_status()
    document = response.json()["documentId"]
    saved = sync_requests.get(base + "/documents/" + str(document), timeout=5)
    saved.raise_for_status()
    assert saved.json() == {"currency":"USD","net":97}
    return {"status":"completed","summary":"Saved fixture verified","effects":[{"service":"fixture","documentId":document}],"verification":{"passed":True}}
''', config={"fixtureBase": f"http://127.0.0.1:{server.server_port}"})
        assert process.returncode == 0, receipt
        assert receipt["verification"]["passed"] is True
        assert events == ["/extract", "/save", "/documents/123"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)
