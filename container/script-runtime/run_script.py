"""Execute one reviewed script. No agent SDK, model loop or code repair."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import runpy


def execute(script: Path, context_path: Path, result_path: Path) -> int:
    context = json.loads(context_path.read_text(encoding="utf-8"))
    identity = {key: context[key] for key in ("companyId", "executionId", "scriptKey", "scriptSha256")}
    try:
        if script.is_symlink() or script.suffix != ".py" or hashlib.sha256(script.read_bytes()).hexdigest() != identity["scriptSha256"]:
            raise ValueError("Script checksum mismatch")
        # The shared client currently prints session cookies. Do not persist or
        # forward raw stdout/stderr, including output from imported modules.
        with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            namespace = runpy.run_path(str(script), run_name="approved_script")
            run = namespace.get("run")
            if not callable(run):
                raise ValueError("Approved script must export run(context)")
            receipt = run(context)
        if not isinstance(receipt, dict) or receipt.get("status") not in {"completed", "failed", "uncertain"}:
            raise ValueError("Script must return a business receipt")
        if not isinstance(receipt.get("summary"), str) or not receipt["summary"].strip():
            raise ValueError("Receipt summary is required")
        if not isinstance(receipt.get("effects"), list):
            raise ValueError("Receipt effects must be a list")
        if receipt["status"] == "completed":
            verification = receipt.get("verification")
            if not isinstance(verification, dict) or verification.get("passed") is not True:
                raise ValueError("Completed receipt requires verification.passed=true")
        for key, value in identity.items():
            if key in receipt and receipt[key] != value:
                raise ValueError("Receipt identity mismatch")
        receipt = {**receipt, **identity}
        # Validate serialization before replacing the result. A script may have
        # failed after a write; an exception never authorizes automatic replay.
        encoded = json.dumps(receipt, allow_nan=False)
        if len(encoded.encode("utf-8")) > 1024 * 1024:
            raise ValueError("Receipt exceeds maximum size")
    except Exception as exc:
        # Raw exception text can contain credentials from HTTP/SDK failures.
        # Scripts should return sanitized owner errors and partial effects in
        # their structured receipt when they can establish the outcome.
        receipt = {**identity, "status": "uncertain", "summary": f"Approved script raised {type(exc).__name__}; reconcile owner effects before retrying", "effects": []}
        encoded = json.dumps(receipt)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = result_path.with_suffix(".tmp")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(result_path)
    return 0 if receipt["status"] == "completed" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    root = Path.cwd().resolve()
    for path in (args.script, args.context, args.result):
        if not path.resolve().is_relative_to(root):
            parser.error("Execution paths must stay within the mounted workspace")
    return execute(args.script, args.context, args.result)


if __name__ == "__main__":
    raise SystemExit(main())
