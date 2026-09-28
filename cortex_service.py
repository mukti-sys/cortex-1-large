"""
Laya Localhost HTTP Daemon (Ollama-style Local AI Server).
Runs ModernBERT-Large permanently in RTX 5050 VRAM on http://127.0.0.1:11435.
Provides instant ~35ms JSON endpoints for any IDE, browser, script, or agent.
"""

import sys
import json
import time
from pathlib import Path
from aiohttp import web

root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from laya_mcp.server import RealLayaDecisionServer

# Global persistent engine instance in GPU VRAM
ENGINE = None


async def handle_health(request):
    return web.json_response({
        "status": "online",
        "service": "laya-decision-engine",
        "model": "ModernBERT-large-421M-marker-pooling",
        "device": str(ENGINE.device),
        "target_port": 11435,
        "mode": "resident_gpu_vram"
    })


async def handle_gating(request):
    data = await request.json()
    plan = data.get("plan") or data.get("prompt") or ""
    if not plan:
        return web.json_response({"error": "Missing 'plan' field"}, status=400)

    res = ENGINE.should_autopilot(plan)
    return web.json_response(res)


async def handle_security(request):
    data = await request.json()
    code = data.get("code") or data.get("snippet") or ""
    file_path = data.get("file_path", "")
    if not code:
        return web.json_response({"error": "Missing 'code' field"}, status=400)

    res = ENGINE.triage_security(code, file_path=file_path)
    return web.json_response(res)


async def handle_aiml(request):
    data = await request.json()
    trace = data.get("trace") or data.get("error") or ""
    if not trace:
        return web.json_response({"error": "Missing 'trace' field"}, status=400)

    res = ENGINE.triage_aiml_error(trace)
    return web.json_response(res)


async def handle_diagnose(request):
    data = await request.json()
    trace = data.get("trace") or data.get("error") or ""
    if not trace:
        return web.json_response({"error": "Missing 'trace' field"}, status=400)

    res = ENGINE.diagnose_root_cause(trace)
    return web.json_response(res)


async def handle_risk(request):
    data = await request.json()
    diff = data.get("diff") or data.get("code") or ""
    if not diff:
        return web.json_response({"error": "Missing 'diff' field"}, status=400)

    res = ENGINE.risk_score(diff)
    return web.json_response(res)


def create_app():
    global ENGINE
    print("\n" + "=" * 65)
    print("STARTING LAYA LOCALHOST DAEMON (OLLAMA-STYLE HTTP SERVICE)")
    print("=" * 65)
    ENGINE = RealLayaDecisionServer()

    app = web.Application()
    app.router.add_get("/", handle_health)
    app.router.add_get("/health", handle_health)
    app.router.add_get("/v1/status", handle_health)
    app.router.add_post("/v1/gating", handle_gating)
    app.router.add_post("/v1/security", handle_security)
    app.router.add_post("/v1/aiml", handle_aiml)
    app.router.add_post("/v1/diagnose", handle_diagnose)
    app.router.add_post("/v1/risk", handle_risk)
    return app


if __name__ == "__main__":
    port = 11435
    print(f"\n[HTTP SERVER] Serving Laya System 1 on http://127.0.0.1:{port}")
    print(f"  * Gating:   POST http://127.0.0.1:{port}/v1/gating   {{\"plan\": \"...\"}}")
    print(f"  * Security: POST http://127.0.0.1:{port}/v1/security {{\"code\": \"...\"}}")
    print(f"  * AI/ML:    POST http://127.0.0.1:{port}/v1/aiml     {{\"trace\": \"...\"}}")
    print(f"  * Diagnose: POST http://127.0.0.1:{port}/v1/diagnose {{\"trace\": \"...\"}}\n")
    app = create_app()
    web.run_app(app, host="127.0.0.1", port=port)
