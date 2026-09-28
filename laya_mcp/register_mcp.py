"""
Registration utility for Laya Decision Engine MCP Server in Google Antigravity.
Reads ~/.gemini/config/mcp_config.json, adds 'laya-brain', and saves backup.
"""

import os
import sys
import json
import shutil
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


def register_laya_mcp(dry_run: bool = False):
    config_path = Path(os.path.expanduser(r"~\.gemini\config\mcp_config.json"))
    workspace_dir = Path(os.path.abspath(".")).resolve()

    if not config_path.exists():
        print(f"[ERROR] Antigravity MCP config not found at: {config_path}")
        return

    # Backup existing config
    backup_path = config_path.with_suffix(".json.bak")
    if not dry_run and not backup_path.exists():
        shutil.copy2(config_path, backup_path)
        print(f"[INFO] Created config backup at: {backup_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "mcpServers" not in data:
        data["mcpServers"] = {}

    laya_server_config = {
        "command": "python",
        "args": [
            "-m",
            "laya_mcp.server"
        ],
        "cwd": str(workspace_dir),
        "env": {
            "PYTHONPATH": str(workspace_dir),
            "PYTHONIOENCODING": "utf-8"
        }
    }

    data["mcpServers"]["laya-brain"] = laya_server_config

    print(f"\n{'='*60}")
    print("REGISTERING LAYA MCP SERVER IN GOOGLE ANTIGRAVITY")
    print(f"{'='*60}")
    print(f"Target Config:      {config_path}")
    print(f"Server Name:        laya-brain")
    print(f"Command:            python -m laya_mcp.server")
    print(f"Working Directory:  {workspace_dir}")
    print(f"{'='*60}\n")

    if not dry_run:
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print("[SUCCESS] 'laya-brain' successfully registered in Antigravity MCP config!")
    else:
        print("[DRY-RUN] Config preview:")
        print(json.dumps(laya_server_config, indent=2))


if __name__ == "__main__":
    is_dry = "--dry-run" in sys.argv
    register_laya_mcp(dry_run=is_dry)
