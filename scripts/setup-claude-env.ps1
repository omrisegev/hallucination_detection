# setup-claude-env.ps1
# Run once on any new machine after git clone.
# Sets up the ~/.claude skills, agents, and settings used in this project.
#
# Usage (from project root):
#   powershell -ExecutionPolicy Bypass -File scripts\setup-claude-env.ps1

$CLAUDE_DIR  = "$env:USERPROFILE\.claude"
$SKILLS_DIR  = "$CLAUDE_DIR\skills"
$AGENTS_DIR  = "$CLAUDE_DIR\agents"
$SETTINGS    = "$CLAUDE_DIR\settings.json"
$STATUSLINE  = "$CLAUDE_DIR\statusline.py"

Write-Host "`n=== MV_EPR Claude Code Environment Setup ===" -ForegroundColor Cyan

# ── 1. deep-research-skills ──────────────────────────────────────────────────
Write-Host "`n[1/3] Installing deep-research-skills..." -ForegroundColor Yellow

$TEMP_DRS = "$env:TEMP\deep-research-skills-install"
if (Test-Path $TEMP_DRS) { Remove-Item $TEMP_DRS -Recurse -Force }

git clone --depth 1 https://github.com/Weizhena/deep-research-skills.git $TEMP_DRS
if ($LASTEXITCODE -ne 0) { Write-Error "git clone failed"; exit 1 }

New-Item -ItemType Directory -Force -Path $SKILLS_DIR | Out-Null
New-Item -ItemType Directory -Force -Path $AGENTS_DIR | Out-Null
New-Item -ItemType Directory -Force -Path "$AGENTS_DIR\web-search-modules" | Out-Null

Copy-Item "$TEMP_DRS\skills\research-en\*" $SKILLS_DIR -Recurse -Force
Copy-Item "$TEMP_DRS\agents\web-search-agent.md" $AGENTS_DIR -Force
Copy-Item "$TEMP_DRS\agents\web-search-modules\*" "$AGENTS_DIR\web-search-modules\" -Force

Remove-Item $TEMP_DRS -Recurse -Force
Write-Host "  Skills installed: research, research-deep, research-add-items, research-add-fields, research-report" -ForegroundColor Green
Write-Host "  Agent installed:  web-search-agent + modules (academic-papers, general-web, github-debug, stackoverflow)" -ForegroundColor Green

# ── 2. pyyaml (required by research-deep validate_json.py) ───────────────────
Write-Host "`n[2/3] Checking pyyaml..." -ForegroundColor Yellow
$yaml_ok = python -c "import yaml; print('ok')" 2>$null
if ($yaml_ok -eq "ok") {
    Write-Host "  pyyaml already installed." -ForegroundColor Green
} else {
    pip install pyyaml -q
    Write-Host "  pyyaml installed." -ForegroundColor Green
}

# ── 3. Claude settings + statusline ──────────────────────────────────────────
Write-Host "`n[3/3] Setting up Claude settings and statusline..." -ForegroundColor Yellow

# Copy statusline script
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
Copy-Item "$SCRIPT_DIR\statusline.py" $STATUSLINE -Force
Write-Host "  statusline.py copied to $STATUSLINE" -ForegroundColor Green

# Write settings.json (only if it doesn't exist, to avoid overwriting user customisations)
if (-not (Test-Path $SETTINGS)) {
    $settings_content = @{
        autoUpdatesChannel = "latest"
        statusLine = @{
            type    = "command"
            command = "python $($STATUSLINE.Replace('\', '/'))"
        }
    } | ConvertTo-Json -Depth 5
    Set-Content -Path $SETTINGS -Value $settings_content -Encoding utf8
    Write-Host "  settings.json written." -ForegroundColor Green
} else {
    Write-Host "  settings.json already exists — skipped (edit manually if needed)." -ForegroundColor Yellow
    Write-Host "  To enable the statusline, add to $SETTINGS :" -ForegroundColor Yellow
    Write-Host '    "statusLine": { "type": "command", "command": "python ~/.claude/statusline.py" }' -ForegroundColor Gray
}

# ── Done ─────────────────────────────────────────────────────────────────────
Write-Host "`n=== Setup complete ===" -ForegroundColor Cyan
Write-Host "Next steps:"
Write-Host "  1. Restart Claude Code for skills to load"
Write-Host "  2. Open this project: cd to the MV_EPR folder"
Write-Host "  3. Claude will auto-load CLAUDE.md — full context restored"
Write-Host "  4. Run /research to start a new research query`n"
