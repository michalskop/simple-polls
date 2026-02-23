# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A Python-based election poll simulation system. For each election, Monte Carlo simulations are run to estimate probability distributions of outcomes (ranking probabilities, vote share intervals, head-to-head duels). Data is read from and written back to Google Sheets. GitHub Actions runs the simulations automatically.

## Running simulations locally

```bash
# Run simulations for a specific election
python pt-2026/simulations_pt-2026.py

# Run the 2nd-round calculator
python pt-2026-2/calculator.py

# Update Polymarket prices
python pt-2026/add_polymarket.py
```

**Credentials:** `gspread.service_account()` looks for credentials at `~/.config/gspread/service_account.json`. The actual credential files live in `secret/credentials.json` and `secret/secrets.json` — pass the path explicitly if not using the default location: `gspread.service_account("secret/credentials.json")`.

For Polymarket scripts, set env vars `POLYMARKET_PRIVATE_KEY` and `POLYMARKET_PROXY_ADDRESS`, or create a `.env` file.

## Dependencies

```bash
pip install -r {election_dir}/requirements_simulations.txt
```

## Creating a new election

1. Edit the parameters at the top of `create_new.py` (candidates, colors, values, sheet key, election date, etc.)
2. Run `python create_new.py` — this scaffolds the GSheet, creates the workflow YAML, requirements, and simulation script for the new election
3. For multi-race or EU-style elections, use `create_new_multi.py` instead

## Architecture

### Per-election directory structure

Each election (e.g. `pt-2026/`, `cl-2025/`, `ar-2025/`) contains:
- `simulations_{code}.py` — main simulation script
- `requirements_simulations.txt` — pip dependencies
- `add_polymarket.py` (optional) — writes Polymarket order book prices back to Google Sheets
- `*.csv` — Polymarket token ID mappings

`pt-2026-2/` is an example of a **2nd-round / runoff calculator** (`calculator.py`) for head-to-head races, as opposed to the standard multi-party simulation.

`archive/` contains finished past elections.

### Simulation logic (simulations_*.py)

1. **Data source:** Reads the `preference` worksheet from a Google Sheet (columns: `party`, `gain`, `date`, `volatilita`, `needed`)
2. **Error model:** Each simulation run adds both a normal error and a uniform error scaled by `volatilita` and sample size (`sample_n = 1000`)
3. **Aging coefficient:** `aging_coeff(today, election_day)` — reduces error as election day approaches, using a power function on days remaining. Applied to create `*_aging` variants
4. **Covariance variant:** A separate set of simulations (`simulations_cov`, `simulations_aging_cov`) uses a multivariate normal distribution seeded from the `median correlations` worksheet, then adds independent uniform noise
5. **Outputs computed:** ranking probabilities, cumulative vote-share probabilities, head-to-head duel matrices, top-2 joint probabilities, victory margin probabilities, parliament entry counts
6. **Write back:** All results go directly to named worksheets in the same Google Sheet

### GitHub Actions workflows

Workflows in `.github/workflows/` are named `run-simulations-{code}.yml`. They:
- Install Python 3.10 and dependencies
- Populate gspread credentials from the `SERVICE_ACCOUNT_JSON` GitHub secret
- Run the simulation script, then optionally `add_polymarket.py`
- Commit and push any changed files back to `main`

Workflows are triggered via `workflow_dispatch` (manual). Scheduled cron runs are commented out and can be re-enabled per election.

### Google Sheets layout

The source sheet always has a `preference` worksheet as input. Output goes to worksheets with Czech names: `pořadí_aktuální_aging_cov`, `pravděpodobnosti_aktuální_aging_cov`, `duely_aging_cov`, `top_2_cov`, `number_in_aging_cov`, `history`, etc. The `_aging_cov` variants are the primary outputs used in practice.
