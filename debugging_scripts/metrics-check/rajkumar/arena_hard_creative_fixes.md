## arena_hard_creative — Fixes Applied (Attempt 1, 2026-04-17)

### Fix 1: Systemic — `scenarios_new.` → `scenarios.` in all run_specs

- **Root cause**: The `scenarios_new/` directory was renamed to `scenarios/` in commit
  `76c09932`, but all 313 run_spec files still referenced `scenarios_new.<module>`.
  HELM could not import any scenario class, resulting in empty stats.json across all datasets.
- **Files changed**: All `run_specs/*.py` (313 files, bulk sed replacement)
- **Change summary**: `sed -i '' 's/scenarios_new\./scenarios./g' run_specs/*.py`
- **Verification**: `grep -rl "scenarios_new\." run_specs/ --include="*.py" | wc -l` → 0

### Fix 2: Scenario — JSONL parsing bug in arena_hard_creative_scenario.py

- **Root cause**: The JSONL file (question.jsonl) contains one record (line 102) where
  the `prompt` field has an unescaped literal newline, splitting it across two physical
  lines. The accumulation parser's `continue` on `json.JSONDecodeError` (without resetting
  the buffer) caused all 648 subsequent records to be silently swallowed into an
  ever-growing buffer that could never parse as valid JSON. Result: only 102/750 records
  were ever processed, 0 of which were `creative_writing` category.
- **Files changed**: `scenarios/arena_hard_creative_scenario.py`
- **Change summary**: Added `if line.startswith("{") and buf: buf = ""` before accumulation,
  so a new JSON object arriving while buf has unresolvable content discards the bad record
  and starts fresh. After fix: `get_instances()` returns 250 instances correctly.
- **Verification**: `python3 -c "from scenarios.arena_hard_creative_scenario import ...; print(len(s.get_instances('/tmp')))"` → 250

### Environment dependency (not a code bug)

- `win_rate` requires `openai/gpt-4-turbo` as judge model (`OPENAI_API_KEY` must be set)
- Locally `OPENAI_API_KEY` is unset → annotation step cannot run → `win_rate` not produced
- The eval also needs `GOOGLE_API_KEY` (or equivalent) to run `google/gemini-2.5-flash-lite`
- Both keys should be set on the evaluation server; code fixes are complete and correct

### Result

- **Code**: FIXED — scenario loads 250 instances, wiring is correct
- **Local eval**: BLOCKED — needs API keys for both inference model and judge
- **Server eval**: Should PASS once run with proper API keys set
