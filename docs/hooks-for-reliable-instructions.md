# Why Claude "Forgets" Instructions (And How to Fix It)

A guide for GAs on using hooks to make Claude reliably follow project rules.

---

## The Problem

You've probably noticed: sometimes Claude ignores rules from CLAUDE.md or LEARNINGS.md, even when they're clearly documented.

This is a **known issue** with multiple bug reports on GitHub. It happens because:

| Cause | What happens |
|-------|--------------|
| **Context compaction** | Claude compresses context to save tokens and "forgets" instructions |
| **Conflicting signals** | Your request seems to need something that contradicts the rules |
| **Distance decay** | Instructions at the start get deprioritized as the conversation grows |

**Bottom line**: CLAUDE.md is read once at session start, then can fade from attention.

---

## The Solution: Hooks

Hooks are shell commands that run automatically at specific points in Claude's workflow. The key one for us:

**`UserPromptSubmit`** — runs every time you send a message, and its output gets injected into Claude's context.

This means you can **re-inject critical rules on every single prompt**. Claude can't forget what's right in front of it.

---

## How It Works

### 1. Create a settings file

File: `.claude/settings.json`

```json
{
  "hooks": {
    "UserPromptSubmit": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "cat \"$CLAUDE_PROJECT_DIR/.claude/rules/critical-rules.txt\""
          }
        ]
      }
    ]
  }
}
```

### 2. Create your critical rules file

File: `.claude/rules/critical-rules.txt`

```
## Benchmark Onboarding Reminders

- ALWAYS read LEARNINGS.md before starting a new benchmark
- NEVER use fields containing "gpt" or "response" as model inputs
- ALWAYS check if the test split has labels before using it
- ALWAYS include CORRECT_TAG on reference answers
- Field names in papers often differ from actual dataset columns—inspect first
```

### 3. That's it

Every time you send a message, those rules get injected fresh into context.

---

## Why This Works Better

| Approach | Behavior |
|----------|----------|
| CLAUDE.md | Read once, can be compacted away |
| UserPromptSubmit hook | **Re-injected every prompt**, always visible |

From the community:

> "Prompts are suggestions. Claude can be convinced to ignore them. Hooks are different—they execute regardless of what Claude thinks it should do."

---

## Other Useful Hooks

### Validate scenarios after writing them

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Write",
        "hooks": [
          {
            "type": "command",
            "command": "if [[ \"$FILE\" == *scenarios/*.py ]]; then python -c \"import $(basename $FILE .py)\" 2>&1 | head -5; fi"
          }
        ]
      }
    ]
  }
}
```

This automatically checks that any scenario Claude writes is valid Python.

### Block dangerous operations

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          {
            "type": "command",
            "command": "if echo \"$COMMAND\" | grep -q 'rm -rf'; then echo 'Blocked: dangerous command' >&2; exit 2; fi"
          }
        ]
      }
    ]
  }
}
```

Exit code 2 = blocked. No negotiation.

---

## Periodic Injection (Advanced)

If you don't want rules on *every* prompt, inject every N prompts:

```bash
#!/bin/bash
# .claude/hooks/periodic-reminder.sh

COUNTER_FILE="/tmp/claude-counter"
COUNT=$(($(cat "$COUNTER_FILE" 2>/dev/null || echo 0) + 1))
echo $COUNT > "$COUNTER_FILE"

# Inject every 5 prompts
if [ $((COUNT % 5)) -eq 0 ]; then
    cat "$CLAUDE_PROJECT_DIR/.claude/rules/critical-rules.txt"
fi
```

---

## Quick Reference: Hook Events

| Hook | When it fires | Common use |
|------|---------------|------------|
| `SessionStart` | Session begins | Load environment, set context |
| `UserPromptSubmit` | You send a message | **Inject reminders** |
| `PreToolUse` | Before Claude runs a tool | Block dangerous operations |
| `PostToolUse` | After a tool succeeds | Validate output, run linters |
| `Stop` | Claude finishes responding | Check if work is complete |

---

## Key Takeaways

1. **CLAUDE.md can be forgotten** — it's read once, then fades
2. **Hooks are deterministic** — they run every time, no exceptions
3. **UserPromptSubmit injects context** — use it for must-follow rules
4. **Exit code 2 blocks** — use PreToolUse to prevent bad operations
5. **Keep critical rules short** — they're injected every prompt, so be concise

---

## Learn More

- [Official Hooks Docs](https://code.claude.com/docs/en/hooks)
- [Hooks Guide with Examples](https://code.claude.com/docs/en/hooks-guide)
- [DataCamp Tutorial](https://www.datacamp.com/tutorial/claude-code-hooks)

---

## Our Project (Not Yet Implemented)

We currently use CLAUDE.md and Skills only. Adding hooks would help ensure LEARNINGS.md rules are consistently followed during benchmark onboarding.

Proposed addition:
```
.claude/
├── settings.json          # Hook configuration
└── rules/
    └── critical-rules.txt # Must-follow rules, injected every prompt
```
