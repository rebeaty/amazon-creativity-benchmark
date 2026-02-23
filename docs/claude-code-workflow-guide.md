# Claude Code Workflow Guide for GAs

A quick guide to getting the most out of Claude Code on this project, based on how top teams are using it.

---

## How Claude Code "Remembers" Things

Claude Code has a **memory hierarchy** that loads context every time you start a session:

| File | Scope | What to put there |
|------|-------|-------------------|
| `~/.claude/CLAUDE.md` | Personal (all projects) | Your initials, preferred style, personal shortcuts |
| `CLAUDE.md` | This project | Project overview, key commands, conventions |
| `.claude/skills/*/SKILL.md` | Specific workflows | Task-specific instructions (like benchmark onboarding) |
| `.claude/skills/*/LEARNINGS.md` | Team knowledge | Mistakes we've caught, patterns that work |

**Key insight**: Claude reads these files automatically. When you teach Claude something once and add it to the right file, it remembers forever.

---

## Essential Commands to Know

| Command | When to use |
|---------|-------------|
| `/cost` | Check token usage (run every 30-45 min) |
| `/compact` | Summarize context when tokens get high (>50k) |
| `/clear` | Fresh start—use before switching to a different benchmark |
| `#` key | Tell Claude to "remember" something (adds to CLAUDE.md) |
| Plan mode toggle | Switch between planning and execution |

**Pro tip**: Use `/clear` between benchmarks to avoid "context contamination" where Claude confuses details from different datasets.

---

## The "Plan First" Workflow

Anthropic's own team follows this pattern:

1. **Start in Plan mode** — Ask Claude to propose a plan in bullets
2. **Iterate on the plan** — Go back and forth until it looks right
3. **Switch to execution** — Turn on auto-accept and let it implement
4. **Review** — Check the output against your plan

This prevents Claude from rushing into implementation before understanding the task.

---

## Capturing Learnings (Critical!)

When Claude makes a mistake or you discover a quirk:

1. **Fix the immediate problem**
2. **Add it to LEARNINGS.md** so it won't happen again

Example entry:
```markdown
## HuggingFace Datasets
- The `test` split often has no labels—always check before using
- Field names in the paper may differ from actual dataset columns
```

The team's LEARNINGS.md is in `.claude/skills/benchmark-onboarder/LEARNINGS.md`. Check it before starting a new benchmark—it'll save you time.

---

## What Makes a Good CLAUDE.md Entry

**Do add:**
- Specific commands: `pytest scenarios/ -v`
- Concrete conventions: "Use lowercase_underscore for scenario filenames"
- Known gotchas: "Don't trust field names in dataset documentation"

**Don't add:**
- Vague instructions: "Write clean code"
- Things Claude already knows: "Use Python best practices"

---

## Session Management Tips

| Time | Action |
|------|--------|
| Every 30 min | Run `/cost` to check token usage |
| When switching tasks | Use `/clear` for a fresh context |
| End of session | Add any new learnings before closing |
| When stuck | Try `/compact` to refocus Claude |

---

## Quick Reference: Project Structure

```
.claude/
├── skills/
│   └── benchmark-onboarder/
│       ├── SKILL.md          # The main workflow (5 steps)
│       ├── LEARNINGS.md      # Team knowledge—READ THIS FIRST
│       ├── TEAM-OVERVIEW.md  # Detailed guide
│       └── examples/         # Reference implementations
```

---

## Useful Resources

- [Official Memory Docs](https://code.claude.com/docs/en/memory) — How the memory hierarchy works
- [Best Practices](https://www.anthropic.com/engineering/claude-code-best-practices) — Anthropic's official guide
- [awesome-claude-code](https://github.com/hesreallyhim/awesome-claude-code) — Community skills and plugins

---

## TL;DR

1. **Read LEARNINGS.md** before starting a new benchmark
2. **Use Plan mode first**, execute second
3. **Run `/cost` regularly**, `/compact` when needed
4. **Add learnings** when you discover something new
5. **Use `/clear`** between different benchmarks

Questions? Check the TEAM-OVERVIEW.md or ask the team.
