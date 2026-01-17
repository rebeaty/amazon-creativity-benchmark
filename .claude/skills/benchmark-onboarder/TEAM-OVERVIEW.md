# Claude Code for Benchmark Onboarding

## Overview for Project Collaborators

---

## Part 1: What is Claude Code?

Claude Code is Anthropic's command-line AI assistant designed for software development and research workflows. Unlike the chat interface (claude.ai), Claude Code operates directly in your terminal with full access to your local filesystem, enabling it to read, write, and execute code autonomously.

### Key Capabilities

- **File System Access**: Reads project files, creates new code, edits existing files
- **Code Execution**: Runs scripts, tests, and build commands directly
- **Web Access**: Fetches documentation, papers, and dataset information
- **Agentic Workflows**: Completes multi-step tasks with minimal supervision
- **Context Awareness**: Understands your entire project structure

### Skills System

Claude Code supports custom "skills" — instruction files that teach it domain-specific workflows. Skills are markdown files placed in `.claude/skills/` that Claude reads before tackling relevant tasks. This allows teams to encode best practices, enforce standards, and create reproducible workflows.

---

## Part 2: Benchmark Onboarding in Our Project

We've developed a custom skill that automates the conversion of creativity benchmarks into a standardized evaluation format. This addresses a core challenge: our systematic review identified **283 unique benchmarks** across diverse formats, and manually writing integration code for each would be prohibitive.

### The Problem

Each benchmark exists in different forms:
- HuggingFace datasets with varying schemas
- GitHub repositories with custom formats
- PDFs with prompts described in prose
- Different evaluation paradigms (MC, open-ended, rating scales)

### Our Solution: The Benchmark Onboarder Skill

The skill encodes a 5-step workflow:

| Step | Action |
|------|--------|
| **1. Qualify** | Verify it's a creativity benchmark with extractable prompts |
| **2. Find Prompt** | Locate the exact prompt in the source paper (section/page citation required) |
| **3. Examine Dataset** | Identify which fields to use vs. skip (e.g., exclude model outputs) |
| **4. Generate Scenario** | Produce standardized Python code following HELM conventions |
| **5. Verify** | Confirm prompt matches paper, code runs, no errors |

### Core Principles

**Prompts are sacred.** The skill enforces that benchmark prompts must be extracted exactly from source papers — never generated or inferred. This preserves benchmark validity and ensures reproducibility.

**Skip model outputs.** Many datasets include prior model responses. The skill teaches Claude to detect and exclude these fields (anything with "response", "generation", "gpt_" in the name, or unusually long text).

### Example Workflow

```
User: Onboard the RiddleSense benchmark

Claude Code:
1. Fetches paper (ACL 2021)
2. Examines HuggingFace dataset schema
3. Notes: test split has no labels → uses validation
4. Generates scenario.py with proper MC format
5. Shows example prompt for verification
```

### Output

Each onboarded benchmark produces:
- `scenario.py` — Standardized loader following HELM Scenario pattern
- Documentation header citing prompt source
- Field mapping (used vs. skipped)

### Benefits

- **Consistency**: All benchmarks follow identical structure
- **Traceability**: Every prompt cites its paper source
- **Speed**: Minutes per benchmark vs. hours of manual work
- **Quality Control**: Built-in verification checklist

---

## Part 3: Team Workflow

### Work Distribution

| RA | Initials | Assigned |
|----|----------|----------|
| [Name] | CL | 70 |
| [Name] | SM | 70 |
| [Name] | VD | 70 |
| [Name] | SA | 70 |

Benchmarks are pre-assigned in `benchmarks.json` via the `assigned_to` field.

### Setup for Each RA

1. **API Key**: Each RA gets their own Claude API key (usage trackable per key)
2. **Clone repo**: `git clone https://github.com/rebeaty/amazon-creativity-benchmark`
3. **Set key**: `export ANTHROPIC_API_KEY=sk-...`
4. **Run**: `claude` in project directory

### Tracking Progress

In `benchmarks.json`, each benchmark has:
```json
{
  "name": "BenchmarkName",
  "paper_id": "abc123",
  "url": "https://...",
  "status": "pending|in_progress|completed|failed|skipped",
  "assigned_to": "CL",
  "scenario_file": "scenarios/benchmark_name.py",
  "notes": "Any issues encountered"
}
```

### Git Push Policy

**Push immediately:**
- CLAUDE.md updates (team learns from issues right away)
- Skipped/failed benchmarks (so others don't duplicate effort)

**Batch (every 3-5 benchmarks or end of session):**
- Completed scenario.py files
- benchmarks.json status updates

Simple rule: **Push when you take a break.**

### Sharing Learnings

Claude auto-generates note blocks when it encounters issues:

```markdown
## Add to CLAUDE.md:

| Benchmark | Issue | Solution |
|-----------|-------|----------|
| RiddleSense | Test split has no labels | Use validation split |
```

Copy-paste these to CLAUDE.md and commit. Team knowledge builds automatically.

### Workflow Commands

```bash
# Start Claude Code in project directory
claude

# Onboard a specific benchmark
> Onboard the [benchmark name] benchmark

# See your assigned benchmarks
> Show my assigned benchmarks (I'm [initials])
```

### Weekly Sync

1. Pull latest (especially CLAUDE.md)
2. Push any uncommitted work
3. Quick standup: blockers, patterns discovered
4. Reassign if someone finishes early

---

## Getting Started

1. Install Claude Code: `npm install -g @anthropic-ai/claude-code`
2. Get your API key from the project lead
3. Set key: `export ANTHROPIC_API_KEY=sk-...`
4. Clone repo and navigate to project directory
5. Run `claude` to start

The skill activates automatically when you mention benchmark onboarding.
