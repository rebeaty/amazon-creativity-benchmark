# Debugging Sprint Preliminaries

Follow these steps **exactly** in order. Do not skip steps.

---

## Step 0: GitHub Setup (Do This First)

### 0.1 Clone the repository (if you don't have it already)
```bash
git clone https://github.com/anthropics/amazon-creativity-benchmark.git
cd amazon-creativity-benchmark
```

If you already have the repo, update it to latest master:
```bash
cd /home/public/vdeshpan/amazon-creativity-benchmark
git checkout master
git pull origin master
```

### 0.2 Configure your git identity (if not already done)
```bash
git config user.name "Your Full Name"
git config user.email "your.email@example.com"
```

Verify it worked:
```bash
git config user.name
git config user.email
```

### 0.3 Create your debugging branch
**Use your assigned name from the list: Clin, Namrata, Sai, or Vijeta**

```bash
git checkout -b debug/<YOUR_NAME>
```

Examples:
```bash
git checkout -b debug/clin
git checkout -b debug/namrata
git checkout -b debug/sai
git checkout -b debug/vijeta
```

Verify you're on the right branch:
```bash
git branch
```

You should see `* debug/<YOUR_NAME>` in the output.

### 0.4 Push your branch to GitHub (creates it remotely)
```bash
git push -u origin debug/<YOUR_NAME>
```

This creates your branch on GitHub and tracks it locally. You only need to do this once.

---

## Step 1: Create Fresh Conda Environment

### 1.1 Remove any existing environment (if you have one)
```bash
conda deactivate
conda remove --name creativity-bench --all -y
```

### 1.2 Create new environment with Python 3.10
```bash
conda create --name creativity-bench python=3.10 -y
```

### 1.3 Activate the environment
```bash
conda activate creativity-bench
```

**Verify you see `(creativity-bench)` in your terminal prompt.**

---

## Step 2: Install Dependencies

### 2.1 Navigate to repo root
```bash
cd /home/public/vdeshpan/amazon-creativity-benchmark
```

### 2.2 Install the package with all dependencies
```bash
pip install -e ".[eval,dev]"
```

This installs:
- `crfm-helm>=0.5.12` (HELM evaluation framework)
- All required data format libraries (Pillow, h5py, openpyxl, PyYAML, etc.)
- Optional eval dependencies (diffusers, clip-score, etc.)
- Dev dependencies (pytest)

**Wait for the installation to complete. This may take 5-10 minutes.**

### 2.3 Verify installation
```bash
python -c "import helm; print(helm.__version__)"
```

You should see a version number (e.g., `0.5.12`). If this fails, reinstall.

---

## Step 3: Set Up API Keys (CRITICAL — READ CAREFULLY)

**⚠️ BILLING PREVENTION**: We will set API keys **locally to this project only**. This prevents:
- Accidentally using a token already in your `~/.bashrc` or `~/.zshrc`
- Accidentally using a personal account login
- Keys being committed to git

### 3.1 Create a local `.env.local` file (for this project only)
This file will **NOT** be committed to git. Each debugger creates their own.

```bash
cat > /home/public/vdeshpan/amazon-creativity-benchmark/.env.local << 'EOF'
# Anthropic (Claude API)
# Get your key from: https://console.anthropic.com/account/keys
# ONLY set this if you have a valid API key with billing enabled
export ANTHROPIC_API_KEY="sk-ant-..."

# OpenAI (GPT API)
# Get your key from: https://platform.openai.com/account/api-keys
# ONLY set this if you have a valid API key with billing enabled
export OPENAI_API_KEY="sk-..."

# OpenRouter (Multi-model API gateway)
# Get your key from: https://openrouter.ai/keys
# ONLY set this if you have a valid API key with billing enabled
export OPENROUTER_API_KEY="sk-or-..."

EOF
```

### 3.2 Add your actual API keys
Edit the `.env.local` file you just created:

```bash
nano /home/public/vdeshpan/amazon-creativity-benchmark/.env.local
```

**For ANTHROPIC_API_KEY:**
1. Go to https://console.anthropic.com/account/keys
2. Create a new API key (or copy an existing one)
3. Paste it after `sk-ant-` in the file
4. **Important**: Make sure this key has billing enabled AND you are NOT using a free trial key

**For OPENAI_API_KEY:**
1. Go to https://platform.openai.com/account/api-keys
2. Create a new API key (or copy an existing one)
3. Paste it after `sk-` in the file
4. **Important**: Make sure this key has billing enabled

**For OPENROUTER_API_KEY:**
1. Go to https://openrouter.ai/keys
2. Create a new API key (or copy an existing one)
3. Paste it after `sk-or-` in the file
4. **Important**: Make sure this key has billing enabled. OpenRouter provides access to multiple models (Claude, GPT, Llama, etc.)

**Save and exit** (Ctrl+X, then Y, then Enter if using nano).

### 3.3 Load the keys into your current shell session
```bash
source /home/public/vdeshpan/amazon-creativity-benchmark/.env.local
```

### 3.4 Verify keys are loaded (and ONLY these keys, not global ones)
```bash
echo "ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY:0:20}..."
echo "OPENAI_API_KEY=${OPENAI_API_KEY:0:20}..."
echo "OPENROUTER_API_KEY=${OPENROUTER_API_KEY:0:20}..."
```

You should see your keys truncated (first 20 chars). If empty, you didn't source the file correctly.

### 3.5 Add sourcing to your shell startup (optional, for convenience)
**ONLY do this if you want the keys to load automatically when you work on this project.**

Add this to your `~/.bashrc` or `~/.zshrc`:

```bash
# Load creativity-bench API keys when entering the repo
if [[ "$PWD" == *"amazon-creativity-benchmark"* ]]; then
    source ~/.env.local.creativity-bench 2>/dev/null
fi
```

Then save your API keys there instead:
```bash
cat > ~/.env.local.creativity-bench << 'EOF'
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
export OPENROUTER_API_KEY="sk-or-..."
EOF
chmod 600 ~/.env.local.creativity-bench
```

---

## Step 4: Verify Setup

### 4.1 Test that you can import the scenarios
```bash
python -c "from scenarios import scenario_registry; print('✓ Scenarios loaded')"
```

### 4.2 Test a quick debug run on ONE dataset
Pick a dataset from your assigned list in `debug_assignments.json`. For example:

```bash
# Make sure you're in the repo root
cd /home/public/vdeshpan/amazon-creativity-benchmark

# Run debug script for a single dataset
bash debugging_scripts/run_debug_one.sh alpaca_eval_2
```

This should run without errors. You may see warnings—those are fine.

### 4.3 Confirm your environment is isolated
Check that your conda environment is active and clean:
```bash
conda info --envs
python -c "import sys; print(sys.prefix)"
```

The prefix should show something with `creativity-bench` in it.

---

## Step 5: Git Workflow During Debugging

Once you start debugging, follow this workflow:

### 5.1 Before you start each day
Pull latest changes from master to stay in sync:
```bash
git fetch origin
git merge origin/master
```

### 5.2 After fixing or debugging a dataset
Commit your changes:
```bash
git add <files_you_changed>
git commit -m "Debug <dataset_name>: <brief description of what you did>"
```

Examples:
```bash
git commit -m "Debug alpaca_eval_2: fixed prompt template formatting"
git commit -m "Debug gauss: verified metric calculation, passed"
```

### 5.3 Push your work to GitHub
```bash
git push origin debug/<YOUR_NAME>
```

Do this regularly (daily or after each dataset completion) to back up your progress.

### 5.4 Avoid merge conflicts
- **NEVER edit files in `data/registry/` directly** unless you're certain your changes don't conflict with others
- If you need to update registry files, ask Vijeta first
- Your own debugging notes/results can go in your branch without conflict

---

## Troubleshooting

### "conda command not found"
Install Miniconda from https://docs.conda.io/projects/miniconda/en/latest/

### "crfm-helm import fails"
Re-run: `pip install -e ".[eval,dev]"` and wait for completion.

### "API key not recognized / authentication failed"
1. Check that you sourced `.env.local`: `echo $ANTHROPIC_API_KEY`
2. Verify your key is valid at https://console.anthropic.com/account/keys
3. Ensure the key has billing enabled (not a free trial key)

### ".env.local is being committed to git"
Stop and tell Vijeta immediately. This file should NEVER be in git.

### "git push fails / permission denied"
1. Verify you have git credentials set: `git config user.name`
2. Check you're pushing to the correct branch: `git branch`
3. If still failing, ask Vijeta to check GitHub permissions

### "I'm on the wrong branch"
Switch to your correct branch:
```bash
git checkout debug/<YOUR_NAME>
git pull origin debug/<YOUR_NAME>
```

### "I made changes to master by mistake"
Don't panic. Move them to your branch:
```bash
git stash
git checkout debug/<YOUR_NAME>
git stash pop
git add <files>
git commit -m "..."
git push origin debug/<YOUR_NAME>
```

### "merge conflict when pulling"
If you see a merge conflict:
1. Open the conflicted file and resolve by hand
2. Mark as resolved: `git add <file>`
3. Finish the merge: `git commit -m "Merge master into debug/<YOUR_NAME>"`
4. Push: `git push origin debug/<YOUR_NAME>`

Or, ask Vijeta for help.

---

## Step 6: Running the Debugging Scripts

Each person has their own script. Replace `<name>` with your lowercase name: `clin`, `namrata`, `sai`, or `vijeta`.

### 6.1 Run your next pending dataset (auto-picked)
```bash
cd /home/public/vdeshpan/amazon-creativity-benchmark
bash debugging_scripts/run_debug_one_<name>.sh
```

This automatically picks the first dataset from your pending list and runs it.

### 6.2 Run a specific dataset
```bash
bash debugging_scripts/run_debug_one_<name>.sh brainteaser
```

### 6.3 Check your progress
```bash
python3 debugging_scripts/_debug_helper.py status <Name>
```

Output example: `Clin: 3 done, 37 pending, 40 total`

### 6.4 What happens after each run

| Outcome | What happens |
|---|---|
| PASSED or FAILED result file written | Dataset moves from `debug_assignments_pending.json` → `debug_assignments_done.json` |
| No result file (crash/timeout) | Dataset **stays in pending** — safe to re-run |

### 6.5 View your pending datasets
```bash
python3 -c "import json; d=json.load(open('debugging_scripts/debug_assignments_pending.json')); print('\n'.join(d['<Name>']))"
```

### 6.6 View your completed datasets
```bash
python3 -c "import json; d=json.load(open('debugging_scripts/debug_assignments_done.json')); print('\n'.join(d['<Name>']))"
```

### 6.7 Typical workflow (repeat until done)
```bash
conda activate creativity-bench
source .env.local
bash debugging_scripts/run_debug_one_<name>.sh   # runs next pending
git add -A && git commit -m "Debug <dataset>: <what happened>"
git push origin debug/<name>
```

---

## File Reference

| File | Purpose |
|---|---|
| `debug_assignments_pending.json` | Datasets still to debug (per person) |
| `debug_assignments_done.json` | Datasets completed (per person) |
| `_debug_helper.py` | Bookkeeping helper (`next`, `done`, `status`) |
| `run_debug_one_<name>.sh` | Per-person debug script |
| `debug_logs/` | Logs from each run |
| `debug_results/` | PASSED/FAILED JSON files per dataset |

Check [CLAUDE.md](../CLAUDE.md) for the **Debugging Protocol for Each Dataset** checklist.
