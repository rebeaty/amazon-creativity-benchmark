#!/bin/bash

# UserPromptSubmit Hook - Injects workflow reminders before every response
# This ensures consistent benchmark processing without asking for confirmation

cat <<'EOF'
═══════════════════════════════════════════════════════════
⚠️  BENCHMARK WORKFLOW - AUTOMATIC UPDATES REQUIRED ⚠️
═══════════════════════════════════════════════════════════

When user says "run benchmark X" or "lets run this [name]":

1. 🔍 INVESTIGATE thoroughly (WebFetch paper/repo)
2. ✅ DETERMINE suitability for HELM onboarding
3. 📝 AUTOMATICALLY update benchmarks.json:
   - Set status: "not_suitable" or "completed"
   - Add detailed reason
   - Update url to arXiv or primary source
4. 📚 AUTOMATICALLY update LEARNINGS.md:
   - Append row to "Benchmarks That Don't Qualify" table
   - Include name, issue, and detailed reason
5. 📢 REPORT what was updated

⚠️ CRITICAL: DO NOT ASK "Should I mark as not_suitable?"
⚠️ CRITICAL: DO NOT ASK "Should I update the files?"
⚠️ The user has ALREADY told you to run it - just investigate and update!

═══════════════════════════════════════════════════════════
EOF
