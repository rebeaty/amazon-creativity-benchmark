# Custom Metric Requirements: Only Connect Wall (OCW)

Source: Paper Section 3.3, GitHub repo src/ocw/evaluate_only_connect.py

## Task 1: Grouping Evaluation

The primary metric is **group accuracy** - whether the model correctly identified all 4 groups.

### Evaluation Logic

From the paper's evaluation script (evaluate_only_connect.py):

1. **Parse model output**: Extract 4 groups of 4 words each from generated text
   - Expected format: newline-separated groups with comma-separated words
   - Handle variations in formatting, extra whitespace, etc.

2. **Group matching**:
   - Order of groups doesn't matter (Group A-B-C-D vs B-A-D-C both correct)
   - Order of words within groups doesn't matter
   - Use set comparison: each predicted group must exactly match one ground truth group

3. **Scoring**:
   - **Exact match per group**: 1 if predicted group exactly matches a GT group (as sets), 0 otherwise
   - **Wall solved**: 1 if all 4 groups are correct, 0 otherwise
   - Report both per-group accuracy and wall-level accuracy

### Implementation Notes

```python
def evaluate_grouping(predicted_groups, gt_groups):
    """
    Args:
        predicted_groups: List of 4 lists of words (from model output)
        gt_groups: List of 4 lists of words (ground truth)

    Returns:
        groups_correct: Number of groups matched correctly (0-4)
        wall_solved: 1 if all 4 groups correct, 0 otherwise
    """
    # Convert to sets for order-invariant comparison
    pred_sets = [set(group) for group in predicted_groups]
    gt_sets = [set(group) for group in gt_groups]

    groups_correct = 0
    matched_gt = set()

    for pred_set in pred_sets:
        for i, gt_set in enumerate(gt_sets):
            if i not in matched_gt and pred_set == gt_set:
                groups_correct += 1
                matched_gt.add(i)
                break

    wall_solved = 1 if groups_correct == 4 else 0
    return groups_correct, wall_solved
```

### Parsing Challenges

Models may output groups in various formats:
- With or without "Group 1:", "Group 2:", etc.
- With periods, semicolons, or other separators
- With connection names included (e.g., "word1, word2. Connection: X")
- Malformed (wrong number of words, hallucinated words)

The evaluation script from the paper handles these by:
- Removing "Group X:" prefixes
- Splitting by newlines, then by commas
- Removing connection annotations (". Connection: ...")
- Truncating/padding to ensure 4 groups of 4 words

### Metrics to Report

1. **Per-group accuracy**: Average number of groups correctly identified per wall (0-4)
2. **Wall-level accuracy**: Fraction of walls completely solved (all 4 groups correct)
3. **Error analysis** (from paper):
   - Misformatted outputs (empty slots due to parsing failures)
   - Hallucinated words (words not in the original 16 clues)

## Task 2: Connections Evaluation (Not Implemented)

Task 2 involves naming the connection for each group after groups are already known.
This is evaluated using:
- Exact match
- ROUGE-1 F1
- BERTScore F1

Task 2 is a separate auxiliary task focused on articulation, not creative problem-solving.

## Human Performance Baseline

From the dataset, human contestants on the show:
- Task 1 (Grouping): Data available in `human_performance.grouping` field (binary per group)
- Task 2 (Connections): ~50% solve rate reported in paper

## Reference Implementation

See the original evaluation script: `src/ocw/evaluate_only_connect.py` in the OCW GitHub repo
https://github.com/TaatiTeam/OCW
