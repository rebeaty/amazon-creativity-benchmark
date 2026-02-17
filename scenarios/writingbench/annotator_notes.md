# Annotator Requirements: WritingBench

Source: Paper Appendix C.6 (pages 32-33), https://arxiv.org/pdf/2503.05244
Repository: `prompt.py` file, `evaluate_benchmark.py` script

## Configuration for LLMAsJuryAnnotator

**Judge model:** Claude models (paper mentions "LLM-as-a-Judge" evaluation pathway)

**Alternative:** Finetuned critic model (paper provides second evaluation pathway)

**Evaluation approach:** Instance-specific rubric-based scoring

**Dimensions:** 5 evaluation criteria per query (instance-specific, vary by task)

**Scale:** 1-10 integer score per criterion

**Output format:** JSON with score (1-10) and detailed reason/justification

## Evaluation Methodology

### Instance-Specific Criteria

Each of the 1,000 queries has 5 unique evaluation criteria stored in the dataset's `checklist` field. Each criterion includes:

1. **name**: Brief criterion label
2. **criteria_description**: Detailed description of what the criterion evaluates
3. **Scoring rubrics**: 5 detailed descriptions for score ranges:
   - "1-2": Critical deficiencies and major issues
   - "3-4": Below average, noticeable shortcomings
   - "5-6": Adequate but not exemplary, baseline performance
   - "7-8": Above average, strong performance with minor refinements needed
   - "9-10": Exceptional performance, all aspects optimally addressed

**Example criteria** (for a paper outline task):
1. Paper structure completeness and academic standards
2. Machine learning algorithm innovation
3. Engineering application effectiveness
4. Content logic and coherence
5. Reference material integration

**Example criteria** (for a building energy control paper):
1. Research Context Alignment
2. Data Analysis Framework
3. Methodological Rigor
4. Technical Depth and Specificity
5. Practical Implementation and Impact Assessment

### Scoring Rules

From `prompt.py`:

```
"1-2": "Critical deficiencies and major issues that prevent adequate functionality."
"3-4": "Lacking with noticeable shortcomings that impact overall effectiveness and require improvement."
"5-6": "Adequate but not exemplary, Baseline performance that meets essential requirements. Most models may achieve this score."
"7-8": "Strong performance characterized by competent execution, though minor refinements are needed to achieve excellence."
"9-10": "Exceptional performance with all aspects optimally addressed, demonstrating superior effectiveness and quality without any flaws."
```

### Strict Evaluation Guidelines

The prompt emphasizes strict evaluation:
- **Be STRICT**: Do not be misled by format or length
- **Check for substance**: Discern whether content appears substantial but is actually fabricated
- **Detect incomplete responses**: Models may only provide an introduction/overview without completing the query
- **Justify with evidence**: Reference exact text passages to support scores
- **Align with rubrics**: Ensure reasons concrete and aligned with criteria requirements

## Judge Prompt Template

From `prompt.py`:

```python
evaluate_system = """
You are an expert evaluator with extensive experience in evaluating response of given query.
"""

evaluate_prompt = """
Evaluate the Response based on the Query and Criteria provided following the Scoring Rules.

** Scoring Rules **

"1-2": "Low score description: Critical deficiencies and major issues that prevent adequate functionality.",
"3-4": "Below average score description: Lacking with noticeable shortcomings that impact overall effectiveness and require improvement.",
"5-6": "Average score description: Adequate but not exemplary, Baseline performance that meets essential requirements. Most models may achieve this score.",
"7-8": "Above average score description: Strong performance characterized by competent execution, though minor refinements are needed to achieve excellence.",
"9-10": "High score description: Exceptional performance with all aspects optimally addressed, demonstrating superior effectiveness and quality without any flaws."

- Provide reasons for each score by indicating specific strengths or deficiencies within the Response.
- Reference exact text passages to justify the score, ensuring that each reason is concrete and aligns with the criteria requirements while highlighting key gaps from the ideal answer.
- Be very STRICT and do not be misled by format or length; ensure that the Response is thoroughly evaluated beyond superficial appearances.
- Carefully discern whether the content of the Response is an illusion, appearing substantial but actually entirely fabricated.
- Sometimes the model may only provide an introduction or an overview without truly completing the query, which should be considered a failed response. Carefully discern this.
- Scoring Range: Assign an integer score between 1 to 10

** Output format **
(Remove symbols that interfere with JSON parsing, don't use " inside reason)
Return the results in the following JSON format, Only output the following JSON format and nothing else:
```json
{
    "score": an integer score between 1 to 10,
    "reason": "Specific and detailed justification for the score using text elements."
}
```

** Criteria **
```{criteria}```

** Query **
```{query}```

** Response **
```{response}```

Provide your evaluation based on the criteria restated below:

```{criteria}```

** Output format **
(Remove symbols that interfere with JSON parsing, don't use " inside reason)
Return the results in the following JSON format, Only output the following JSON format and nothing else:
```json
{
    "score": an integer score between 1 to 10,
    "reason": "Specific and detailed justification for the score using text elements."
}
```
"""
```

## Implementation Process

1. **For each query**: Evaluate the model's response against all 5 instance-specific criteria
2. **Per criterion evaluation**:
   - Read the criterion name and description
   - Review the 5 scoring rubrics (1-2, 3-4, 5-6, 7-8, 9-10)
   - Evaluate the response against these rubrics
   - Assign an integer score from 1-10
   - Provide detailed justification with text evidence
3. **Output**: JSON with `score` and `reason` fields per criterion
4. **Aggregate**: Final score can be mean/median of 5 criteria scores

## Dataset Structure

The `checklist` field in each query contains the evaluation criteria:

```python
{
    "index": 1,
    "domain1": "Academic & Engineering",
    "domain2": "Paper Outline",
    "lang": "zh",
    "query": "...",
    "checklist": [
        {
            "name": "Criterion 1 Name",
            "criteria_description": "What this criterion evaluates",
            "1-2": "Description for low scores",
            "3-4": "Description for below average",
            "5-6": "Description for average",
            "7-8": "Description for above average",
            "9-10": "Description for exceptional"
        },
        # ... 4 more criteria
    ]
}
```

## Model Parameters

From repository code:

**Response generation:**
- `top_p`: 0.8
- `top_k`: 20
- `temperature`: 0.7
- `max_length`: 16000

**Scoring/Evaluation:**
- `top_p`: 0.95
- `temperature`: 1.0
- `max_length`: 2048

## Domain Coverage

WritingBench spans 6 primary domains with 100 fine-grained subdomains:

1. **Academic & Engineering** (167 queries)
   - Paper sections: Outline, Abstract, Introduction, Literature Review, Experiments, Conclusion
   - Technical documentation, Research proposals, Defense presentations

2. **Finance & Business** (210 queries)
   - Business plans, Financial reports, Market analysis
   - Contracts, Proposals, Meeting minutes

3. **Politics & Law** (201 queries)
   - Legal documents, Policy drafts, Judicial opinions
   - Government reports, Diplomatic correspondence

4. **Literature & Arts** (183 queries)
   - Creative writing: Stories, Poems, Scripts
   - Literary analysis, Art critiques, Reviews

5. **Education** (111 queries)
   - Lesson plans, Course materials, Educational assessments
   - Student feedback, Teaching reflections

6. **Advertising & Marketing** (128 queries)
   - Ad copy, Marketing campaigns, Brand stories
   - Product descriptions, Social media content

## Multilingual Evaluation

- **Chinese queries**: 445 (44.5%)
- **English queries**: 555 (55.5%)

Criteria names and rubric descriptions are in the corresponding language of the query.

## Human Correlation

The paper mentions rubric-based scoring with both LLM-as-judge and a finetuned critic model, but specific human correlation coefficients are not provided in the available documentation.

## Leaderboards

- **HuggingFace**: https://huggingface.co/spaces/WritingBench/WritingBench
- **ModelScope**: https://modelscope.cn/studios/WritingBench/WritingBench

## Implementation Notes

- Each query requires 5 separate evaluations (one per criterion)
- Total API calls per model: 1,000 queries × 5 criteria = 5,000 evaluations
- Consider batching or parallel evaluation to reduce latency
- Store criterion-level scores separately for analysis
- Final aggregate score is typically the mean across 5 criteria
