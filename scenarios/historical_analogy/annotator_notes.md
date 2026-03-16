# Annotator Requirements: Historical Analogy Acquisition

Source: evaluation.py from https://github.com/Nianqi-Li/Historical-Analogy-of-LLMs

## Configuration for LLMAsJuryAnnotator

Judge model: GPT-4
Temperature: 0.0000000001 (effectively 0 for deterministic scoring)
Task: Score abstract-level similarity between predicted and reference analogies
Scale: 1-4 Likert scale

## Evaluation Metrics

### 1. Pass@1 (Exact Match)
- Uses Wikipedia search API to find variants of both target and predicted events
- Checks if any variant of the predicted event matches any variant of the target
- Binary: 1 if match found, 0 otherwise
- Implementation: `pass_1()` function in evaluation.py

### 2. Multi-dimensional Abstract Similarity
- Focus: Abstract-level similarity (topic, general situation, themes)
- NOT surface-level similarity (specific entities, dates, locations)
- Scored by GPT-4 judge using structured prompt

## Judge Prompt Template

```
You are a sentence-level analogy scoring robot. For the two input texts, please judge
the quality of the analogy and give it a score (1-4). It should be noted that the
quality of an analogy only focuses on the abstract-level similarity of descriptions,
not the surface similarity of descriptions. For example, in a good analogy, two
descriptions may belong to the same topic and express similar general situations,
but they may not necessarily be the same specific process or description.

## Grading
1 point: The description belongs to a completely different topic or field, has no
         connection, and cannot be compared.
2 points: The descriptions belong to the same general theme, but the general situation
          or aspect expressed is significantly different, and the quality of the analogy
          is low.
3 points: The descriptions belong to the same topic and express similar general
          situations, but are somewhat different in details or focus. This is an
          acceptable analogy.
4 points: The descriptions belong to exactly the same topic, the general situation
          expressed is highly similar, the concepts and key points are highly
          overlapping, and it is a good analogy.

In addition, there are several points to note:
1. [Self-analogy is bad!!!]. Similarly, if one description overwrites another
   description, it is also a bad analogy.
2. The quality of an analogy is only affected by abstract-level similarity and the
   similarity or identity of entities does not affect the quality of the analogy.
   For example, "The United States attacked Japan" and "The United States helped Japan"
   are completely incomparable; while "The United States attacked Japan" and "Germany
   invaded France" are good analogies.

## The following is two case:
Case Description 1: On September 11, 2001, 19 terrorists hijacked four planes, crashing
them into the World Trade Center, the Pentagon, and a field in Pennsylvania after a
passenger revolt.
Case Description 2: On December 7, 1941, 353 Japanese aircraft attacked Pearl Harbor,
damaging or sinking eight battleships and destroying over 180 U.S. aircraft.
Score: 3

Case Description 1: The spillover of the Syrian Civil War had significant impacts in
the Arab world and beyond, leading to a wider regional conflict and the rise of the
Islamic State of Iraq and the Levant.
Case Description 2: The Revolutions of 1989 were a series of political changes that
led to the end of communist rule in Central and Eastern Europe, marking the end of
the Cold War.
Score: 2

Case Description 1: {predicted_event_description}
Case Description 2: {target_event_description}
Score:
```

## Input Preparation

Before scoring, the judge needs event descriptions (not just names):
1. Fetch Wikipedia summary for predicted analogy event
2. Fetch Wikipedia summary for target event
3. Extract 4-part summary using GPT-4:
   - Summary: One concise sentence
   - Background: Historical context
   - Process: What happened
   - Result: Consequences and impact
4. Score similarity on the "Process" component (core event dynamics)

See `extract_features()` and `abstract_similarity()` functions in evaluation.py

## Additional Metrics

The paper also uses Jaccard similarity on keywords as a supplementary metric:
- Tokenize both descriptions
- Remove stopwords
- Calculate Jaccard index: |intersection| / |union|
- Implementation: `jacc()` function in evaluation.py

## Notes

- Human evaluation was also conducted with 3 annotators
- Annotators compensated above local minimum wage
- Inter-annotator agreement details not specified in code
- Focus on abstract patterns, not surface similarity
- Self-analogies (same event) are explicitly discouraged
