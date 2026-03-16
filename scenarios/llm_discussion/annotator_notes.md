# Annotator Requirements: LLM Discussion Creativity Tests

Source: Paper (arXiv:2405.06373, COLM 2024) and GitHub Repository (lawraa/LLM-Discussion)

## Overview

The LLM Discussion benchmark includes 4 classic divergent thinking creativity tests, each with 30 test items:

1. **Alternative Uses Test (AUT)**: Generate creative uses for everyday objects
2. **Similarities Test**: Find ways in which two things are alike
3. **Instances Test**: Name all things in a category
4. **Scientific Creativity Test**: Scientific creativity across 5 question types

All tests use the same evaluation framework with 4 dimensions.

## Configuration for LLMAsJuryAnnotator

Judge model: GPT-4 or GPT-3.5-turbo
Dimensions: Fluency, Flexibility, Originality, Elaboration
Output: Numeric scores for each dimension

## Evaluation Framework

These tests are classic divergent thinking tasks used in creativity research. Models generate multiple creative responses to open-ended prompts.

**Setup:**
1. The model receives a prompt for one of the 4 test types
2. The model generates a list of creative responses
3. An LLM judge evaluates the response on four dimensions

## Test Types

### 1. Alternative Uses Test (AUT)
- **Task**: Generate creative and diverse uses for everyday objects
- **Examples**: "What are some creative uses for an Umbrella?", "...for a Key?", "...for Scissors?"
- **30 test objects**: Umbrella, Key, Camera, Whistle, Guitar, Couch, Jeans, Pillowcase, Broom, Envelope, Scissors, Calculator, Towel, Rope, Helmet, Lamp, Table, Wrench, Shoe, Card, Tire, Basket, Candle, Teddy bear, Drum, Shovel, Flashlight, Lid, Zipper

### 2. Similarities Test
- **Task**: Find ways in which two things are alike
- **Examples**: "Tell me all the ways in which a hedgehog and a porcupine are alike.", "...a diary and a planner...", "...a crayon and a marker..."
- **30 test pairs**: hedgehog/porcupine, diary/planner, crayon/marker, duck/swan, passport/ticket, salad/soup, fox/wolf, jellyfish/octopus, poem/novel, butterfly/bee, brush/broom, shoe/sock, movie/play, park/garden, lamp/flashlight, goldfish/guppy, lion/cheetah, comet/asteroid, skateboard/rollerblade, ice cream/yogurt, sandwich/pizza, cloud/fog, frog/toad, snake/lizard, diary/journal, violin/cello, ballet/opera, donkey/horse, apple/pear, berry/nut

### 3. Instances Test
- **Task**: Name all things that fit in a category
- **Examples**: "Name all the things you can think of that are soft.", "...that are found in a bathroom.", "...that use electricity."
- **30 test categories**: used in culture, found in a bathroom, soft, used in celebrations, can be written on, used in pastimes, used for cleaning, use electricity, languages, used in a pet store, used in sports, spicy, used in recreation, used by a dentist, used in a library, used by a sailor, found in a forest, used in a kitchen, used in sociology, hot, used in technology, worn on feet, used by an electrician, have keys, used in sports (duplicate), fruits, used on a computer, used in music, used in enjoyment, require electricity (duplicate)

### 4. Scientific Creativity Test
- **Task**: Scientific creativity across 5 question types with 6 examples each (30 total)
- **Question types**:
  1. Scientific uses for objects (e.g., "Creative Scientific Applications of a Pencil")
  2. Scientific questions to research (e.g., "What experiments would you design if you discovered a mirror that showed the future?")
  3. Improvements to objects (e.g., "How can you upgrade a conventional pen to make it more innovative?")
  4. Hypothetical scenarios (e.g., "Suppose gravity could be turned on and off like a switch. How would this change architecture?")
  5. Testing/comparison methods (e.g., "How can you test which of two smartphones has a longer battery life?")

## Judge Prompt Templates

The evaluation uses GPT-4 or GPT-3.5 to assess responses across four dimensions (same for all 4 tests):

### 1. Fluency
**Definition:** Count of unique, relevant responses

**Prompt excerpt:**
```
Identify and count the number of unique, relevant responses and explain why.
```

**Output:** Integer count of unique relevant ideas

### 2. Flexibility
**Definition:** Variety of distinct categories or perspectives

**Prompt excerpt:**
```
Evaluate the flexibility of the relevant responses, where flexibility refers to
the variety of distinct categories or perspectives represented in the responses.
```

**Output:** Integer count of distinct categories

### 3. Originality
**Definition:** Novelty and uniqueness of ideas

**Prompt excerpt:**
```
Evaluate the overall originality of the responses on a scale from 1 to 5,
with 5 being the highest level of originality.
```

**Output:** Score from 1-5 (5 = most original)

### 4. Elaboration
**Definition:** Detail and development of responses

**Prompt excerpt:**
```
Evaluate the overall level of elaboration in the responses. Elaboration refers
to the amount of detail and development provided for each response. Rate the
elaboration on a scale of 1 to 5.
```

**Output:** Score from 1-5 (5 = most elaborated)

## Evaluation Procedure

From the paper and repository:
1. Use 30 test items for each of the 4 tests (120 total items across all tests)
2. Generate creative responses for each prompt
3. Submit each response to LLM judge for scoring on all four dimensions
4. Report scores for each dimension (can be averaged across all items or by test type)

## Implementation Requirements

For HELM's `LLMAsJuryAnnotator`:
- Judge model: `gpt-4` or `gpt-3.5-turbo`
- Parse numeric scores for each dimension:
  - fluency_score (integer)
  - flexibility_score (integer)
  - originality_score (1-5 scale)
  - elaboration_score (1-5 scale)
- Multiple prompt variations available in repository (default, fewshot, rubric, sampling, pairwise)
- Primary metrics: All four dimensions are equally important
- Repository provides detailed judge prompts in `Evaluation/eval_functions/eval_prompts.py`

## Notes

- All 4 tests are well-established creativity assessments in psychology and cognitive science
- The four dimensions (Fluency, Flexibility, Originality, Elaboration) are standard in creativity research (Guilford's work)
- The paper demonstrates that their LLM Discussion framework outperforms single-LLM approaches across all 4 tests
- Repository includes code for automated grading using GPT models
- Human evaluation was also conducted to validate the automated scoring
- The benchmark evaluates divergent thinking - the ability to generate multiple creative solutions

## Test-Specific Evaluation Notes

### AUT
- Focus on unconventional uses (not typical/common uses)
- Creative ideas should be clever, unusual, interesting, humorous, innovative, or different

### Similarities
- Look for abstract and conceptual similarities (not just obvious physical similarities)
- More creative responses identify deeper connections

### Instances
- Evaluate breadth of category coverage
- More creative responses include unusual category members

### Scientific
- Evaluate scientific validity along with creativity
- Question types vary in complexity and creativity demands
- Some questions are more hypothetical/speculative than others

## Alternative Metrics

While the paper uses LLM-as-judge evaluation, complementary metrics could include:
- Self-BLEU (diversity across generated responses)
- Semantic similarity to training data (measuring originality via distance from common responses)
- Category detection via clustering (automated flexibility measurement)
- Response length distribution (proxy for elaboration)
