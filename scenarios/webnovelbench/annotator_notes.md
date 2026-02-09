# Annotator Requirements: WebNovelBench

Source: Paper Section 3.2, GitHub repository (novel_original_critic.py)

## Configuration for LLMAsJuryAnnotator

**Judge model:** DeepSeek-V3
**Temperature:** 0.6
**Max tokens:** 1024
**Evaluation scope:** Chapter-level (aggregate to novel-level by averaging)

## Eight Narrative Quality Dimensions

All dimensions use a **1-5 Likert scale** rating:

1. **修辞手法 (Literary Devices)** - Weight: 13.04%
   - Quantity and quality of rhetorical devices like metaphor, symbolism, personification

2. **感官描述丰富度 (Sensory Detail)** - Weight: 11.60%
   - Frequency and vividness of visual, auditory, olfactory, tactile descriptions

3. **角色平衡度 (Character Balance)** - Weight: 11.52%
   - Character appearance frequency, dialogue proportion, psychological depth balance

4. **角色对白独特性 (Dialogue Distinctiveness)** - Weight: 11.71%
   - Whether dialogue reflects distinct personalities and character voices

5. **角色一致性 (Character Consistency)** - Weight: 13.77% (highest)
   - Language and actions align with established character identity

6. **意境匹配度 (Thematic Alignment)** - Weight: 12.90%
   - Scenes and descriptions support overall atmosphere and themes

7. **语境适配度 (Contextual Appropriateness)** - Weight: 12.81%
   - Settings and details match time period, place, cultural background

8. **场景衔接度 (Scene Coherence)** - Weight: 12.63%
   - Smooth, natural transitions between scenes and narrative segments

## Scoring Pipeline

### Step 1: Individual Dimension Scoring
For each of the 8 dimensions, the judge provides a 1-5 rating with brief explanation.

### Step 2: Score Normalization
Apply z-score standardization using pre-computed parameters from 4,000-novel reference:

```python
# Fixed parameters from fixed_parameters.json
MEANS = [2.524, 2.786, 3.320, 3.221, 3.598, 3.120, 3.518, 3.288]
STDS = [0.481, 0.516, 0.262, 0.380, 0.316, 0.405, 0.318, 0.248]

# Normalize each dimension
standardized_scores = [(raw_score - mean) / std
                       for raw_score, mean, std in zip(raw_scores, MEANS, STDS)]
```

### Step 3: PCA-Weighted Aggregation
Combine normalized scores using PCA-derived weights:

```python
WEIGHTS = [0.1304, 0.1160, 0.1152, 0.1171, 0.1377, 0.1290, 0.1281, 0.1263]

weighted_score = sum(s * w for s, w in zip(standardized_scores, WEIGHTS))
```

### Step 4: Min-Max Scaling
Scale to [0, 1] range using fixed bounds:

```python
MIN_SCORE = -3.564746953970711
MAX_SCORE = 2.758144772817456

normalized_score = (weighted_score - MIN_SCORE) / (MAX_SCORE - MIN_SCORE)
```

### Step 5: Percentile Ranking
Map normalized score to percentile rank using ECDF over 4,000 reference novels.

## Judge Prompt Template

The paper does not specify exact prompt wording. Based on the methodology and code structure, the prompt should follow this pattern for each dimension:

```
请评估以下小说文本在"{维度名称}"方面的表现，使用1-5分进行评分：

{章节文本}

评分标准：
1分 - 非常差
2分 - 较差
3分 - 一般
4分 - 较好
5分 - 优秀

请提供你的评分（仅数字）。
```

**Translation:**
```
Please evaluate the following novel text on "{dimension_name}", rating from 1-5:

{chapter_text}

Rating criteria:
1 - Very poor
2 - Poor
3 - Average
4 - Good
5 - Excellent

Please provide your rating (number only).
```

## Implementation Notes

1. **Multi-chapter aggregation**: Each novel has 10 chapters. Score each chapter independently, then average across chapters for the novel-level score.

2. **Reference distribution**: The 4,000-novel reference corpus defines the percentile mapping. This corpus is not publicly released, so reproduction requires:
   - Using the fixed normalization parameters provided
   - Implementing a compatible percentile mapping function
   - Or comparing against a new reference distribution

3. **Human correlation**: Paper reports human-LLM correlation of 0.82 (Spearman's ρ) for the overall percentile ranking.

4. **Position bias**: The authors note potential position bias in the judge model but do not specify mitigation strategies.

5. **Language**: All prompts and evaluation must be conducted in Chinese. The judge model must have strong Chinese language understanding.

## API Configuration Example

From the repository's `config_example.json`:

```json
{
  "critic": {
    "num_threads": 2,
    "api_url": "https://api.myapi.cn/v1",
    "model": "DeepSeek-V3",
    "temperature": 0.6,
    "max_tokens": 1024,
    "api_key": "..."
  }
}
```

## Required Resources

1. DeepSeek-V3 API access
2. Fixed normalization parameters (`fixed_parameters.json` from repository)
3. 4,000-novel reference distribution (for percentile mapping)
4. Chinese language processing capabilities
