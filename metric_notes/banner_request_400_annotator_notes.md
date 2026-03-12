# Annotator Requirements: BannerRequest400

Source: eval.py in https://github.com/sony/BannerAgency

## Configuration for LLMAsJuryAnnotator

Judge model: GPT-4o (Azure) or Claude 3.5 Sonnet
Temperature: 0.3
Dimensions: TAA, LPS, AQS, CTAE, CPYQ, BIS
Scale: 1-5 Likert per dimension
Output format: {"score": int, "explanation": str}

## Master Prompt Template

```
You are an expert in advertising design, marketing, and visual communication.
Your task is to evaluate a banner ad image based on the following principle
given the advertiser's logo and banner request. You should rate on a scale
of 1 to 5, where 1 is poor and 5 is excellent. You should also provide a
brief justification for your score.

{score_principle}

Please start evaluating the banner ad image. Output your answer in the format
of {"score": , "explanation": "explain concisely why you gave this score"}
```

## Metric Rubrics (from eval.py)

### TAA - Theme & Audience Alignment

Definition: Measures how well the generated banner ad aligns with the given request, including the theme, target audience, and primary purpose.

- 5: Perfectly aligns with the request (theme, audience, purpose are all clearly reflected)
- 4: Mostly aligns, but minor details could be improved
- 3: Somewhat aligns, but key elements are missing or unclear
- 2: Barely aligns, with major missing or incorrect elements
- 1: Does not align with the request at all

Justification Required: Explain how well the banner captures the requested theme and audience.

### LPS - Logo Placement Score

Definition: Evaluates whether the logo is well-integrated into the design in terms of visibility, size, and positioning.

- 5: Logo is well-placed, clearly visible, proportionate, and blends seamlessly
- 4: Logo is well-placed but could be slightly improved (e.g., minor size or position adjustments)
- 3: Logo is visible but not ideally placed (e.g., too small, too large, or slightly obstructed)
- 2: Logo placement is poor (e.g., difficult to notice, awkward positioning)
- 1: Logo is either missing or completely misplaced

Justification Required: Explain how the logo is positioned and whether it contributes to brand identity.

### AQS - Aesthetic Quality Score

Definition: Measures the visual appeal, including color harmony, layout balance, typography, and overall design quality.

- 5: Visually outstanding, professional design, well-balanced, with harmonious colors and readable text
- 4: Well-designed, but small refinements could enhance it
- 3: Acceptable but has notable design flaws (e.g., poor contrast, unbalanced elements)
- 2: Visually weak, with noticeable design mistakes
- 1: Poor design, lacks professionalism or coherence

Justification Required: Explain what makes the design appealing or unappealing.

### CTAE - CTA Effectiveness

Definition: Evaluates whether the Call-to-Action (CTA) is clear, engaging, and visually emphasized.

- 5: CTA is clear, compelling, well-placed, and visually prominent
- 4: CTA is effective but could be slightly improved (e.g., contrast, size)
- 3: CTA is present but lacks emphasis or clarity
- 2: CTA is weak, hard to notice, or poorly worded
- 1: No clear CTA is present

Justification Required: Explain how effective the CTA is in prompting user action.

### CPYQ - Copy Quality

Definition: Evaluates the effectiveness of the headline, subheadline, and any other text in the banner ad, focusing on clarity, readability, persuasiveness, and grammatical correctness.

- 5: Copy is clear, engaging, grammatically correct, and persuasive, making the message effective
- 4: Copy is well-written but could be slightly improved (e.g., minor word choice refinements)
- 3: Copy is somewhat effective but has issues in clarity, grammar, or persuasiveness
- 2: Copy is weak, hard to read, contains noticeable grammatical mistakes, or lacks impact
- 1: Copy is unclear, irrelevant, or difficult to read due to poor design or bad wording

Justification Required:
- Is the copy easy to read against the background?
- Does it match the banner's purpose and target audience?
- Is it persuasive and action-driven?
- Are there any grammatical or spelling errors?

### BIS - Brand Identity Score

Definition: Measures how well the banner ad visually and stylistically aligns with the brand's identity beyond just logo placement.

- 5: Strong brand consistency; the banner design aligns well with the provided logo and conveys a recognizable brand identity
- 4: Mostly aligns, but minor refinements could improve brand consistency
- 3: Somewhat aligns, but noticeable inconsistencies exist (e.g., off-brand colors, incorrect typography)
- 2: Weak brand alignment, only the logo represents the brand while other design choices feel unrelated
- 1: No brand identity is reflected; the banner appears generic or disconnected from the brand

Justification Required:
- Are the colors and typography in line with the brand's usual style?
- Does the imagery and layout reinforce the brand's visual identity?
- Does the overall aesthetic feel like it belongs to the brand, or does it look generic?

## Adaptation for Blueprint Evaluation

The original eval.py evaluates rendered banner images. For HELM evaluation of
JSON blueprint outputs, the rubrics should be adapted to evaluate the design
specification rather than the visual output. Key adaptations:

- TAA: Evaluate whether the blueprint's element choices, copy text, and layout
  address the theme, audience, and purpose from the request
- LPS: Evaluate the logo position/size specifications in the blueprint
- AQS: Evaluate layout balance, spacing proportions, color harmony from specs
- CTAE: Evaluate the CTA text, position, and visual emphasis in the blueprint
- CPYQ: Evaluate the actual copy text (fully text-evaluable without rendering)
- BIS: Evaluate whether specified colors, fonts, and styling align with the
  brand described in the logo_description

## Notes

- Human-LLM correlation: Pearson 0.854-0.955 (all p < 0.001) between GPT-4o
  scoring and human perception (Paper Table 3)
- Original evaluation is multimodal (banner image + logo image + request text)
- Blueprint evaluation is an adaptation; CPYQ is the most directly evaluable
  dimension from text output alone
