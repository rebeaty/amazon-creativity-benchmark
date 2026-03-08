# Metric Notes: FuxiBench

Source: Paper Section 4.2; `evaluate.py`, `src/utils.py`, `src/cipai_utils.py`

## Subset-Metric Mapping

| Subset | Metric | Type |
|--------|--------|------|
| ci_gen | pacc (poetry format accuracy) | Rule-based |
| couplet_gen | cacc (couplet format accuracy) | Rule-based |
| poem_gen | lacc (LLM accuracy) | LLM-as-judge |
| poem_nmt_inv | lacc | LLM-as-judge |
| poem_appre | bleu (SacreBLEU) | Automatic |

## PACC — Ci Poetry Format Accuracy (ci_gen)

Rule-based metric checking adherence to cipai (词牌) tonal patterns.

**Implementation** (`src/cipai_utils.py`):
1. `FormatEvaluator` loads 36+ cipai templates from `cipai2info.json`
2. Plus 4 classical formats (五言绝句, 七言绝句, 五言律诗, 七言律诗)
3. Each cipai defines expected segment count and character count per segment
4. `eval_single()` tries 3 variants: original text, without title, without title+author
5. Passes if ANY variant matches the prescribed template structure
6. Binary pass/fail per instance; aggregated as percentage

**What it checks:** Structural compliance (segment count, character count per segment)
**What it does NOT check:** Tonal correctness (pingze), rhyme, semantic quality

## CACC — Couplet Format Accuracy (couplet_gen)

Rule-based metric for parallel structure in couplets.

**Implementation** (`src/utils.py`, `couplet_format_acc()`):
1. Template field uses "C" characters to encode expected structure (e.g., "CCCCCCC" = 7 chars)
2. Segments separated by Chinese punctuation (，、。)
3. Checks: generated second line has same number of segments AND same character count per segment as the template
4. Binary pass/fail per instance

**What it checks:** Parallel structure (matching segment/character counts)
**What it does NOT check:** Semantic parallelism, antithesis quality, tonal opposition

## LACC — LLM-as-Judge Accuracy (poem_gen, poem_nmt_inv)

See `annotator_notes.md` for full judge configuration.

## BLEU (poem_appre)

Standard SacreBLEU with Chinese tokenization (jieba). Compares generated appreciation text against reference literary analysis.
