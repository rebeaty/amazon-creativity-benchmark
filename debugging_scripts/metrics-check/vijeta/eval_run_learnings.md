# Eval Run Learnings

## cdat — SUCCESS
- **Attempts**: 0
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: cdat passed on the first attempt with no debugging required.

## cii_bench — SUCCESS
- **Attempts**: 0
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: cii_bench passed on the first attempt with no debugging required.

## dat — SUCCESS
- **Attempts**: 1
- **Root cause**: Cached request states reference local image files (e.g. `benchmark_output/scenarios/artinsight/images/*.jpeg`) that don't exist on disk, causing `MediaObject.__post_init__` assertion failures during deserialization.
- **Fix applied**: Downloaded/placed the missing artinsight images into `benchmark_output/scenarios/artinsight/images/` before re-running metrics.
- **Key learning**: For image-input datasets, ensure referenced image files exist at their cached paths before running metrics; HELM's codec will hard-assert on file existence during request-state loading.

## geo_story — SUCCESS
- **Attempts**: 2
- **Root cause**: Cached request states reference local artinsight image files (e.g. `benchmark_output/scenarios/artinsight/images/*.jpeg`) that don't exist on disk, causing `MediaObject.__post_init__` assertion failures during deserialization.
- **Fix applied**: Placed the missing artinsight images at their expected cached paths before re-running metrics.
- **Key learning**: Same image-path assertion pattern as `dat` — if geo_story shares cached request states that embed artinsight image references, ensure those files exist locally; this error will recur any time the benchmark_output cache is stale or moved.

## kiva — SUCCESS
- **Attempts**: 0
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: kiva passed on the first attempt with no debugging required.

## metaphor_generation — SUCCESS (session 1)
- **Attempts**: 0
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: metaphor_generation passed on the first attempt with no debugging required.

## metaphor_generation — SUCCESS (session 2)
- **Attempts**: 2
- **Root cause**: Cached request states embed references to artinsight image files (e.g. `benchmark_output/scenarios/artinsight/images/*.jpeg`) that don't exist locally, causing `MediaObject.__post_init__` assertion failures during HELM deserialization.
- **Fix applied**: Placed missing artinsight images at their expected cached paths before re-running metrics.
- **Key learning**: Same cross-dataset image-path contamination pattern as `dat`/`geo_story` — metaphor_generation is text-only but shares a benchmark_output cache that may embed image references from other datasets; ensure artinsight images are present whenever this error appears.

## miqa — SUCCESS
- **Attempts**: 0
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: miqa passed on the first attempt with no debugging required.

## oogiri_go — SUCCESS
- **Attempts**: 0
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: oogiri_go passed on the first attempt with no debugging required.

## proparalogy — SUCCESS
- **Attempts**: 0
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: proparalogy passed on the first attempt with no debugging required.

## riddlesense — SUCCESS
- **Attempts**: 0
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: riddlesense passed on the first attempt with no debugging required.

## story_quality — SUCCESS
- **Attempts**: 0
- **Root cause**: N/A
- **Fix applied**: N/A
- **Key learning**: story_quality passed on the first attempt with no debugging required.

## story_quality — SUCCESS (session 2)
- **Attempts**: 2
- **Root cause**: Cached request states embed references to artinsight image files (e.g. `benchmark_output/scenarios/artinsight/images/bulldog.jpeg`, `mandala.jpeg`, `pink_flower.jpeg`) that don't exist locally, causing `MediaObject.__post_init__` assertion failures during HELM deserialization.
- **Fix applied**: Placed missing artinsight images at their expected cached paths before re-running metrics.
- **Key learning**: Recurring cross-dataset image-path contamination — story_quality is text-only but its cached request states can embed artinsight image references; ensure artinsight images are present at `benchmark_output/scenarios/artinsight/images/` whenever this error appears.
