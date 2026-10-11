# artifacts/ (shipped judge cache)

This folder holds the notebook's on-disk cache. Each `call_<hash>.json` file is one
model response, keyed by a hash of the exact request `(model, temperature, prompt)`.
When `cached_call` in the notebook sees a matching file, it returns the saved reply
and makes no network request.

**Why these files are committed.** Sections 7 and 9 use an LLM as a judge, which
needs the OpenAI API. Most learners run this tutorial without a key. So we ship the
real judge responses for the exact prompts those sections send. That way the
calibration numbers you see (the separation gap, the correlation with human ratings)
are genuine model output, not placeholders, and they appear with zero setup.

**These numbers are pinned to one model version.** They were produced by the default
`JUDGE_MODEL` at the time this module was written. That is the whole point of section
9's caution: a judge is calibrated against one model version, and the day the model
behind the endpoint changes, the numbers move. If you set your own key and change the
prompt or the model, `cached_call` will make fresh calls and write new files here.

**To regenerate from scratch:** delete this folder's `call_*.json` files, set your
`OPENAI_API_KEY` in `.env`, and re-run the notebook. Every unique request is paid for
once and then cached again.
