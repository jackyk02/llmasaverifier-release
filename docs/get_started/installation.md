# Installation

A pure-Python package (Python 3.9+, dependencies: `google-genai`, `openai`, `tqdm`):

```bash
pip install llm-verifier
```

To also self-host the verifier model with [vLLM](https://docs.vllm.ai), install the `vllm` extra (GPU required):

```bash
pip install "llm-verifier[vllm]"
```

Or from source — this also gives you the bundled benchmarks (`run.py`, `data/`, `criteria/`):

```bash
git clone https://github.com/llm-as-a-verifier/llm-as-a-verifier.git
cd llm-as-a-verifier
pip install -e .
```

## Set up a verifier backend

LLM-as-a-Verifier extracts **token-level logprobs** from the verifier model.
The client is picked automatically from the environment.

### Option 1: Gemini via Vertex AI (default)

The default verifier model is `gemini-2.5-flash`.
Put your Vertex AI API key in a `.env` file (or export it):

```bash
echo "VERTEX_API_KEY=your_key_here" > .env
```

### Option 2: OpenAI-compatible server (vLLM / SGLang)

Any OpenAI-compatible server that returns logprobs works — e.g. a local open model served by [vLLM](https://docs.vllm.ai):

```bash
vllm serve Qwen/Qwen3.5-9B --port 8000
export OPENAI_BASE_URL=http://localhost:8000/v1
```

`OPENAI_BASE_URL` takes precedence over `VERTEX_API_KEY`, and the served model is auto-detected — no `model=` argument needed (`OPENAI_API_KEY` is only needed for authenticated endpoints).
On this backend each `<score_A>` / `<score_B>` value is read by prefilling the tag with the score position constrained to the 20 scale letters, so the distribution stays calibrated even for models that don't reliably emit the tags — see [Logit-Restricted Frontier Models](../advanced_features/logit_restricted_models.md), which also covers frontier models that withhold logprobs entirely.

You can also pass a pre-built `openai` or `google-genai` client via the `client` argument; then no environment variable is needed.

<!--
## Verify the installation

Preview a bundled criteria file (no API key needed):

```bash
python -m llm_verifier swe_bench
```

Then run a first end-to-end selection (makes real verifier calls):

```python
import llm_verifier

result = llm_verifier.select(
    problem="Write a function that reverses a string.",
    candidates=["def rev(s): return s[::-1]", "def rev(s): return s"],
    criteria={"Correctness": "Does the code actually reverse the string?"},
)
print(result.index, result.scores)
```

`MissingAPIKeyError` means no credentials were found and no `client` was given — it is only raised when uncached comparisons actually need scoring, so fully cached runs work offline.
-->

