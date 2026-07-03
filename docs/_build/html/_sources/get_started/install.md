# Install LLM-as-a-Verifier

You can install LLM-as-a-Verifier using one of the methods below.
The framework is a pure-Python package with two runtime dependencies (`google-genai` and `tqdm`) and requires Python 3.9+.

## Method 1: With pip or uv

```bash
pip install llm-verifier
```

It is recommended to use uv for faster installation:

```bash
pip install --upgrade pip
pip install uv
uv pip install llm-verifier
```

## Method 2: From source

```bash
git clone https://github.com/llm-as-a-verifier/llm-as-a-verifier.git
cd llm-as-a-verifier

pip install --upgrade pip
pip install -e .
```

Installing from source also gives you the bundled benchmarks (`run.py`, `data/`, `criteria/`) described in [Running the Bundled Benchmarks](../benchmarks/running_benchmarks.md).

## Set up a verifier backend

LLM-as-a-Verifier extracts **token-level logprobs** from the verifier model.
Two backends are supported; the client is picked automatically from the environment.

### Option 1: Gemini via Vertex AI (default)

The default verifier model is `gemini-2.5-flash`.
Create a `.env` file in your working directory with your Vertex AI API key:

```bash
echo "VERTEX_API_KEY=your_key_here" > .env
```

Alternatively, export it as an environment variable:

```bash
export VERTEX_API_KEY=your_key_here
```

### Option 2: OpenAI-compatible server (vLLM / SGLang)

Any OpenAI-compatible server that returns logprobs works as a verifier backend — e.g. a local open model served by [vLLM](https://docs.vllm.ai):

```bash
vllm serve Qwen/Qwen3.5-9B --port 8000
export OPENAI_BASE_URL=http://localhost:8000/v1
```

When `OPENAI_BASE_URL` is set it takes precedence over `VERTEX_API_KEY`, and the served model is auto-detected — `select` / `compare` / `track` work without a `model=` argument.
`OPENAI_API_KEY` is only needed for authenticated endpoints (any string works for a local vLLM server).

On this backend the verifier first generates its analysis, then each `<score_A>` / `<score_B>` value is read by **prefilling the tag** (`continue_final_message`) with the position constrained to the 20 scale letters via structured outputs — so the extracted distribution is the renormalized belief over the scale itself, even for models that don't reliably emit the tags.

You can also pass a pre-built `openai` or `google-genai` client to any API call via the `client` argument, in which case no environment variable is needed.

```{note}
If you want to use a frontier model that withholds logprobs (e.g., GPT-5.5 or Claude Opus) as the verifier, see [Logit-Restricted Frontier Models](../advanced_features/logit_restricted_models.md) for a two-stage workaround.
```

## Verify the installation

Preview a bundled criteria file exactly as the verifier will see it — this needs no API key:

```bash
python -m llm_verifier swe_bench
```

Then run a first end-to-end selection (this makes real verifier calls and requires `VERTEX_API_KEY`):

```python
import llm_verifier

result = llm_verifier.select(
    problem="Write a function that reverses a string.",
    candidates=["def rev(s): return s[::-1]", "def rev(s): return s"],
    criteria={"Correctness": "Does the code actually reverse the string?"},
)
print(result.index, result.scores)
```

## Quick fixes to common problems

- **`MissingAPIKeyError`**: no credentials were found and no `client` was given. Set `VERTEX_API_KEY` in a `.env` file or in your environment. Note that the error is only raised when uncached comparisons actually need scoring — fully cached runs work offline.
- **Logprobs are empty or scores look degenerate**: make sure your key is a Vertex AI key. Keys for APIs that do not expose logprobs cannot be used for the continuous reward.
- **Benchmark runs need extra packages**: `pip install google-genai tqdm` (installed automatically with the package, but required if you copied `run.py` standalone).
