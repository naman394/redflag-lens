"""Label sentences with an open-weight LLM via vLLM (GPU), applying guide.md with the same system prompt as
llm_label.py. One sentence per request, JSON output constrained to the label schema, temperature 0.
Used for the v1.1 test-set extension: annotator B = Qwen2.5-32B-Instruct, tie-breaker C = Llama-3.1-70B-Instruct.
  python src/label/open_llm_label.py data/labels/test_ext.jsonl --model Qwen/Qwen2.5-32B-Instruct-AWQ --tag qwen32b [--ids ids.txt]
Writes data/labels/<stem>.<tag>.jsonl. The model never sees other annotators' labels.
"""
import argparse, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GUIDE = (ROOT / "data/annotations/guide.md").read_text()
SYSTEM = f"""You are an annotator for a dataset of sentences from Indian listed-company annual reports.
Apply the annotation guide below exactly. Label each sentence independently, using only the sentence
text and its section name. If is_risk = 0, category and specificity must be "None".

<guide>
{GUIDE}
</guide>"""
SCHEMA = {"type": "object", "properties": {
    "is_risk": {"type": "integer", "enum": [0, 1]},
    "category": {"type": "string", "enum": ["Financial", "Legal", "Regulatory", "Operational", "Business", "None"]},
    "specificity": {"type": "string", "enum": ["Specific", "Generic", "None"]}},
    "required": ["is_risk", "category", "specificity"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("infile"); ap.add_argument("--model", required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--ids"); ap.add_argument("--max-len", type=int, default=8192)
    a = ap.parse_args()
    from vllm import LLM, SamplingParams
    try:  # vLLM >= 0.11 renamed guided decoding to structured outputs
        from vllm.sampling_params import StructuredOutputsParams
        constrained = {"structured_outputs": StructuredOutputsParams(json=SCHEMA)}
    except ImportError:
        from vllm.sampling_params import GuidedDecodingParams
        constrained = {"guided_decoding": GuidedDecodingParams(json=SCHEMA)}

    rows = [json.loads(l) for l in open(a.infile)]
    if a.ids:
        keep = set(Path(a.ids).read_text().split())
        rows = [r for r in rows if r["sentence_id"] in keep]
    llm = LLM(a.model, max_model_len=a.max_len, gpu_memory_utilization=float(__import__("os").environ.get("VLLM_UTIL", "0.92")), enable_prefix_caching=True,
              enforce_eager=__import__("os").environ.get("VLLM_EAGER") == "1")
    sp = SamplingParams(temperature=0, max_tokens=60, **constrained)
    msgs = [[{"role": "system", "content": SYSTEM},
             {"role": "user", "content": f"Label this sentence.\nsection: {r['section']}\nsentence: {r['text']}"}] for r in rows]
    outs = llm.chat(msgs, sp)
    out_path = Path(a.infile).with_name(Path(a.infile).stem + f".{a.tag}.jsonl")
    bad = 0
    with open(out_path, "w") as f:
        for r, o in zip(rows, outs):
            try:
                d = json.loads(o.outputs[0].text)
            except Exception:
                d, bad = {"is_risk": 0, "category": "None", "specificity": "None"}, bad + 1
            if d["is_risk"] == 0:
                d["category"], d["specificity"] = "None", "None"
            f.write(json.dumps({"sentence_id": r["sentence_id"], **d, "annotator": a.model}) + "\n")
    print(f"{len(rows)} labelled -> {out_path} ({bad} unparsed), risk = {sum(json.loads(l)['is_risk'] for l in open(out_path))}")


if __name__ == "__main__":
    main()
