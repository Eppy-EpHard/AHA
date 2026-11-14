"""Run BFCL with the served checkpoint registered as a model.

Runs INSIDE the BFCL environment, by path, because that environment does not have afwsp installed:

  <bfcl-venv>/bin/python src/afwsp/eval/bfcl_entry.py generate --test-category ... [any BFCL flag]
  <bfcl-venv>/bin/python src/afwsp/eval/bfcl_entry.py evaluate --test-category ...

It adds MODEL to BFCL's model table, then hands the command line to BFCL unchanged. The model is BFCL's own
self-hosted Qwen thinking handler (QwenAgentThinkHandler, via Qwen-Agent), pointed at $AFWSP_BASE_URL (the logging
proxy in front of vLLM), with thinking on and the sampling in $AFWSP_SAMPLING. $AFWSP_BFCL_MODE selects how tools
reach the model:

  raw   tools go to the server natively, and vLLM's qwen3_coder parser reads the calls back. This is the Qwen3.5
        model card's Qwen-Agent setup (use_raw_api), and the default.
  nous  Qwen-Agent writes the tools into the prompt and parses calls itself; thinking stays in the content. This
        is BFCL's recipe for self-hosted Qwen3.

Registration happens at import, so processes BFCL spawns see the model too.
"""
import json
import os
import sys

# BFCL's base OpenAI handler builds a client at construction and the SDK refuses without a key; ours talks to the
# proxy through its own client, so a placeholder is enough when no real key is set.
os.environ.setdefault("OPENAI_API_KEY", "EMPTY")

from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING, ModelConfig  # noqa: E402
from bfcl_eval.model_handler.api_inference.qwen import QwenAgentThinkHandler  # noqa: E402
from qwen_agent.llm import get_chat_model  # noqa: E402

MODEL = "afwsp-qwen3.5-think-FC"
MODE = os.environ.get("AFWSP_BFCL_MODE", "raw")
SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5, "max_tokens": 32768,
            "max_input_tokens": 58000, **json.loads(os.environ.get("AFWSP_SAMPLING") or "{}")}


class ServedQwenThink(QwenAgentThinkHandler):
    def __init__(self, model_name, temperature, registry_name, is_fc_model, **kwargs):
        super().__init__(model_name, temperature, registry_name, is_fc_model, **kwargs)
        cfg = {"extra_body": {"chat_template_kwargs": {"enable_thinking": True}},
               "temperature": SAMPLING["temperature"], "top_p": SAMPLING["top_p"], "top_k": SAMPLING["top_k"],
               "repetition_penalty": 1.0, "presence_penalty": SAMPLING["presence_penalty"],
               "max_input_tokens": SAMPLING["max_input_tokens"], "timeout": 7200, "max_tokens": SAMPLING["max_tokens"],
               "seed": SAMPLING.get("seed", 0)}    # Qwen-Agent otherwise draws a random seed per call
        if MODE == "raw":
            cfg["use_raw_api"] = True
        else:
            cfg.update({"fncall_prompt_type": "nous", "thought_in_content": True})
        self.llm = get_chat_model({"model": model_name, "model_type": "oai", "api_key": "EMPTY",
                                   "model_server": os.environ["AFWSP_BASE_URL"], "generate_cfg": cfg})


MODEL_CONFIG_MAPPING[MODEL] = ModelConfig(
    model_name=os.environ.get("AFWSP_SERVED_NAME", "afwsp"), display_name=f"Qwen3.5 via afwsp ({MODE}, Think, FC)",
    url="https://huggingface.co/Qwen", org="Qwen", license="apache-2.0", model_handler=ServedQwenThink,
    input_price=None, output_price=None, is_fc_model=True, underscore_to_dot=False)


if __name__ == "__main__":
    from bfcl_eval.__main__ import cli
    args = sys.argv[1:]
    if "--model" not in args:
        args += ["--model", MODEL]
    cli(args)
