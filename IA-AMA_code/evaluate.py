# -*- coding: utf-8 -*-


import os
import re
import json
import argparse
from typing import List, Dict, Any, Set, Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm

try:
    from sklearn.metrics import f1_score
except Exception:
    f1_score = None


# ===================== 0) UTILS =====================

def ensure_parent_dir(path: str):
    if not path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)


def load_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    assert os.path.exists(path), f"[ERROR] File not found: {path}"
    if path.endswith(".jsonl"):
        out = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out
    else:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict) and "data" in obj and isinstance(obj["data"], list):
            return obj["data"]
        raise ValueError("[ERROR] Unsupported json format. Expect list[...] or {'data':[...]} or jsonl.")


def load_labels(labels_json: str) -> List[str]:
    """
    labels_json can be:
    - a JSON list: ["label1", "label2", ...]
    - or a JSON dict: {"labels":[...]}
    """
    assert os.path.exists(labels_json), f"[ERROR] labels_json not found: {labels_json}"
    with open(labels_json, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return [str(x) for x in obj]
    if isinstance(obj, dict) and "labels" in obj and isinstance(obj["labels"], list):
        return [str(x) for x in obj["labels"]]
    raise ValueError("[ERROR] labels_json must be a list or {'labels': [...]}.")


# ===================== 1) TOKENIZER PATCH (IMPORTANT) =====================

def patch_tokenizer_for_glm4(tokenizer):
    """
    Fixes:
    - ChatGLM4Tokenizer._pad() does not support padding_side kwarg
      transformers may pass padding_side -> pop it
    - enforce left padding
    - ensure [PAD] token exists
    """
    if hasattr(tokenizer, "_pad"):
        orig_pad = tokenizer._pad

        def _pad_compat(*args, **kwargs):
            kwargs.pop("padding_side", None)
            return orig_pad(*args, **kwargs)

        tokenizer._pad = _pad_compat

    try:
        tokenizer.padding_side = "left"
    except Exception:
        pass

    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    return tokenizer


# ===================== 2) LABEL EXTRACTION (STRICT) =====================

def extract_labels_strict(text: str, label_space: List[str]) -> Set[str]:
    if not text:
        return set()
    t = text.strip()
    found = set()
    for lab in label_space:
        if lab in t:
            found.add(lab)
    return found


def to_multihot(label_set: Set[str], label_space: List[str]) -> List[int]:
    return [1 if lab in label_set else 0 for lab in label_space]


# ===================== 3) ROUTING =====================

def route_adapter(sample: Dict[str, Any], default_adapter: str = "lora_cbt") -> str:
    """
    Default is customizable via CLI.
    Heuristics:
      - if task_key == cognitive_distortion -> lora_cbt
      - if English letters present -> lora_en
      - if contains '认知' -> lora_cbt
      - else -> lora_risk
    """
    task_key = (sample.get("task_key", "") or "").lower()
    if task_key == "cognitive_distortion":
        return "lora_cbt"

    full = (sample.get("instruction", "") or "") + "\n" + (sample.get("constructed_prompt", "") or "")
    if re.search(r"[A-Za-z]", full):
        return "lora_en"

    if "认知" in full:
        return "lora_cbt"

    # fallback
    if default_adapter in {"lora_en", "lora_risk", "lora_cbt"}:
        return default_adapter
    return "lora_risk"


# ===================== 4) ADAPTER DIR RESOLVE =====================

def _has_adapter_config(folder: str) -> bool:
    return os.path.isfile(os.path.join(folder, "adapter_config.json"))


def resolve_adapter_dir(epoch_dir: str, adapter_name: str) -> str:
    root = os.path.abspath(os.path.join(epoch_dir, "adapters", adapter_name))
    if _has_adapter_config(root):
        return root

    nested = os.path.join(root, adapter_name)
    if _has_adapter_config(nested):
        return nested

    if os.path.isdir(root):
        for sub in os.listdir(root):
            subp = os.path.join(root, sub)
            if os.path.isdir(subp) and _has_adapter_config(subp):
                return subp

    raise FileNotFoundError(f"[ERROR] adapter_config.json not found for {adapter_name} under {root}")


# ===================== 5) LOAD MODEL + ADAPTERS =====================

def load_model_with_3_adapters(
    model_path: str,
    epoch_dir: str,
    device: str,
    dtype: torch.dtype,
    tokenizer,
    print_resolved: bool = True,
):
    en_dir = resolve_adapter_dir(epoch_dir, "lora_en")
    risk_dir = resolve_adapter_dir(epoch_dir, "lora_risk")
    cbt_dir = resolve_adapter_dir(epoch_dir, "lora_cbt")

    if print_resolved:
        print("[INFO] Resolved adapter dirs:")
        print("  lora_en  =", en_dir)
        print("  lora_risk=", risk_dir)
        print("  lora_cbt =", cbt_dir)

    base = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map=None,
    )

    # If tokenizer got new [PAD], resize embeddings
    if hasattr(base, "resize_token_embeddings"):
        base.resize_token_embeddings(len(tokenizer))

    if hasattr(base, "config"):
        base.config.use_cache = False

    model = PeftModel.from_pretrained(base, en_dir, adapter_name="lora_en", is_trainable=False)
    model.load_adapter(risk_dir, adapter_name="lora_risk", is_trainable=False)
    model.load_adapter(cbt_dir, adapter_name="lora_cbt", is_trainable=False)

    model = model.to(device)
    model.eval()
    return model


# ===================== 6) MANUAL DECODING =====================

@torch.no_grad()
def manual_decode(
    model,
    tokenizer,
    prompt: str,
    adapter_name: str,
    device: str,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
) -> str:
    if hasattr(model, "set_adapter"):
        model.set_adapter(adapter_name)

    enc = tokenizer(prompt, return_tensors="pt", padding=False, truncation=False)
    input_ids = enc["input_ids"].to(device)

    generated = input_ids
    eos_id = tokenizer.eos_token_id

    for _ in range(int(max_new_tokens)):
        attn = torch.ones_like(generated, dtype=torch.long, device=device)
        out = model(
            input_ids=generated,
            attention_mask=attn,
            use_cache=False,
            return_dict=True,
        )
        logits = out.logits[:, -1, :]  # [1, V]

        if do_sample:
            temp = max(float(temperature), 1e-6)
            probs = torch.softmax(logits / temp, dim=-1)

            # nucleus sampling
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cum = torch.cumsum(sorted_probs, dim=-1)
            mask = cum > float(top_p)
            mask[..., 0] = False
            sorted_probs[mask] = 0
            sorted_probs = sorted_probs / (sorted_probs.sum(dim=-1, keepdim=True) + 1e-12)

            next_id = torch.multinomial(sorted_probs, num_samples=1)
            next_token = sorted_idx.gather(-1, next_id)
        else:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)

        generated = torch.cat([generated, next_token], dim=1)

        if eos_id is not None and int(next_token.item()) == int(eos_id):
            break

    new_ids = generated[0, input_ids.shape[1]:]
    text = tokenizer.decode(new_ids, skip_special_tokens=True)
    return text.strip()


# ===================== 7) PROMPT BUILDING =====================

def build_prompt_from_sample(sample: Dict[str, Any]) -> str:
    """
    Priority:
    - constructed_prompt
    - else: instruction + original_input (or input)
    """
    prompt = (sample.get("constructed_prompt", "") or "").strip()
    if prompt:
        return prompt

    inst = (sample.get("instruction", "") or "").strip()
    inp = (sample.get("original_input", "") or sample.get("input", "") or "").strip()
    if inst and inp:
        return inst + "\n" + inp
    return inst or inp


# ===================== 8) MODES =====================

def run_interactive(
    model,
    tokenizer,
    label_space: List[str],
    device: str,
    default_adapter: str,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
):
    print("\n[Interactive Mode]")
    print("Enter text to classify. Type '/exit' to quit.")
    print("Optional: you can paste a JSON object with fields like instruction/constructed_prompt/task_key.\n")

    while True:
        user_in = input(">> ").strip()
        if not user_in:
            continue
        if user_in.lower() in {"/exit", "exit", "quit"}:
            break

        # Allow JSON input
        sample = {}
        if user_in.startswith("{") and user_in.endswith("}"):
            try:
                sample = json.loads(user_in)
            except Exception:
                sample = {"instruction": user_in}
        else:
            sample = {"instruction": user_in}

        prompt = build_prompt_from_sample(sample)
        used_adapter = route_adapter(sample, default_adapter=default_adapter)
        pred_text = manual_decode(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            adapter_name=used_adapter,
            device=device,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
        )

        pred_set = extract_labels_strict(pred_text, label_space)

        print("\n--- RESULT ---")
        print("adapter:", used_adapter)
        print("pred_labels:", sorted(list(pred_set)))
        print("raw_output:", pred_text)
        print("-------------\n")


def run_file_eval(
    model,
    tokenizer,
    label_space: List[str],
    device: str,
    test_path: str,
    save_pred_path: str,
    default_adapter: str,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    gold_field: str,
):
    assert test_path, "[ERROR] --test_path is required for file mode."
    assert save_pred_path, "[ERROR] --save_pred_path is required for file mode."

    ensure_parent_dir(save_pred_path)

    data = load_json_or_jsonl(test_path)
    print(f"[DATA] Loaded {len(data)} samples from: {test_path}")

    # Collect metrics only if gold exists and sklearn is available
    y_true, y_pred = [], []
    has_any_gold = False

    with open(save_pred_path, "w", encoding="utf-8") as wf:
        for sample in tqdm(data, desc="Testing"):
            prompt = build_prompt_from_sample(sample)
            used_adapter = route_adapter(sample, default_adapter=default_adapter)

            pred_text = manual_decode(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                adapter_name=used_adapter,
                device=device,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
            )

            pred_set = extract_labels_strict(pred_text, label_space)

            out = dict(sample)
            out["used_adapter"] = used_adapter
            out["model_output"] = pred_text
            out["pred_labels"] = sorted(list(pred_set))

            # Optional gold
            gold_text = (sample.get(gold_field, "") or "").strip()
            if gold_text:
                has_any_gold = True
                gold_set = extract_labels_strict(gold_text, label_space)
                out["gold_labels"] = sorted(list(gold_set))

                y_true.append(to_multihot(gold_set, label_space))
                y_pred.append(to_multihot(pred_set, label_space))

            wf.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"[SAVE] Predictions saved to: {os.path.abspath(save_pred_path)}")

    if has_any_gold and f1_score is not None:
        micro = f1_score(y_true, y_pred, average="micro", zero_division=0)
        print("\n========== MULTI-LABEL RESULTS ==========")
        print(f"Micro-F1: {micro:.6f}")
    elif has_any_gold and f1_score is None:
        print("[WARN] Gold labels exist but scikit-learn not installed. Skip Micro-F1.")
        print("       Install with: pip install scikit-learn")
    else:
        print("[INFO] No gold field found in samples; skipped metric computation.")


# ===================== 9) MAIN =====================

def parse_args():
    p = argparse.ArgumentParser("IA-AMoE Test / Inference (Sanitized)")

    p.add_argument("--mode", type=str, choices=["interactive", "file"], required=True)

    # model
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--epoch_dir", type=str, required=True)
    p.add_argument("--local_files_only", action="store_true")

    # label space
    p.add_argument("--labels_json", type=str, required=True, help="JSON list or {'labels':[...]}")

    # device
    p.add_argument("--device", type=str, default=None, help="cuda/cpu; default auto")
    p.add_argument("--gpu_id", type=int, default=None, help="Optional: set CUDA_VISIBLE_DEVICES")

    # decoding
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--do_sample", type=str, default="false", help="true/false")
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--top_p", type=float, default=0.9)

    # routing
    p.add_argument("--default_adapter", type=str, default="lora_cbt", choices=["lora_en", "lora_risk", "lora_cbt"])

    # file mode IO
    p.add_argument("--test_path", type=str, default=None)
    p.add_argument("--save_pred_path", type=str, default=None)
    p.add_argument("--gold_field", type=str, default="original_output",
                   help="Field name in samples that contains gold text to extract labels from (optional).")

    return p.parse_args()


def main():
    args = parse_args()

    # device + gpu selection
    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32

    do_sample = str(args.do_sample).lower() in {"1", "true", "yes", "y"}

    print(f"[INFO] device={device}, CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}")
    print(f"[INFO] model_path={args.model_path}")
    print(f"[INFO] epoch_dir={args.epoch_dir}")

    label_space = load_labels(args.labels_json)
    print(f"[INFO] label_space size={len(label_space)}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=bool(args.local_files_only),
    )
    tokenizer = patch_tokenizer_for_glm4(tokenizer)

    model = load_model_with_3_adapters(
        model_path=args.model_path,
        epoch_dir=args.epoch_dir,
        device=device,
        dtype=dtype,
        tokenizer=tokenizer,
        print_resolved=True,
    )

    if args.mode == "interactive":
        run_interactive(
            model=model,
            tokenizer=tokenizer,
            label_space=label_space,
            device=device,
            default_adapter=args.default_adapter,
            max_new_tokens=args.max_new_tokens,
            do_sample=do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
        )
    else:
        run_file_eval(
            model=model,
            tokenizer=tokenizer,
            label_space=label_space,
            device=device,
            test_path=args.test_path,
            save_pred_path=args.save_pred_path,
            default_adapter=args.default_adapter,
            max_new_tokens=args.max_new_tokens,
            do_sample=do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            gold_field=args.gold_field,
        )


if __name__ == "__main__":
    main()
