import argparse
import json
import os
import re
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.distributed as dist

from routing import route, TASK_KEY_TO_EXPERT, EXPERT_EN, EXPERT_RISK, EXPERT_CBT

EXPERT_CLA = "lora_cla"

def setup_ddp():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        lr = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(lr)
        return True, lr, int(os.environ["WORLD_SIZE"]), int(os.environ["RANK"])
    return False, 0, 1, 0


def cleanup_ddp(is_ddp):
    if is_ddp:
        dist.destroy_process_group()


def is_main(rank):
    return rank == 0


def resolve_stack(task_expert: str, has_cla: bool) -> List[str]:

    if task_expert == EXPERT_EN:
        return [EXPERT_EN]
    if has_cla:
        return [task_expert, EXPERT_CLA]
    return [task_expert]


def set_active_adapters(model, names):
    try:
        model.set_adapter(names if isinstance(names, list) else [names])
    except Exception:
        if isinstance(names, list):
            model.set_adapter(names[0])
        else:
            model.set_adapter(names)


def load_model(model_path, ia_ama, adapter_en, adapter_risk, adapter_cbt,
               adapter_cla, single_adapter, dtype, local_rank):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print(f"[rank {local_rank}] loading base LLM from LOCAL {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype,
        trust_remote_code=True, local_files_only=True,
    )
    model.to(f"cuda:{local_rank}")
    model.eval()

    has_cla = False
    if ia_ama:
        assert adapter_en and adapter_risk and adapter_cbt
        model = PeftModel.from_pretrained(model, adapter_en, adapter_name=EXPERT_EN)
        model.load_adapter(adapter_risk, adapter_name=EXPERT_RISK)
        model.load_adapter(adapter_cbt,  adapter_name=EXPERT_CBT)
        print(f"[rank {local_rank}] loaded 3 task expert adapters")
        if adapter_cla:
            model.load_adapter(adapter_cla, adapter_name=EXPERT_CLA)
            has_cla = True
            print(f"[rank {local_rank}] loaded CLA adapter (stacked inference)")
        else:
            print(f"[rank {local_rank}] CLA adapter NOT loaded (Stage-1-only inference)")
    elif single_adapter:
        model = PeftModel.from_pretrained(model, single_adapter)
        print(f"[rank {local_rank}] loaded single Mixed-SFT adapter")

    model.eval()
    return model, tokenizer, has_cla


SYS_ZH = ("你是一位心理健康分析专家。这是一个科研数据分析任务。"
          "请根据定义客观地对用户的帖子进行分类。\n\n"
          "输出格式要求：\n标签：[你的选项]\n理由：[你的分析]")
SYS_EN = ("You are a mental-health annotation assistant. "
          "Follow the user's instruction.")


def _cjk(text):
    return sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")


def build_prompt(sample):
    instruction = (sample.get("instruction", "") or "").strip()
    input_text = (sample.get("original_input", "") or sample.get("input", "") or "").strip()
    total = instruction + input_text
    is_en = _cjk(total) < 3
    sys_text = SYS_EN if is_en else SYS_ZH
    user_text = instruction if not input_text else (instruction + "\n\n" + input_text)
    return sys_text, user_text


@torch.no_grad()
def generate(model, tokenizer, sample, adapter_stack, max_new_tokens, device):
    if hasattr(model, "set_adapter") and adapter_stack:
        set_active_adapters(model, adapter_stack)

    sys_text, user_text = build_prompt(sample)
    messages = [{"role": "system", "content": sys_text},
                {"role": "user",   "content": user_text}]

    if hasattr(tokenizer, "apply_chat_template"):
        ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt",
        )
    else:
        raw = f"[SYSTEM]\n{sys_text}\n[USER]\n{user_text}\n[ASSISTANT]\n"
        ids = tokenizer(raw, return_tensors="pt", add_special_tokens=False).input_ids

    ids = ids.to(device)
    out = model.generate(
        ids, max_new_tokens=max_new_tokens, do_sample=False,
        temperature=1.0, pad_token_id=tokenizer.pad_token_id,
        repetition_penalty=1.2,
    )
    gen = out[0, ids.shape[1]:]
    return tokenizer.decode(gen, skip_special_tokens=True)


def extract_labels(text, label_space):
    text = text or ""
    m = re.search(r"(?:标签|label)\s*[:：]?\s*(.+?)(?:\n|理由|rationale|$)",
                  text, re.IGNORECASE | re.DOTALL)
    scan = m.group(1) if m else text
    hits = []
    ordered = sorted(label_space, key=lambda s: -len(s))
    for lab in ordered:
        if lab and lab in scan:
            if not any(lab in h or h in lab for h in hits):
                hits.append(lab)
    return hits


def multihot(labels, space):
    s = set(labels)
    return [1 if l in s else 0 for l in space]


TASK_METRIC = {
    "detect_depression": "weighted", "detect_stress": "weighted",
    "stress_cause": "weighted", "irf_factors": "weighted",
    "wellness_dims": "weighted",
    "cognitive_distortion": "micro", "cognitive_path": "micro",
    "suicide_coarse": "micro", "suicide_scale": "micro",
    "depression_level": "micro", "relation": "micro",
}


def compute_f1(y_true, y_pred, metric):
    from sklearn.metrics import f1_score
    return float(f1_score(y_true, y_pred, average=metric, zero_division=0))


def load_task_file(data_dir, task_key, split):
    for name in (f"{task_key}_{split}_prompts.json", f"{task_key}_{split}.json"):
        fp = os.path.join(data_dir, name)
        if os.path.exists(fp):
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
            for r in data:
                if not r.get("task_key"):
                    r["task_key"] = task_key
            return data
    return []


def load_label_spaces(labels_dir):
    m = {}
    for fn in os.listdir(labels_dir):
        if not fn.endswith(".json"):
            continue
        with open(os.path.join(labels_dir, fn), "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict) and "labels" in obj:
            obj = obj["labels"]
        m[fn[:-5]] = [str(x) for x in obj]
    return m


def evaluate_task(model, tokenizer, task_key, samples, label_space,
                  ia_ama, has_cla, is_ddp, world_size, rank,
                  device, out_dir, max_new_tokens):

    if ia_ama:
        task_expert = route({"task_key": task_key})
        stack = resolve_stack(task_expert, has_cla)
    else:
        task_expert = None
        stack = None

    if is_main(rank):
        print(f"\n=== Task: {task_key}  n={len(samples)}  stack={stack}  "
              f"metric={TASK_METRIC.get(task_key,'micro')} ===")

    my_idxs = list(range(rank, len(samples), world_size))
    my_results = []

    for i, idx in enumerate(my_idxs):
        sample = samples[idx]
        gen = generate(model, tokenizer, sample, stack, max_new_tokens, device)
        pred = extract_labels(gen, label_space)
        gold_text = sample.get("original_output", "") or ""
        gold = extract_labels(gold_text, label_space)
        my_results.append({
            "idx": idx, "task_key": task_key,
            "instruction": sample.get("instruction", ""),
            "gold_text": gold_text, "model_output": gen,
            "gold_labels": gold, "pred_labels": pred,
        })
        if is_main(rank) and (i + 1) % 20 == 0:
            print(f"  [rank 0 {i+1}/{len(my_idxs)}] pred={pred} gold={gold}")

    if is_ddp:
        gather = [None] * world_size
        dist.all_gather_object(gather, my_results)
        if is_main(rank):
            all_r = []
            for g in gather:
                all_r.extend(g)
            all_r.sort(key=lambda r: r["idx"])
        else:
            all_r = None
    else:
        all_r = sorted(my_results, key=lambda r: r["idx"])

    if not is_main(rank):
        return {}

    y_true = [multihot(r["gold_labels"], label_space) for r in all_r]
    y_pred = [multihot(r["pred_labels"], label_space) for r in all_r]
    if not any(any(row) for row in y_true):
        f1, metric_name = 0.0, "n/a"
    else:
        metric_name = TASK_METRIC.get(task_key, "micro")
        f1 = compute_f1(y_true, y_pred, metric_name)

    print(f"  [{task_key}] {metric_name}-F1 = {f1:.4f}")

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{task_key}_predictions.json"),
              "w", encoding="utf-8") as f:
        json.dump(all_r, f, ensure_ascii=False, indent=2)

    return {
        "task_key": task_key, "n": len(all_r),
        "metric": metric_name, "f1": f1,
        "task_expert": task_expert, "stack": stack,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--ia_ama", action="store_true")
    p.add_argument("--adapter_en",   type=str, default=None)
    p.add_argument("--adapter_risk", type=str, default=None)
    p.add_argument("--adapter_cbt",  type=str, default=None)
    p.add_argument("--adapter_cla",  type=str, default=None,
                   help="Optional CLA adapter for stacked inference. "
                        "Omit for Stage-1-only evaluation.")
    p.add_argument("--single_adapter", type=str, default=None)

    p.add_argument("--test_data_dir", required=True)
    p.add_argument("--labels_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--tasks", nargs="*", default=None)
    p.add_argument("--max_samples_per_task", type=int, default=None)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    args = p.parse_args()

    is_ddp, local_rank, world_size, global_rank = setup_ddp()
    device = f"cuda:{local_rank}"
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}

    if is_main(global_rank):
        print(f"[eval] is_ddp={is_ddp} world_size={world_size} rank={global_rank}")
        os.makedirs(args.out_dir, exist_ok=True)

    model, tokenizer, has_cla = load_model(
        model_path=args.model_path,
        ia_ama=args.ia_ama,
        adapter_en=args.adapter_en, adapter_risk=args.adapter_risk,
        adapter_cbt=args.adapter_cbt, adapter_cla=args.adapter_cla,
        single_adapter=args.single_adapter,
        dtype=dtype_map[args.dtype], local_rank=local_rank,
    )

    label_spaces = load_label_spaces(args.labels_dir)
    all_tasks = args.tasks or list(TASK_KEY_TO_EXPERT.keys())
    results = {}

    for tk in all_tasks:
        samples = load_task_file(args.test_data_dir, tk, split="test")
        if not samples:
            if is_main(global_rank):
                print(f"[SKIP] {tk}: no test file")
            continue
        if tk not in label_spaces:
            if is_main(global_rank):
                print(f"[SKIP] {tk}: no label space")
            continue
        if args.max_samples_per_task:
            samples = samples[:args.max_samples_per_task]

        res = evaluate_task(
            model, tokenizer, tk, samples, label_spaces[tk],
            ia_ama=args.ia_ama, has_cla=has_cla,
            is_ddp=is_ddp, world_size=world_size, rank=global_rank,
            device=device, out_dir=args.out_dir,
            max_new_tokens=args.max_new_tokens,
        )
        if is_main(global_rank) and res:
            results[tk] = res
        if is_ddp:
            dist.barrier()

    if is_main(global_rank):
        print("\n======== EVALUATION SUMMARY ========")
        f1_list = []
        for tk, r in results.items():
            print(f"  {tk:24s}  {r['metric']:>8s}-F1 = {r['f1']:.4f}  (n={r['n']})")
            f1_list.append(r["f1"])
        if f1_list:
            print(f"  {'AVG':24s}           = {np.mean(f1_list):.4f}")
        with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"[DONE] saved to {args.out_dir}")

    cleanup_ddp(is_ddp)


if __name__ == "__main__":
    main()
