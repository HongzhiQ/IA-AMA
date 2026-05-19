import argparse
import json
import os
from typing import Dict, List

import numpy as np
import torch

from routing import route, EXPERT_RISK, RISK_TASKS
from calibration_metrics import (
    TemperatureScaler,
    expected_calibration_error,
    maximum_calibration_error,
    brier_score,
    accuracy,
    risk_coverage_curve,
    risk_coverage_auc,
    selective_f1_at_coverage,
)
from llm_classifier import LLMClassifier


def load_json_any(path: str) -> List[dict]:
    assert os.path.exists(path), f"Not found: {path}"
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith(".jsonl"):
            return [json.loads(line) for line in f if line.strip()]
        return json.load(f)


def load_label_spaces(labels_dir: str) -> Dict[str, List[str]]:
    mapping = {}
    for fn in os.listdir(labels_dir):
        if not fn.endswith(".json"):
            continue
        task = fn[:-5]
        with open(os.path.join(labels_dir, fn), "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict) and "labels" in obj:
            obj = obj["labels"]
        mapping[task] = [str(x) for x in obj]
    return mapping


def filter_by_task(samples: List[dict], task_key: str) -> List[dict]:
    return [s for s in samples if (s.get("task_key") or "").strip() == task_key]


def extract_gold_label_idx(sample: dict, label_space: List[str]) -> int:

    text = sample.get("original_output", "") or ""
    ordered = sorted(enumerate(label_space), key=lambda x: -len(x[1]))
    for idx, lab in ordered:
        if lab and lab in text:
            return idx
    return -1


def run_inference_on_task(clf: LLMClassifier,
                          samples: List[dict],
                          label_space: List[str],
                          ia_ama: bool,
                          task_key: str):

    if ia_ama:
        task_expert = route({"task_key": task_key}, default=EXPERT_RISK)
        clf.set_stack(task_expert)
        stack_info = [task_expert, "lora_cla"] if clf.has_cla and task_expert != "lora_en" else [task_expert]
        print(f"    [stack] {task_key} -> {stack_info}")

    probs_list = []
    label_list = []
    n_skipped = 0
    for i, sample in enumerate(samples):
        gold = extract_gold_label_idx(sample, label_space)
        if gold < 0:
            n_skipped += 1
            continue
        p = clf.classify(sample, label_space)
        probs_list.append(p)
        label_list.append(gold)
        if (i + 1) % 50 == 0:
            print(f"    [{i+1}/{len(samples)}]  kept={len(probs_list)}  skipped={n_skipped}")

    if not probs_list:
        return np.zeros((0, len(label_space))), np.zeros((0,), dtype=np.int64)
    return np.stack(probs_list, axis=0), np.array(label_list, dtype=np.int64)


def evaluate_task(clf: LLMClassifier,
                  val_samples: List[dict],
                  test_samples: List[dict],
                  label_space: List[str],
                  ia_ama: bool,
                  task_key: str,
                  out_dir: str) -> dict:

    print(f"\n=== Task: {task_key}  |  |labels|={len(label_space)} ===")
    print(f"  val={len(val_samples)}  test={len(test_samples)}")

    print("  [1/3] inference on VAL ...")
    val_probs, val_labels = run_inference_on_task(
        clf, val_samples, label_space, ia_ama, task_key
    )
    if len(val_labels) == 0:
        print("  [!] no parsable val samples -> skip task")
        return {"task_key": task_key, "error": "no parsable val samples"}


    val_logits = torch.tensor(np.log(val_probs + 1e-12), dtype=torch.float32)
    val_labels_t = torch.tensor(val_labels, dtype=torch.long)

    print("  [2/3] fitting temperature scaler ...")
    ts = TemperatureScaler(init_T=1.5)
    T_hat = ts.fit(val_logits, val_labels_t, max_iter=100)
    print(f"    learned T = {T_hat:.4f}")

    print("  [3/3] inference on TEST ...")
    test_probs, test_labels = run_inference_on_task(
        clf, test_samples, label_space, ia_ama, task_key
    )
    if len(test_labels) == 0:
        print("  [!] no parsable test samples -> skip task")
        return {"task_key": task_key, "error": "no parsable test samples"}


    test_logits = torch.tensor(np.log(test_probs + 1e-12), dtype=torch.float32)
    test_logits_cal = ts(test_logits).detach().numpy()

    test_probs_cal = np.exp(test_logits_cal - test_logits_cal.max(axis=1, keepdims=True))
    test_probs_cal = test_probs_cal / test_probs_cal.sum(axis=1, keepdims=True)


    results = {
        "task_key": task_key,
        "n_val":  int(len(val_labels)),
        "n_test": int(len(test_labels)),
        "temperature": T_hat,
        "accuracy": accuracy(test_probs, test_labels),
        "uncalibrated": {
            "ECE":    expected_calibration_error(test_probs, test_labels),
            "MCE":    maximum_calibration_error(test_probs, test_labels),
            "Brier":  brier_score(test_probs, test_labels),
            "RC_AUC": risk_coverage_auc(test_probs, test_labels),
        },
        "calibrated_TS": {
            "ECE":    expected_calibration_error(test_probs_cal, test_labels),
            "MCE":    maximum_calibration_error(test_probs_cal, test_labels),
            "Brier":  brier_score(test_probs_cal, test_labels),
            "RC_AUC": risk_coverage_auc(test_probs_cal, test_labels),
        },
    }


    results["selective"] = {}
    for cov in (0.7, 0.8, 0.85, 0.9, 0.95, 1.0):
        f1, tau = selective_f1_at_coverage(test_probs_cal, test_labels, cov)
        results["selective"][f"cov={cov}"] = {"macroF1": f1, "threshold": tau}


    os.makedirs(out_dir, exist_ok=True)
    cov_arr, risk_arr = risk_coverage_curve(test_probs_cal, test_labels, n_points=100)
    np.savez(
        os.path.join(out_dir, f"{task_key}_rc.npz"),
        coverage=cov_arr, risk=risk_arr,
        probs_uncal=test_probs,
        probs_cal=test_probs_cal,
        labels=test_labels,
        temperature=np.array([T_hat]),
    )

    def fmt(d):
        return " ".join(f"{k}={v:.4f}" for k, v in d.items())
    print(f"  [uncal] {fmt(results['uncalibrated'])}")
    print(f"  [ +TS ] {fmt(results['calibrated_TS'])}")
    print(f"  selF1@0.85 = {results['selective']['cov=0.85']['macroF1']:.4f}  "
          f"(tau={results['selective']['cov=0.85']['threshold']:.4f})")
    return results


def main():
    p = argparse.ArgumentParser()
    # model
    p.add_argument("--backbone", choices=["glm", "llama", "qwen", "auto"], default="auto")
    p.add_argument("--model_path", required=True)
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    # adapter mode
    p.add_argument("--ia_ama", action="store_true",
                   help="If set, load 3 LoRA adapters and route by task_key")
    p.add_argument("--adapter_en",   type=str, default=None)
    p.add_argument("--adapter_risk", type=str, default=None)
    p.add_argument("--adapter_cbt",  type=str, default=None)
    p.add_argument("--adapter_cla",  type=str, default=None,
                   help="Optional shared CLA adapter for stacked IA-AMA+SCLA inference. "
                        "Omit for Stage-1-only evaluation.")
    p.add_argument("--single_adapter", type=str, default=None,
                   help="For baselines: a single LoRA path")
    # data
    p.add_argument("--val_file",   required=True,
                   help="Val JSON file (may contain mixed task_keys)")
    p.add_argument("--test_file",  required=True)
    p.add_argument("--labels_dir", required=True)
    p.add_argument("--out_dir",    required=True)
    p.add_argument("--tasks", nargs="*", default=None,
                   help="Subset of risk task_keys (default = all 4)")
    p.add_argument("--max_val",  type=int, default=None,
                   help="Optionally cap val samples per task (for speed)")
    p.add_argument("--max_test", type=int, default=None,
                   help="Optionally cap test samples per task")
    args = p.parse_args()

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dtype = dtype_map[args.dtype]


    adapter_paths = None
    if args.ia_ama:
        assert args.adapter_en and args.adapter_risk and args.adapter_cbt, \
            "--ia_ama requires --adapter_en / --adapter_risk / --adapter_cbt"
        adapter_paths = {
            "lora_en":   args.adapter_en,
            "lora_risk": args.adapter_risk,
            "lora_cbt":  args.adapter_cbt,
        }

    clf = LLMClassifier.from_pretrained(
        backbone=args.backbone,
        model_path=args.model_path,
        adapter_paths=adapter_paths,
        single_adapter_path=args.single_adapter,
        adapter_cla_path=args.adapter_cla,
        dtype=dtype,
    )

    val_all  = load_json_any(args.val_file)
    test_all = load_json_any(args.test_file)
    label_spaces = load_label_spaces(args.labels_dir)

    tasks = args.tasks or RISK_TASKS
    print(f"[calibration_eval] running tasks: {tasks}")

    all_results = {}
    for tk in tasks:
        val_samples  = filter_by_task(val_all,  tk)
        test_samples = filter_by_task(test_all, tk)
        if args.max_val:
            val_samples = val_samples[:args.max_val]
        if args.max_test:
            test_samples = test_samples[:args.max_test]

        if not val_samples or not test_samples:
            print(f"[SKIP] {tk}: val={len(val_samples)}, test={len(test_samples)}")
            continue
        if tk not in label_spaces:
            print(f"[SKIP] {tk}: no label_space file in {args.labels_dir}")
            continue

        res = evaluate_task(
            clf,
            val_samples, test_samples,
            label_spaces[tk],
            ia_ama=args.ia_ama,
            task_key=tk,
            out_dir=args.out_dir,
        )
        all_results[tk] = res

    if all_results:
        avg = {"uncalibrated": {}, "calibrated_TS": {}}
        for group in ("uncalibrated", "calibrated_TS"):
            for metric in ("ECE", "MCE", "Brier", "RC_AUC"):
                vals = [r[group][metric] for r in all_results.values()
                        if isinstance(r.get(group), dict)]
                avg[group][metric] = float(np.mean(vals)) if vals else None
        all_results["__AVG__"] = avg
        print("\n=== AVERAGE over tasks ===")
        print("  [uncal] " + " ".join(f"{k}={v:.4f}" for k, v in avg["uncalibrated"].items()))
        print("  [ +TS ] " + " ".join(f"{k}={v:.4f}" for k, v in avg["calibrated_TS"].items()))

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "calibration_results.json"),
              "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n[DONE] Results saved to {args.out_dir}")


if __name__ == "__main__":
    main()
