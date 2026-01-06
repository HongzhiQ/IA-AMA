# -*- coding: utf-8 -*-


import os
import json
import math
import argparse
from itertools import cycle
from contextlib import nullcontext
from typing import List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from peft import get_peft_model, LoraConfig, TaskType
from tqdm import tqdm


# ---------------------------
# 1) ROUTING
# ---------------------------

DEFAULT_KEYWORDS_RISK = [
    # zh
    "自杀", "自残", "轻生", "想死", "结束生命", "活不下去", "危机", "伤害",
    # en
    "suicide", "kill myself", "die", "harm", "risk"
]


def get_routing_decision(instruction: str, input_text: str = "") -> str:

    instruction = instruction or ""
    has_en = any('a' <= ch.lower() <= 'z' for ch in instruction)
    if has_en:
        return "lora_en"
    if "认知" in instruction:
        return "lora_cbt"
    return "lora_risk"


def get_align_zh_adapter(text: str, risk_keywords: List[str]) -> str:

    t = text or ""
    for kw in risk_keywords:
        if kw in t:
            return "lora_risk"
    return "lora_cbt"


# ---------------------------
# 2) MODULES
# ---------------------------

class ConceptAlignmentLayer(nn.Module):
    def __init__(self, hidden_size: int, bottleneck_size: int):
        super().__init__()
        self.w_down = nn.Linear(hidden_size, bottleneck_size)
        self.relu = nn.ReLU()
        self.w_up = nn.Linear(bottleneck_size, hidden_size)

    def forward(self, h):
        return h + self.w_up(self.relu(self.w_down(h)))


class IA_AMoE_GLM(nn.Module):
    def __init__(
        self,
        model_path: str,
        lora_targets: List[str],
        hidden_size: int,
        bottleneck_size: int,
        lora_en_cfg: Dict[str, Any],
        lora_risk_cfg: Dict[str, Any],
        lora_cbt_cfg: Dict[str, Any],
        device: str,
    ):
        super().__init__()
        self.device_str = device

        print(f"[INFO] Loading base model from: {model_path}")

        if device == "cuda":
            use_bf16 = torch.cuda.is_bf16_supported()
            self.compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
        else:
            self.compute_dtype = torch.float32

        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=self.compute_dtype,
            output_hidden_states=True,
        )

        if hasattr(self.base_model, "config"):
            self.base_model.config.use_cache = False

        # Keep checkpointing enabled
        self.base_model.gradient_checkpointing_enable()
        self.base_model.enable_input_require_grads()

        self.adapter_en = "lora_en"
        self.adapter_risk = "lora_risk"
        self.adapter_cbt = "lora_cbt"

        peft_config_en = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=int(lora_en_cfg["r"]),
            lora_alpha=float(lora_en_cfg["alpha"]),
            lora_dropout=float(lora_en_cfg["dropout"]),
            target_modules=lora_targets,
        )
        peft_config_risk = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=int(lora_risk_cfg["r"]),
            lora_alpha=float(lora_risk_cfg["alpha"]),
            lora_dropout=float(lora_risk_cfg["dropout"]),
            target_modules=lora_targets,
        )
        peft_config_cbt = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=int(lora_cbt_cfg["r"]),
            lora_alpha=float(lora_cbt_cfg["alpha"]),
            lora_dropout=float(lora_cbt_cfg["dropout"]),
            target_modules=lora_targets,
        )

        self.base_model = get_peft_model(self.base_model, peft_config_en, adapter_name=self.adapter_en)
        self.base_model.add_adapter(self.adapter_risk, peft_config_risk)
        self.base_model.add_adapter(self.adapter_cbt, peft_config_cbt)

        self.alignment_layer = ConceptAlignmentLayer(hidden_size, bottleneck_size).to(self.compute_dtype)

        print("[INFO] IA-AMoE Initialized.")
        print(f"      - {self.adapter_en} (Anchor)    : r={lora_en_cfg['r']}")
        print(f"      - {self.adapter_risk} (Risk)    : r={lora_risk_cfg['r']}")
        print(f"      - {self.adapter_cbt} (Cognitive): r={lora_cbt_cfg['r']}")

    def _safe_set_adapter(self, adapter_name: str):
        if hasattr(self.base_model, "set_adapter"):
            self.base_model.set_adapter(adapter_name)

    def forward_sft(self, input_ids, attention_mask, labels, adapter_name: str):
        self._safe_set_adapter(adapter_name)
        out = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,  # must be LONG
            labels=labels,
        )
        return out.loss

    def get_embedding(self, input_ids, attention_mask, adapter_name: str):
        self._safe_set_adapter(adapter_name)
        out = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,  # must be LONG
            output_hidden_states=True,
        )
        last_hidden = out.hidden_states[-1]  # [B, T, H]
        h = last_hidden[:, -1, :]           # left padding -> last token pooling
        z = self.alignment_layer(h)
        return z


# ---------------------------
# 3) DATASETS
# ---------------------------

class MentalSFTDataset(Dataset):
    def __init__(self, json_path: str, tokenizer, max_length: int):
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        print(f"[DATA] Loaded SFT examples: {len(self.data)}")

        self.zh_sys = "你是一名心理健康标注助理。请严格遵循用户指令完成任务。"
        self.en_sys = "You are a mental-health annotation assistant. Follow the user's instruction."
        self.has_chat_template = hasattr(tokenizer, "apply_chat_template")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def _build_ids(self, system_text: str, user_text: str, assistant_text: str):
        if self.has_chat_template:
            messages_prompt = [
                {"role": "system", "content": system_text},
                {"role": "user", "content": user_text},
            ]
            messages_full = [
                {"role": "system", "content": system_text},
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": assistant_text},
            ]
            prompt_ids = self.tokenizer.apply_chat_template(
                messages_prompt, tokenize=True, add_generation_prompt=True
            )
            full_ids = self.tokenizer.apply_chat_template(
                messages_full, tokenize=True, add_generation_prompt=False
            )
            return prompt_ids, full_ids

        prompt_text = f"[SYSTEM]\n{system_text}\n[USER]\n{user_text}\n[ASSISTANT]\n"
        full_text = prompt_text + assistant_text
        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full_ids = self.tokenizer(full_text, add_special_tokens=False)["input_ids"]
        return prompt_ids, full_ids

    def collate_fn(self, batch):
        input_ids_list, labels_list, adapters = [], [], []

        for item in batch:
            instruction = (item.get("instruction", "") or "").strip()
            input_context = (item.get("input", "") or "").strip()
            output_text = (item.get("output", "") or "").strip()

            adapter_name = get_routing_decision(instruction, input_context)
            adapters.append(adapter_name)

            is_zh = (adapter_name != "lora_en")
            sys_text = self.zh_sys if is_zh else self.en_sys
            user_text = instruction if not input_context else (instruction + "\n\n" + input_context)

            prompt_ids, full_ids = self._build_ids(sys_text, user_text, output_text)

            if len(full_ids) > self.max_length:
                full_ids = full_ids[:self.max_length]
            if len(prompt_ids) > self.max_length:
                prompt_ids = prompt_ids[:self.max_length]

            labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
            labels = labels[:len(full_ids)]

            input_ids_list.append(torch.tensor(full_ids, dtype=torch.long))
            labels_list.append(torch.tensor(labels, dtype=torch.long))

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels_list, batch_first=True, padding_value=-100
        )
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id).long()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "adapters": adapters,
        }


class AlignmentDataset(Dataset):
    def __init__(self, json_path: str, tokenizer, max_length: int, risk_keywords: List[str]):
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.risk_keywords = risk_keywords
        print(f"[DATA] Loaded alignment pairs: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def collate_fn(self, batch):
        zh_texts = [it["zh"] for it in batch]
        en_texts = [it["en"] for it in batch]
        zh_adapters = [get_align_zh_adapter(t, self.risk_keywords) for t in zh_texts]

        zh_enc = self.tokenizer(
            zh_texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
        )
        en_enc = self.tokenizer(
            en_texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
        )

        if "attention_mask" in zh_enc:
            zh_enc["attention_mask"] = zh_enc["attention_mask"].long()
        if "attention_mask" in en_enc:
            en_enc["attention_mask"] = en_enc["attention_mask"].long()

        return {"zh_input": zh_enc, "en_input": en_enc, "zh_adapters": zh_adapters}


# ---------------------------
# 4) LOSSES
# ---------------------------

def compute_infonce_loss(z_zh, z_en, temperature: float):
    z_zh = F.normalize(z_zh, dim=1, eps=1e-8)
    z_en = F.normalize(z_en, dim=1, eps=1e-8)
    logits = torch.matmul(z_zh, z_en.t()) / float(temperature)
    labels = torch.arange(logits.size(0), device=logits.device)
    return F.cross_entropy(logits, labels)


# ---------------------------
# 5) HELPERS
# ---------------------------

def patch_tokenizer_for_glm4(tokenizer):
    # force left padding (GLM _pad asserts it)
    try:
        tokenizer.padding_side = "left"
    except Exception:
        pass

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    # patch _pad to ignore unexpected kwargs
    if hasattr(tokenizer, "_pad"):
        orig_pad = tokenizer._pad

        def _pad_compat(*args, **kwargs):
            kwargs.pop("padding_side", None)
            return orig_pad(*args, **kwargs)

        tokenizer._pad = _pad_compat

    return tokenizer


def save_checkpoint(model: IA_AMoE_GLM, tokenizer, output_dir: str, epoch_idx: int):
    epoch_dir = os.path.join(output_dir, f"epoch_{epoch_idx + 1:02d}")
    os.makedirs(epoch_dir, exist_ok=True)
    print(f"\n[SAVE] Saving checkpoint to {epoch_dir}...")

    torch.save(model.alignment_layer.state_dict(), os.path.join(epoch_dir, "concept_alignment_layer.pth"))
    tokenizer.save_pretrained(epoch_dir)

    adapters_dir = os.path.join(epoch_dir, "adapters")
    os.makedirs(adapters_dir, exist_ok=True)

    for adp in ["lora_en", "lora_risk", "lora_cbt"]:
        out_dir = os.path.join(adapters_dir, adp)
        os.makedirs(out_dir, exist_ok=True)
        try:
            model.base_model.save_pretrained(out_dir, selected_adapters=[adp], safe_serialization=True)
        except TypeError:
            model.base_model.set_adapter(adp)
            model.base_model.save_pretrained(out_dir, safe_serialization=True)

    print("[SAVE] Done.")


def load_config(args: argparse.Namespace) -> Dict[str, Any]:
    # Priority: JSON config overrides CLI where provided (but CLI can still override after load if you want)
    cfg = {}
    if args.config_json:
        with open(args.config_json, "r", encoding="utf-8") as f:
            cfg = json.load(f)

    # Helper to get value from CLI if not None else from cfg else None
    def pick(name, default=None):
        v = getattr(args, name, None)
        if v is not None:
            return v
        return cfg.get(name, default)

    config = {
        "model_path": pick("model_path"),
        "mental_data_path": pick("mental_data_path"),
        "align_data_path": pick("align_data_path"),
        "output_dir": pick("output_dir", "./output"),

        "max_length": int(pick("max_length")),
        "align_max_len": int(pick("align_max_len")),

        "align_every": int(pick("align_every")),
        "batch_size": int(pick("batch_size")),
        "grad_accum_steps": int(pick("grad_accum_steps")),

        "lr": float(pick("lr")),
        "epochs": int(pick("epochs")),
        "max_grad_norm": float(pick("max_grad_norm")),
        "warmup_ratio": float(pick("warmup_ratio")),
        "weight_decay": float(pick("weight_decay")),

        "alpha_sft": float(pick("alpha_sft")),
        "alpha_align": float(pick("alpha_align")),
        "temperature": float(pick("temperature")),

        "hidden_size": int(pick("hidden_size")),
        "bottleneck_size": int(pick("bottleneck_size")),

        "lora_targets": pick("lora_targets"),
        "risk_keywords": pick("risk_keywords", DEFAULT_KEYWORDS_RISK),

        "local_files_only": bool(pick("local_files_only", False)),
        "device": pick("device", "cuda" if torch.cuda.is_available() else "cpu"),
        "set_tf32": bool(pick("set_tf32", True)),
    }

    # LoRA configs
    config["lora_en_cfg"] = {
        "r": pick("lora_en_r"),
        "alpha": pick("lora_en_alpha"),
        "dropout": pick("lora_en_dropout"),
    }
    config["lora_risk_cfg"] = {
        "r": pick("lora_risk_r"),
        "alpha": pick("lora_risk_alpha"),
        "dropout": pick("lora_risk_dropout"),
    }
    config["lora_cbt_cfg"] = {
        "r": pick("lora_cbt_r"),
        "alpha": pick("lora_cbt_alpha"),
        "dropout": pick("lora_cbt_dropout"),
    }

    # Basic validation: ensure required fields are set
    required = ["model_path", "mental_data_path", "align_data_path",
                "max_length", "align_max_len", "align_every",
                "batch_size", "grad_accum_steps", "lr", "epochs",
                "max_grad_norm", "warmup_ratio", "weight_decay",
                "alpha_sft", "alpha_align", "temperature",
                "hidden_size", "bottleneck_size", "lora_targets",
                "lora_en_cfg", "lora_risk_cfg", "lora_cbt_cfg"]
    missing = []
    for k in required:
        if config.get(k) is None:
            missing.append(k)
    # Also check LoRA subfields
    for sub in ["r", "alpha", "dropout"]:
        for name in ["lora_en_cfg", "lora_risk_cfg", "lora_cbt_cfg"]:
            if config[name].get(sub) is None:
                missing.append(f"{name}.{sub}")

    if missing:
        raise ValueError(f"Missing required config fields: {missing}")

    # lora_targets: allow CLI list or config list
    if isinstance(config["lora_targets"], str):
        # if config passed as single string
        config["lora_targets"] = [config["lora_targets"]]

    os.makedirs(config["output_dir"], exist_ok=True)
    return config


# ---------------------------
# 6) TRAIN
# ---------------------------

def train(config: Dict[str, Any]):
    device = config["device"]

    if config["set_tf32"]:
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass

    tokenizer = AutoTokenizer.from_pretrained(
        config["model_path"],
        trust_remote_code=True,
        local_files_only=bool(config["local_files_only"]),
    )
    tokenizer = patch_tokenizer_for_glm4(tokenizer)

    model = IA_AMoE_GLM(
        model_path=config["model_path"],
        lora_targets=config["lora_targets"],
        hidden_size=config["hidden_size"],
        bottleneck_size=config["bottleneck_size"],
        lora_en_cfg=config["lora_en_cfg"],
        lora_risk_cfg=config["lora_risk_cfg"],
        lora_cbt_cfg=config["lora_cbt_cfg"],
        device=device,
    ).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"[INFO] Trainable params: {sum(p.numel() for p in trainable_params)}")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(config["lr"]),
        weight_decay=float(config["weight_decay"]),
    )

    ds_sft = MentalSFTDataset(config["mental_data_path"], tokenizer, max_length=config["max_length"])
    ds_align = AlignmentDataset(
        config["align_data_path"],
        tokenizer,
        max_length=config["align_max_len"],
        risk_keywords=list(config["risk_keywords"]),
    )

    dl_sft = DataLoader(
        ds_sft,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        collate_fn=ds_sft.collate_fn,
        num_workers=0,
    )
    dl_align = DataLoader(
        ds_align,
        batch_size=max(2, int(config["batch_size"])),
        shuffle=True,
        collate_fn=ds_align.collate_fn,
        drop_last=True,
        num_workers=0,
    )

    total_steps = math.ceil(len(dl_sft) / int(config["grad_accum_steps"])) * int(config["epochs"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * float(config["warmup_ratio"])),
        num_training_steps=total_steps,
    )

    use_scaler = (device == "cuda" and model.compute_dtype == torch.float16)
    scaler = torch.amp.GradScaler() if use_scaler else None

    print(
        f"🚀 Start Training IA-AMoE... "
        f"epochs={config['epochs']} batch={config['batch_size']} align_every={config['align_every']}"
    )
    model.train()

    for epoch in range(int(config["epochs"])):
        print(f"=== Epoch {epoch + 1}/{config['epochs']} ===")
        iter_align = cycle(dl_align)

        progress = tqdm(enumerate(dl_sft), total=len(dl_sft))
        optimizer.zero_grad(set_to_none=True)

        for step, batch_sft in progress:
            input_ids = batch_sft["input_ids"].to(device, non_blocking=True)
            attention_mask = batch_sft["attention_mask"].to(device, non_blocking=True).long()
            labels = batch_sft["labels"].to(device, non_blocking=True)
            adapters = batch_sft["adapters"]

            is_last_step = (step + 1) == len(dl_sft)
            ga = int(config["grad_accum_steps"])
            accum_steps = ga if not is_last_step else (len(dl_sft) % ga or ga)

            autocast_ctx = (
                torch.amp.autocast(device_type="cuda", dtype=model.compute_dtype)
                if device == "cuda"
                else nullcontext()
            )

            # -------- 1) SFT (grouped by adapter) --------
            unique_adapters = list(set(adapters))
            groups_count = max(len(unique_adapters), 1)
            loss_sft_total_val = 0.0

            for ad_name in unique_adapters:
                idx_list = [i for i, a in enumerate(adapters) if a == ad_name]
                if not idx_list:
                    continue
                idx = torch.tensor(idx_list, device=device)

                with autocast_ctx:
                    loss_curr = model.forward_sft(
                        input_ids[idx], attention_mask[idx], labels[idx], adapter_name=ad_name
                    )
                    loss_curr_scaled = (float(config["alpha_sft"]) * loss_curr / groups_count) / accum_steps

                loss_sft_total_val += float(loss_curr.detach().item())

                if scaler:
                    scaler.scale(loss_curr_scaled).backward()
                else:
                    loss_curr_scaled.backward()

            loss_sft_total_val = loss_sft_total_val / groups_count

            # -------- 2) Alignment (periodic) --------
            do_align = ((step + 1) % int(config["align_every"]) == 0)
            loss_align_val = 0.0

            if do_align:
                batch_align = next(iter_align)
                zh_in = {k: v.to(device) for k, v in batch_align["zh_input"].items()}
                en_in = {k: v.to(device) for k, v in batch_align["en_input"].items()}
                zh_in["attention_mask"] = zh_in["attention_mask"].long()
                en_in["attention_mask"] = en_in["attention_mask"].long()
                zh_adapters_list = batch_align["zh_adapters"]

                # Anchor en embedding (no grad)
                with torch.no_grad():
                    z_en_all = model.get_embedding(
                        en_in["input_ids"], en_in["attention_mask"], adapter_name="lora_en"
                    ).detach()

                B = z_en_all.size(0)
                for ad_name in sorted(set(zh_adapters_list)):
                    idx_list = [i for i, a in enumerate(zh_adapters_list) if a == ad_name]
                    if len(idx_list) < 2:
                        continue
                    idx = torch.tensor(idx_list, device=device)

                    with autocast_ctx:
                        z_zh = model.get_embedding(
                            zh_in["input_ids"][idx],
                            zh_in["attention_mask"][idx],
                            adapter_name=ad_name,
                        )
                        z_en = z_en_all[idx]
                        loss_g = compute_infonce_loss(z_zh, z_en, temperature=float(config["temperature"]))

                        w = float(len(idx_list)) / float(B)
                        loss_g_scaled = (float(config["alpha_align"]) * w * loss_g) / accum_steps

                    loss_align_val += float(loss_g.detach().item()) * w

                    if scaler:
                        scaler.scale(loss_g_scaled).backward()
                    else:
                        loss_g_scaled.backward()

            # -------- optimizer step --------
            if ((step + 1) % int(config["grad_accum_steps"]) == 0) or is_last_step:
                if scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable_params, float(config["max_grad_norm"]))
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(trainable_params, float(config["max_grad_norm"]))
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            progress.set_postfix({
                "L_sft": f"{loss_sft_total_val:.4f}",
                "L_align": f"{loss_align_val:.4f}",
                "align?": int(do_align),
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
            })

        save_checkpoint(model, tokenizer, config["output_dir"], epoch)

    print("[DONE] Training finished. Checkpoints saved to:", config["output_dir"])


def build_argparser():
    p = argparse.ArgumentParser("IA-AMoE Training (Sanitized)")

    p.add_argument("--config_json", type=str, default=None, help="Path to a JSON config file (optional).")

    # paths
    p.add_argument("--model_path", type=str, default=None)
    p.add_argument("--mental_data_path", type=str, default=None)
    p.add_argument("--align_data_path", type=str, default=None)
    p.add_argument("--output_dir", type=str, default=None)

    # general
    p.add_argument("--device", type=str, default=None, help="cuda / cpu (default auto)")
    p.add_argument("--local_files_only", action="store_true", help="If set, only load local model files.")
    p.add_argument("--set_tf32", action="store_true", help="If set, enable TF32 (CUDA only).")

    # lengths
    p.add_argument("--max_length", type=int, default=None)
    p.add_argument("--align_max_len", type=int, default=None)

    # schedule
    p.add_argument("--align_every", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--grad_accum_steps", type=int, default=None)

    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--warmup_ratio", type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=None)
    p.add_argument("--max_grad_norm", type=float, default=None)

    # losses
    p.add_argument("--alpha_sft", type=float, default=None)
    p.add_argument("--alpha_align", type=float, default=None)
    p.add_argument("--temperature", type=float, default=None)

    # model dims
    p.add_argument("--hidden_size", type=int, default=None)
    p.add_argument("--bottleneck_size", type=int, default=None)

    # lora targets
    p.add_argument(
        "--lora_targets",
        nargs="+",
        default=None,
        help="List of target module names for LoRA.",
    )

    # lora params (EN)
    p.add_argument("--lora_en_r", type=int, default=None)
    p.add_argument("--lora_en_alpha", type=float, default=None)
    p.add_argument("--lora_en_dropout", type=float, default=None)

    # lora params (RISK)
    p.add_argument("--lora_risk_r", type=int, default=None)
    p.add_argument("--lora_risk_alpha", type=float, default=None)
    p.add_argument("--lora_risk_dropout", type=float, default=None)

    # lora params (CBT)
    p.add_argument("--lora_cbt_r", type=int, default=None)
    p.add_argument("--lora_cbt_alpha", type=float, default=None)
    p.add_argument("--lora_cbt_dropout", type=float, default=None)

    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    cfg = load_config(args)
    train(cfg)
