import argparse
import json
import math
import os
import random


import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, PeftModel

from routing import EXPERT_EN, EXPERT_RISK, EXPERT_CBT

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


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class AlignmentDataset(torch.utils.data.Dataset):

    def __init__(self, json_path, tokenizer, max_length=256):
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        print(f"[align] loaded {len(self.data)} EN-ZH pairs from {json_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def collate_fn(self, batch):
        en_texts = [b.get("en_text") or b.get("en") for b in batch]
        zh_texts = [b.get("zh_text") or b.get("zh") for b in batch]
        en_enc = self.tokenizer(
            en_texts, padding=True, truncation=True,
            max_length=self.max_length, return_tensors="pt",
        )
        zh_enc = self.tokenizer(
            zh_texts, padding=True, truncation=True,
            max_length=self.max_length, return_tensors="pt",
        )
        return {"en": en_enc, "zh": zh_enc}


def build_stage2_model(model_path, stage1_dir,
                       lora_cla_cfg, lora_targets,
                       dtype, local_rank):
    print(f"[rank {local_rank}] loading base LLM from LOCAL: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype,
        trust_remote_code=True, local_files_only=True,
    )
    base.to(f"cuda:{local_rank}")

    adapter_en_path   = os.path.join(stage1_dir, "adapters", EXPERT_EN)
    adapter_risk_path = os.path.join(stage1_dir, "adapters", EXPERT_RISK)
    adapter_cbt_path  = os.path.join(stage1_dir, "adapters", EXPERT_CBT)
    for p in (adapter_en_path, adapter_risk_path, adapter_cbt_path):
        assert os.path.exists(p), f"Stage-1 adapter not found: {p}"

    model = PeftModel.from_pretrained(base, adapter_en_path, adapter_name=EXPERT_EN)
    model.load_adapter(adapter_risk_path, adapter_name=EXPERT_RISK)
    model.load_adapter(adapter_cbt_path,  adapter_name=EXPERT_CBT)
    print(f"[rank {local_rank}] loaded 3 frozen Stage-1 adapters")


    for n, p in model.named_parameters():
        if "lora_" in n:
            p.requires_grad = False


    cla_cfg = LoraConfig(
        task_type="CAUSAL_LM", inference_mode=False,
        r=int(lora_cla_cfg["r"]),
        lora_alpha=float(lora_cla_cfg["alpha"]),
        lora_dropout=float(lora_cla_cfg["dropout"]),
        target_modules=lora_targets, bias="none",
    )
    model.add_adapter(EXPERT_CLA, cla_cfg)


    for n, p in model.named_parameters():
        if EXPERT_CLA in n and "lora_" in n:
            p.requires_grad = True

    if is_main(local_rank):
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        print(f"[stage2 model] trainable CLA params: {trainable/1e6:.2f}M "
              f"(frozen rest: {frozen/1e9:.2f}B)")
        print(f"[stage2 model] CLA rank={lora_cla_cfg['r']} "
              f"alpha={lora_cla_cfg['alpha']} dropout={lora_cla_cfg['dropout']}")

    return model, tokenizer


def mean_pool(hidden, mask):
    m = mask.unsqueeze(-1).float()
    s = (hidden * m).sum(dim=1)
    d = m.sum(dim=1).clamp(min=1.0)
    return s / d


def set_active_adapters(model, names):

    try:
        model.set_adapter(names if isinstance(names, list) else [names])
    except Exception:
        # Fallback: just activate the first adapter in the list
        if isinstance(names, list):
            model.set_adapter(names[0])
        else:
            model.set_adapter(names)


def compute_repr(model, adapter_stack, enc, device):

    set_active_adapters(model, adapter_stack)
    out = model(
        input_ids=enc["input_ids"].to(device),
        attention_mask=enc["attention_mask"].to(device),
        output_hidden_states=True,
        use_cache=False,
    )
    h = out.hidden_states[-1]
    return mean_pool(h, enc["attention_mask"].to(device))


def infonce_loss(z_zh, z_en, tau):

    z_zh = F.normalize(z_zh, dim=-1)
    z_en = F.normalize(z_en, dim=-1).detach()
    logits = z_zh @ z_en.t() / tau
    labels = torch.arange(z_zh.size(0), device=z_zh.device)
    return F.cross_entropy(logits, labels)


def save_cla_adapter(model, out_dir, rank):
    if not is_main(rank):
        return
    adir = os.path.join(out_dir, "adapters", EXPERT_CLA)
    os.makedirs(adir, exist_ok=True)
    model.set_adapter(EXPERT_CLA)
    model.save_pretrained(adir, selected_adapters=[EXPERT_CLA])
    print(f"[rank 0] saved CLA adapter -> {adir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--stage1_dir", required=True,
                   help="Path to Stage-1 epoch dir containing adapters/{lora_en,lora_risk,lora_cbt}")
    p.add_argument("--align_file", required=True)
    p.add_argument("--out_dir", required=True)

    p.add_argument("--lora_targets", nargs="+",
                   default=["q_proj", "k_proj", "v_proj", "o_proj"])
    p.add_argument("--lora_cla_r", type=int, default=16)
    p.add_argument("--lora_cla_alpha", type=float, default=32.0)
    p.add_argument("--lora_cla_dropout", type=float, default=0.05)

    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--align_tau", type=float, default=0.07)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--num_workers", type=int, default=1)
    args = p.parse_args()

    is_ddp, local_rank, world_size, global_rank = setup_ddp()
    seed_all(args.seed + global_rank)
    device = f"cuda:{local_rank}"

    if is_main(global_rank):
        print("[STAGE 2: Cross-Lingual Alignment Adapter Training]")
        print(f"[ddp] is_ddp={is_ddp} world_size={world_size} rank={global_rank}")
        os.makedirs(args.out_dir, exist_ok=True)
        with open(os.path.join(args.out_dir, "stage2_args.json"), "w") as f:
            json.dump(vars(args), f, indent=2, ensure_ascii=False)

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dtype = dtype_map[args.dtype]

    model, tokenizer = build_stage2_model(
        model_path=args.model_path,
        stage1_dir=args.stage1_dir,
        lora_cla_cfg={"r": args.lora_cla_r, "alpha": args.lora_cla_alpha,
                      "dropout": args.lora_cla_dropout},
        lora_targets=args.lora_targets,
        dtype=dtype, local_rank=local_rank,
    )

    if is_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=True,
        )
        base_model = model.module
    else:
        base_model = model


    align_set = AlignmentDataset(args.align_file, tokenizer, args.max_length)
    sampler = torch.utils.data.distributed.DistributedSampler(
        align_set, num_replicas=world_size, rank=global_rank, shuffle=True, seed=args.seed,
    ) if is_ddp else None
    loader = torch.utils.data.DataLoader(
        align_set, batch_size=args.batch_size,
        shuffle=(sampler is None), sampler=sampler,
        collate_fn=align_set.collate_fn,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )


    trainable = [p for p in base_model.parameters() if p.requires_grad]
    optim = AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    steps_per_epoch = len(loader)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

    if is_main(global_rank):
        print(f"[train] steps/epoch={steps_per_epoch} total={total_steps} warmup={warmup_steps}")


    global_step = 0
    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        running = {"loss": 0.0, "n": 0}

        for batch_idx, batch in enumerate(loader):
            # EN forward: only the EN anchor adapter, no CLA
            with torch.no_grad():
                z_en = compute_repr(base_model, [EXPERT_EN], batch["en"], device)

            # ZH forward: CBT expert + CLA adapter stacked
            z_zh = compute_repr(base_model, [EXPERT_CBT, EXPERT_CLA], batch["zh"], device)

            loss = infonce_loss(z_zh, z_en, args.align_tau)
            loss.backward()

            running["loss"] += float(loss.item())
            running["n"] += 1

            torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
            optim.step()
            scheduler.step()
            optim.zero_grad()
            global_step += 1

            if is_main(global_rank) and global_step % 20 == 0:
                avg = running["loss"] / max(1, running["n"])
                lr_now = scheduler.get_last_lr()[0]
                print(f"[ep {epoch} step {global_step}/{total_steps}] "
                      f"infonce={avg:.4f} lr={lr_now:.2e}")
                running = {"loss": 0.0, "n": 0}

        if is_main(global_rank):
            ep_dir = os.path.join(args.out_dir, f"epoch{epoch+1}")
            save_cla_adapter(base_model, ep_dir, global_rank)
        if is_ddp:
            dist.barrier()

    cleanup_ddp(is_ddp)
    if is_main(global_rank):
        print("[STAGE 2 DONE]")


if __name__ == "__main__":
    main()
