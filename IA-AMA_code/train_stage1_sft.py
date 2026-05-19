import argparse
import json
import math
import os
import random

import numpy as np
import torch
import torch.distributed as dist
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

from routing import EXPERT_EN, EXPERT_RISK, EXPERT_CBT
from data_loader import MentalSFTDataset, build_dataloader


def setup_ddp():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        return True, local_rank, int(os.environ["WORLD_SIZE"]), int(os.environ["RANK"])
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


def build_model(model_path, lora_targets,
                lora_en_cfg, lora_risk_cfg, lora_cbt_cfg,
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

    def cfg(d):
        return LoraConfig(
            task_type="CAUSAL_LM", inference_mode=False,
            r=int(d["r"]), lora_alpha=float(d["alpha"]),
            lora_dropout=float(d["dropout"]),
            target_modules=lora_targets, bias="none",
        )

    model = get_peft_model(base, cfg(lora_en_cfg), adapter_name=EXPERT_EN)
    model.add_adapter(EXPERT_RISK, cfg(lora_risk_cfg))
    model.add_adapter(EXPERT_CBT,  cfg(lora_cbt_cfg))


    for n, p in model.named_parameters():
        if "lora_" in n:
            p.requires_grad = True

    if is_main(local_rank):
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[model] total={total/1e9:.2f}B trainable={trainable/1e6:.2f}M "
              f"({100*trainable/total:.3f}%)")
        print(f"[model] adapters: {EXPERT_EN}(r={lora_en_cfg['r']}), "
              f"{EXPERT_RISK}(r={lora_risk_cfg['r']}), "
              f"{EXPERT_CBT}(r={lora_cbt_cfg['r']})")

    return model, tokenizer


def save_task_adapters(model, out_dir, rank):
    if not is_main(rank):
        return
    os.makedirs(out_dir, exist_ok=True)
    for name in (EXPERT_EN, EXPERT_RISK, EXPERT_CBT):
        adir = os.path.join(out_dir, "adapters", name)
        os.makedirs(adir, exist_ok=True)
        model.set_adapter(name)
        model.save_pretrained(adir, selected_adapters=[name])
    print(f"[rank 0] saved task adapters -> {out_dir}/adapters/")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out_dir", required=True)

    p.add_argument("--lora_targets", nargs="+",
                   default=["q_proj", "k_proj", "v_proj", "o_proj"])
    p.add_argument("--lora_en_r", type=int, default=16)
    p.add_argument("--lora_en_alpha", type=float, default=32.0)
    p.add_argument("--lora_en_dropout", type=float, default=0.05)
    p.add_argument("--lora_risk_r", type=int, default=8)
    p.add_argument("--lora_risk_alpha", type=float, default=16.0)
    p.add_argument("--lora_risk_dropout", type=float, default=0.05)
    p.add_argument("--lora_cbt_r", type=int, default=32)
    p.add_argument("--lora_cbt_alpha", type=float, default=64.0)
    p.add_argument("--lora_cbt_dropout", type=float, default=0.05)

    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--num_workers", type=int, default=2)
    args = p.parse_args()

    is_ddp, local_rank, world_size, global_rank = setup_ddp()
    seed_all(args.seed + global_rank)

    if is_main(global_rank):
        print(f"[STAGE 1: Task Expert SFT]")
        print(f"[ddp] is_ddp={is_ddp} world_size={world_size} rank={global_rank}")
        os.makedirs(args.out_dir, exist_ok=True)
        with open(os.path.join(args.out_dir, "stage1_args.json"), "w") as f:
            json.dump(vars(args), f, indent=2, ensure_ascii=False)

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dtype = dtype_map[args.dtype]

    model, tokenizer = build_model(
        model_path=args.model_path,
        lora_targets=args.lora_targets,
        lora_en_cfg={"r": args.lora_en_r, "alpha": args.lora_en_alpha,
                     "dropout": args.lora_en_dropout},
        lora_risk_cfg={"r": args.lora_risk_r, "alpha": args.lora_risk_alpha,
                       "dropout": args.lora_risk_dropout},
        lora_cbt_cfg={"r": args.lora_cbt_r, "alpha": args.lora_cbt_alpha,
                      "dropout": args.lora_cbt_dropout},
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

    train_set = MentalSFTDataset(
        data_dir=args.data_dir, tokenizer=tokenizer,
        max_length=args.max_length, split="train",
    )
    train_loader, train_sampler = build_dataloader(
        train_set, batch_size=args.batch_size, shuffle=True,
        seed=args.seed, num_replicas=world_size, rank=global_rank,
        num_workers=args.num_workers,
    )

    trainable = [p for p in base_model.parameters() if p.requires_grad]
    optim = AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    steps_per_epoch = len(train_sampler) // args.grad_accum
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

    if is_main(global_rank):
        print(f"[train] steps/epoch={steps_per_epoch} total={total_steps} warmup={warmup_steps}")

    global_step = 0
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        running = {"sft": 0.0, "n": 0}
        optim.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            adapter = batch["adapter"]
            input_ids = batch["input_ids"].to(f"cuda:{local_rank}")
            attn_mask = batch["attention_mask"].to(f"cuda:{local_rank}")
            labels = batch["labels"].to(f"cuda:{local_rank}")

            base_model.set_adapter(adapter)
            out = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
            loss = out.loss / args.grad_accum
            loss.backward()

            running["sft"] += float(out.loss.item())
            running["n"] += 1

            if (batch_idx + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
                optim.step()
                scheduler.step()
                optim.zero_grad()
                global_step += 1

                if is_main(global_rank) and global_step % 10 == 0:
                    avg = running["sft"] / max(1, running["n"])
                    lr_now = scheduler.get_last_lr()[0]
                    print(f"[ep {epoch} step {global_step}/{total_steps}] "
                          f"sft={avg:.4f} lr={lr_now:.2e} adapter={adapter}")
                    running = {"sft": 0.0, "n": 0}

        if is_main(global_rank):
            epoch_dir = os.path.join(args.out_dir, f"epoch{epoch+1}")
            save_task_adapters(base_model, epoch_dir, global_rank)
        if is_ddp:
            dist.barrier()

    cleanup_ddp(is_ddp)
    if is_main(global_rank):
        print("[STAGE 1 DONE]")


if __name__ == "__main__":
    main()
