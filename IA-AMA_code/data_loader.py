import glob
import json
import os
import random
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset, DataLoader, Sampler

from routing import route, EXPERT_EN, EXPERT_RISK, EXPERT_CBT


class MentalSFTDataset(Dataset):

    def __init__(self,
                 data_dir: str,
                 tokenizer,
                 max_length: int = 1024,
                 split: str = "train",
                 file_glob: Optional[str] = None):
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.split = split

        pattern = file_glob or f"*_{split}_prompts.json"
        files = sorted(glob.glob(os.path.join(data_dir, pattern)))
        if not files:
            raise FileNotFoundError(
                f"No files matching {pattern} under {data_dir}"
            )

        self.data: List[dict] = []
        per_file_counts = {}
        for fp in files:
            with open(fp, "r", encoding="utf-8") as f:
                rows = json.load(f)

            tk_from_name = os.path.basename(fp).replace(f"_{split}_prompts.json", "")
            for r in rows:
                if not r.get("task_key"):
                    r["task_key"] = tk_from_name
                self.data.append(r)
            per_file_counts[os.path.basename(fp)] = len(rows)


        self.sample_experts: List[str] = [route(r) for r in self.data]


        print(f"[MentalSFTDataset/{split}] loaded {len(self.data)} samples "
              f"from {len(files)} files")
        for name, n in per_file_counts.items():
            print(f"    - {name:55s}  n={n}")
        from collections import Counter
        exp_counts = Counter(self.sample_experts)
        print(f"[MentalSFTDataset/{split}] expert distribution: {dict(exp_counts)}")


        self.zh_sys = ("你是一位心理健康分析专家。这是一个科研数据分析任务。"
                       "请根据定义客观地对用户的帖子进行分类。\n\n"
                       "输出格式要求：\n标签：[你的选项]\n理由：[你的分析]")
        self.en_sys = ("You are a mental-health annotation assistant. "
                       "Follow the user's instruction.")
        self.has_chat_template = hasattr(tokenizer, "apply_chat_template")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        return {"_row": self.data[idx],
                "_expert": self.sample_experts[idx]}

    def _build_ids(self, system_text, user_text, assistant_text):
        if self.has_chat_template:
            msgs_prompt = [
                {"role": "system", "content": system_text},
                {"role": "user",   "content": user_text},
            ]
            msgs_full = msgs_prompt + [
                {"role": "assistant", "content": assistant_text}
            ]
            prompt_ids = self.tokenizer.apply_chat_template(
                msgs_prompt, tokenize=True, add_generation_prompt=True
            )
            full_ids = self.tokenizer.apply_chat_template(
                msgs_full, tokenize=True, add_generation_prompt=False
            )
            return prompt_ids, full_ids
        raw_prompt = f"[SYSTEM]\n{system_text}\n[USER]\n{user_text}\n[ASSISTANT]\n"
        raw_full = raw_prompt + assistant_text
        prompt_ids = self.tokenizer(raw_prompt, add_special_tokens=False)["input_ids"]
        full_ids = self.tokenizer(raw_full, add_special_tokens=False)["input_ids"]
        return prompt_ids, full_ids


    def collate_fn(self, batch_items):
        experts = [bi["_expert"] for bi in batch_items]
        assert len(set(experts)) == 1, \
            f"collate_fn received a heterogeneous batch: {set(experts)}"
        adapter_name = experts[0]

        input_ids_list, labels_list = [], []
        for bi in batch_items:
            row = bi["_row"]
            instruction = (row.get("instruction", "") or "").strip()
            input_context = (row.get("original_input", "")
                             or row.get("input", "") or "").strip()
            output_text = (row.get("original_output", "")
                           or row.get("output", "") or "").strip()

            is_en = (adapter_name == EXPERT_EN)
            sys_text = self.en_sys if is_en else self.zh_sys
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

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids_list, batch_first=True, padding_value=pad_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels_list, batch_first=True, padding_value=-100
        )
        attention_mask = input_ids.ne(pad_id).long()

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
            "adapter":        adapter_name,
        }


class ExpertHomogeneousBatchSampler(Sampler):

    def __init__(self,
                 dataset: MentalSFTDataset,
                 batch_size: int,
                 shuffle: bool = True,
                 seed: int = 0,
                 num_replicas: int = 1,
                 rank: int = 0,
                 drop_last: bool = True):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.seed = seed
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.drop_last = drop_last
        self.epoch = 0


        self.indices_by_expert: Dict[str, List[int]] = {
            EXPERT_EN: [], EXPERT_RISK: [], EXPERT_CBT: [],
        }
        for i, exp in enumerate(dataset.sample_experts):
            self.indices_by_expert[exp].append(i)

        self._recompute_batches()

    def _recompute_batches(self):
        total = 0
        for exp, idxs in self.indices_by_expert.items():
            n = len(idxs) // self.batch_size
            total += n

        self.num_batches_global = total
        self.num_batches_per_rank = total // self.num_replicas

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        all_batches: List[List[int]] = []

        for exp, idxs in self.indices_by_expert.items():
            idxs = list(idxs)
            if self.shuffle:
                rng.shuffle(idxs)
            n_full = len(idxs) // self.batch_size
            for b in range(n_full):
                all_batches.append(idxs[b*self.batch_size:(b+1)*self.batch_size])

        if self.shuffle:
            rng.shuffle(all_batches)


        total = (len(all_batches) // self.num_replicas) * self.num_replicas
        all_batches = all_batches[:total]
        my_batches = all_batches[self.rank::self.num_replicas]
        for b in my_batches:
            yield b

    def __len__(self):
        return self.num_batches_per_rank


def build_dataloader(dataset: MentalSFTDataset,
                     batch_size: int,
                     shuffle: bool = True,
                     seed: int = 0,
                     num_replicas: int = 1,
                     rank: int = 0,
                     num_workers: int = 2) -> DataLoader:
    sampler = ExpertHomogeneousBatchSampler(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
        num_replicas=num_replicas,
        rank=rank,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=dataset.collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader, sampler
