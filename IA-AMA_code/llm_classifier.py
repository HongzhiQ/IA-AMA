
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

EXPERT_EN = "lora_en"
EXPERT_RISK = "lora_risk"
EXPERT_CBT = "lora_cbt"
EXPERT_CLA = "lora_cla"


SYS_ZH = ("你是一位心理健康分析专家。这是一个科研数据分析任务。"
          "请根据定义客观地对用户的帖子进行分类。\n\n"
          "输出格式要求：\n标签：[你的选项]\n理由：[你的分析]")
SYS_EN = ("You are a mental-health annotation assistant. "
          "Given the user's request, output a label and a brief rationale.\n\n"
          "Output format:\nLabel: [your choice]\nRationale: [your analysis]")
LABEL_PREFIX_ZH = "标签："
LABEL_PREFIX_EN = "Label: "


def _cjk(text):
    return sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")


def _pick_system(sample):
    text = (sample.get("instruction", "") or "") + (sample.get("input", "") or "")
    return SYS_ZH if _cjk(text) >= 3 else SYS_EN


def _pick_prefix(sample):
    text = (sample.get("instruction", "") or "") + (sample.get("input", "") or "")
    return LABEL_PREFIX_ZH if _cjk(text) >= 3 else LABEL_PREFIX_EN


class LLMClassifier:
    def __init__(self, model, tokenizer, adapter_names=None,
                 has_cla=False, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.adapter_names = adapter_names or []
        self.has_cla = has_cla
        self.device = device
        self.model.eval()

    @classmethod
    def from_pretrained(cls, backbone, model_path,
                        adapter_paths=None, single_adapter_path=None,
                        adapter_cla_path=None,
                        dtype=torch.bfloat16, device="cuda"):
        print(f"[LLMClassifier] backbone={backbone} from LOCAL {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, local_files_only=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype,
            trust_remote_code=True, local_files_only=True,
        )
        model.to(device)
        model.eval()

        adapter_names = []
        has_cla = False

        if adapter_paths:
            from peft import PeftModel
            first = True
            for name, path in adapter_paths.items():
                if first:
                    model = PeftModel.from_pretrained(model, path, adapter_name=name)
                    first = False
                else:
                    model.load_adapter(path, adapter_name=name)
                adapter_names.append(name)
                print(f"  + loaded {name} <- {path}")

            if adapter_cla_path:
                model.load_adapter(adapter_cla_path, adapter_name=EXPERT_CLA)
                adapter_names.append(EXPERT_CLA)
                has_cla = True
                print(f"  + loaded {EXPERT_CLA} (CLA stacked) <- {adapter_cla_path}")

        elif single_adapter_path:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, single_adapter_path)
            print(f"  + loaded single adapter <- {single_adapter_path}")

        model.eval()
        return cls(model=model, tokenizer=tokenizer,
                   adapter_names=adapter_names, has_cla=has_cla,
                   device=device)

    def set_stack(self, task_expert: str):

        if task_expert not in self.adapter_names and task_expert not in (EXPERT_EN, EXPERT_RISK, EXPERT_CBT):
            return
        if task_expert == EXPERT_EN:
            stack = [EXPERT_EN]
        elif self.has_cla:
            stack = [task_expert, EXPERT_CLA]
        else:
            stack = [task_expert]

        try:
            self.model.set_adapter(stack)
        except Exception:
            self.model.set_adapter(stack[0])

    def _build_prompt_ids(self, sample):
        sys_text = _pick_system(sample)
        instruction = (sample.get("instruction", "") or "").strip()
        input_text = (sample.get("input", "") or sample.get("original_input", "") or "").strip()
        user_text = instruction if not input_text else (instruction + "\n\n" + input_text)

        messages = [
            {"role": "system", "content": sys_text},
            {"role": "user",   "content": user_text},
        ]
        if hasattr(self.tokenizer, "apply_chat_template"):
            ids = self.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True
            )
        else:
            raw = f"[SYSTEM]\n{sys_text}\n[USER]\n{user_text}\n[ASSISTANT]\n"
            ids = self.tokenizer(raw, add_special_tokens=False)["input_ids"]

        prefix = _pick_prefix(sample)
        prefix_ids = self.tokenizer(prefix, add_special_tokens=False)["input_ids"]
        ids = list(ids) + list(prefix_ids)
        return torch.tensor(ids, dtype=torch.long, device=self.device)

    @torch.no_grad()
    def classify(self, sample, label_space):
        prompt_ids = self._build_prompt_ids(sample)
        scores = []
        for label in label_space:
            label_ids = self.tokenizer(label, add_special_tokens=False)["input_ids"]
            if len(label_ids) == 0:
                scores.append(-1e9)
                continue
            lb_t = torch.tensor(label_ids, dtype=torch.long, device=self.device)
            full = torch.cat([prompt_ids, lb_t], dim=0).unsqueeze(0)

            out = self.model(input_ids=full, use_cache=False)
            logits = out.logits[0]

            start = prompt_ids.shape[0] - 1
            label_logits = logits[start:start + len(label_ids)]
            logp = F.log_softmax(label_logits.float(), dim=-1)
            idx = torch.tensor(label_ids, dtype=torch.long, device=self.device)
            token_logps = logp.gather(1, idx.unsqueeze(1)).squeeze(1)
            score = token_logps.sum().item() / max(len(label_ids), 1)
            scores.append(score)

        arr = np.array(scores, dtype=np.float64)
        arr -= arr.max()
        probs = np.exp(arr)
        probs = probs / probs.sum()
        return probs
