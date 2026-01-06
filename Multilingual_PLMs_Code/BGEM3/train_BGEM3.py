import json
import torch
import torch.nn as nn
import re
import numpy as np
import os
import random
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm
from collections import defaultdict

MODEL_NAME = "BAAI/bge-m3"
TRAIN_FILE = "YOUR_TRAIN_DATA_PATH"
TEST_DATA_DIR = "YOUR_TEST_DATA_DIR"
OUTPUT_DIR = "YOUR_OUTPUT_DIR"

BATCH_SIZE = "YOUR_BATCH_SIZE"
GRAD_ACCUM_STEPS = "YOUR_ACCUM_STEPS"
EPOCHS = "YOUR_EPOCHS"
MAX_LEN = "YOUR_MAX_LEN"
LEARNING_RATE = "YOUR_LEARNING_RATE"
SEED = "YOUR_SEED_HERE"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TASK_CONFIG = {
    "suicide_scale":      { "id": 1, "type": "single_label", "num_labels": 11, "labels": [str(i) for i in range(11)] },
    "relation":           { "id": 2, "type": "multi_label", "num_labels": 5, "labels": ["哀伤与丧失", "人际缺陷", "角色演变", "角色内冲突", "角色间冲突"] },
    "depression_level":   { "id": 3, "type": "single_label", "num_labels": 4, "labels": ["无抑郁", "轻度抑郁", "中度抑郁", "重度抑郁"] },
    "stress_cause":       { "id": 4, "type": "single_label", "num_labels": 9, "labels": ["School", "Financial", "Family", "Work", "Social Relations", "Health", "Emotional Turmoil", "Everyday Decision Making", "Other"] },
    "cognitive_path":     { "id": 5, "type": "multi_label", "num_labels": 19, "labels": ["疾病症状", "社会关系", "生活", "学习工作", "情感", "非此即彼", "以偏概全", "心理过滤", "否定正面思考", "妄下结论", "放大和缩小", "情绪化推理", "应该句式", "乱贴标签", "罪责归己/他", "情感效应", "行为效应", "习惯性驳斥", "有效驳斥"] },
    "cognitive_distortion": { "id": 6, "type": "multi_label", "num_labels": 12, "labels": ["乱贴标签", "先知错误", "心理过滤", "罪责归己", "放大", "读心术", "以偏概全", "应该句式", "非此即彼", "罪责归他", "否定正面思考", "情绪化推理"] },
    "detect_depression":  { "id": 7, "type": "single_label", "num_labels": 2, "labels": ["No", "Yes"] },
    "irf_factors":        { "id": 8, "type": "single_label", "num_labels": 2, "labels": ["No", "Yes"] },
    "wellness_dims":      { "id": 9, "type": "single_label", "num_labels": 2, "labels": ["No", "Yes"] },
    "suicide_coarse":     { "id": 10, "type": "single_label", "num_labels": 2, "labels": ["低自杀风险", "高自杀风险"] },
    "detect_stress":      { "id": 11, "type": "single_label", "num_labels": 2, "labels": ["No", "Yes"] }
}

TASK_ID_MAP = {cfg["id"]: name for name, cfg in TASK_CONFIG.items()}
MAX_LABEL_DIM = max(cfg["num_labels"] for cfg in TASK_CONFIG.values())

MLB_MAPS = {}
for name, cfg in TASK_CONFIG.items():
    if cfg["type"] == "multi_label" and cfg["labels"]:
        mlb = MultiLabelBinarizer(classes=cfg["labels"])
        mlb.fit([cfg["labels"]])
        MLB_MAPS[name] = mlb

def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

def guess_task_from_text(text):
    text_lower = text.lower()
    if "自杀风险" in text:
        if "0-10级" in text: return "suicide_scale"
        elif "高还是低" in text: return "suicide_coarse"
        else: return None
    if "人际关系" in text: return "relation"
    if "认知路径" in text: return "cognitive_path"
    if "认知歪曲" in text: return "cognitive_distortion"
    if "抑郁等级" in text: return "depression_level"
    if "stress cause" in text_lower: return "stress_cause"
    if "thwarted belongingness" in text_lower or "perceived burdensomeness" in text_lower: return "irf_factors"
    if "wellness" in text_lower: return "wellness_dims"
    if "suffer from depression" in text_lower: return "detect_depression"
    if "suffer from stress" in text_lower: return "detect_stress"
    return None

def parse_label_for_task(task_name, text_input, output_text):
    cfg = TASK_CONFIG[task_name]
    output = str(output_text).strip()
    
    if task_name == "suicide_scale":
        nums = re.findall(r"\d+", output)
        if nums:
            val = int(nums[0])
            if 0 <= val <= 10: return torch.tensor(val, dtype=torch.long)
        return None

    if task_name in ["irf_factors", "wellness_dims"]:
        is_yes = False
        lower_out = output.lower()
        if "yes" in lower_out: is_yes = True
        elif "label: yes" in lower_out: is_yes = True
        elif lower_out.startswith("yes"): is_yes = True
        return torch.tensor(1 if is_yes else 0, dtype=torch.long)

    if cfg["type"] == "multi_label":
        clean_text = re.sub(r'[;；，、\s]+', ',', output)
        labels = [l.strip() for l in clean_text.split(',') if l.strip()]
        valid = [l for l in labels if l in cfg["labels"]]
        vec = MLB_MAPS[task_name].transform([valid])[0]
        return torch.tensor(vec, dtype=torch.float)

    if cfg["type"] == "single_label":
        target_idx = -1
        lower_out = output.lower()
        if lower_out.startswith("yes"): target_idx = 1
        elif lower_out.startswith("no"): target_idx = 0
        elif "label: yes" in lower_out: target_idx = 1
        elif "label: no" in lower_out: target_idx = 0
        else:
            for i, l in enumerate(cfg["labels"]):
                if l in output: target_idx = i; break
            if target_idx == -1:
                if "高" in output: target_idx = 1
                elif "低" in output: target_idx = 0
        if target_idx == -1: return None
        return torch.tensor(target_idx, dtype=torch.long)
    return None

def check_label_consistency(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    counts = defaultdict(int)
    ignored_count = 0
    
    for item in tqdm(data, desc="Scanning"):
        task_name = guess_task_from_text(item.get('instruction', '') + " " + item.get('input', ''))
        if task_name and task_name in TASK_CONFIG:
            counts[task_name] += 1
        else:
            ignored_count += 1
            
    for t, c in counts.items():
        print(f"Task: {t:<20} Samples: {c}")
    print(f"Ignored: {ignored_count}")

class MixedTrainDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_len):
        self.data = []
        self.tokenizer = tokenizer
        self.max_len = max_len
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        for item in raw_data:
            text = item.get('instruction', '') + " " + item.get('input', '')
            task_name = guess_task_from_text(text)
            if task_name and task_name in TASK_CONFIG:
                label = parse_label_for_task(task_name, text, item.get('output', ''))
                if label is not None:
                    self.data.append({'text': text, 'task_id': TASK_CONFIG[task_name]["id"], 'label': label})

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        enc = self.tokenizer.encode_plus(
            item['text'], add_special_tokens=True, max_length=self.max_len,
            padding='max_length', truncation=True, return_tensors='pt'
        )
        raw_label = item['label']
        padded_label = torch.full((MAX_LABEL_DIM,), -100.0, dtype=torch.float)
        
        if raw_label.dim() == 0: padded_label[0] = raw_label.float()
        else: padded_label[:raw_label.size(0)] = raw_label.float()
            
        return {
            'input_ids': enc['input_ids'].flatten(),
            'attention_mask': enc['attention_mask'].flatten(),
            'task_ids': torch.tensor(item['task_id'], dtype=torch.long),
            'labels': padded_label
        }

class UnifiedValidationDataset(Dataset):
    def __init__(self, eval_tasks, tokenizer, max_len):
        self.data = []
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        for task_name, display_name, file_path in eval_tasks:
            if not os.path.exists(file_path):
                print(f"Skipping {display_name}: File not found")
                continue
            
            if task_name not in TASK_CONFIG:
                continue

            task_id = TASK_CONFIG[task_name]["id"]
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
                
            count = 0
            for item in raw_data:
                text = item.get('instruction', '') + " " + item.get('input', '')
                label = parse_label_for_task(task_name, text, item.get('output', ''))
                if label is not None:
                    self.data.append({'text': text, 'task_id': task_id, 'label': label})
                    count += 1
            print(f"Loaded {display_name}: {count}")

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        enc = self.tokenizer.encode_plus(
            item['text'], add_special_tokens=True, max_length=self.max_len,
            padding='max_length', truncation=True, return_tensors='pt'
        )
        raw_label = item['label']
        padded_label = torch.full((MAX_LABEL_DIM,), -100.0, dtype=torch.float)
        
        if raw_label.dim() == 0: padded_label[0] = raw_label.float()
        else: padded_label[:raw_label.size(0)] = raw_label.float()
            
        return {
            'input_ids': enc['input_ids'].flatten(),
            'attention_mask': enc['attention_mask'].flatten(),
            'task_ids': torch.tensor(item['task_id'], dtype=torch.long),
            'labels': padded_label
        }

class UnifiedBGEM3(nn.Module):
    def __init__(self, model_name, task_config):
        super().__init__()
        self.bge = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.heads = nn.ModuleDict()
        
        hidden_size = self.bge.config.hidden_size
        for name, cfg in task_config.items():
            self.heads[name] = nn.Linear(hidden_size, cfg["num_labels"])
            
    def forward(self, input_ids, attention_mask, task_ids, labels=None):
        outputs = self.bge(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        cls_token_state = sequence_output[:, 0, :] 
        
        pooled = self.dropout(cls_token_state)
        
        total_loss = 0
        hits = 0
        logits_dict = {} 

        unique_tasks = torch.unique(task_ids)
        for t_id in unique_tasks:
            name = TASK_ID_MAP[t_id.item()]
            idxs = (task_ids == t_id)
            sub_in = pooled[idxs]
            sub_lbl_padded = labels[idxs] if labels is not None else None
            logits = self.heads[name](sub_in)
            logits_dict[name] = logits 

            if sub_lbl_padded is not None:
                cfg = TASK_CONFIG[name]
                if cfg["type"] == "multi_label":
                    real_targets = sub_lbl_padded[:, :cfg["num_labels"]]
                    pos_weight = torch.tensor([10.0] * cfg["num_labels"]).to(logits.device)
                    loss_fct = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                    loss = loss_fct(logits, real_targets)
                else: 
                    real_targets = sub_lbl_padded[:, 0].long()
                    loss = nn.CrossEntropyLoss()(logits, real_targets)
                total_loss += loss
                hits += 1
        
        final_loss = total_loss / hits if hits > 0 else torch.tensor(0.0, device=input_ids.device)
        return final_loss, logits_dict

def evaluate(model, loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating", leave=False):
            input_ids = batch['input_ids'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
            t_ids = batch['task_ids'].to(DEVICE)
            lbls = batch['labels'].to(DEVICE)
            loss, _ = model(input_ids, mask, t_ids, lbls)
            total_loss += loss.item()
    return total_loss / len(loader) if len(loader) > 0 else float('inf')

if __name__ == "__main__":
    set_seed(SEED)
    
    check_label_consistency(TRAIN_FILE)
    
    eval_tasks_list = [
        ("suicide_scale",      "Suicide Scale",    os.path.join(TEST_DATA_DIR, "suicide_scale.json")),
        ("suicide_coarse",     "Suicide Binary",   os.path.join(TEST_DATA_DIR, "suicide_binary.json")),
        ("relation",           "Relation",         os.path.join(TEST_DATA_DIR, "relation.json")),
        ("depression_level",   "Depression Level", os.path.join(TEST_DATA_DIR, "depression.json")),
        ("stress_cause",       "Stress Cause",     os.path.join(TEST_DATA_DIR, "stress_cause.json")),
        ("cognitive_path",     "Cognitive Path",   os.path.join(TEST_DATA_DIR, "cognitive_path.json")),
        ("cognitive_distortion","Cognitive Distortion", os.path.join(TEST_DATA_DIR, "cognitive_distortion.json")),
        ("detect_depression",  "Detect Depression",os.path.join(TEST_DATA_DIR, "detect_depression.json")),
        ("detect_stress",      "Detect Stress",    os.path.join(TEST_DATA_DIR, "detect_stress.json")),
        ("irf_factors",        "IRF",              os.path.join(TEST_DATA_DIR, "irf.json")),
        ("wellness_dims",      "Wellness",         os.path.join(TEST_DATA_DIR, "wellness.json"))
    ]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    model = UnifiedBGEM3(MODEL_NAME, TASK_CONFIG)
    model.to(DEVICE)
    
    train_dataset = MixedTrainDataset(TRAIN_FILE, tokenizer, MAX_LEN)
    val_dataset = UnifiedValidationDataset(eval_tasks_list, tokenizer, MAX_LEN)
    
    if len(train_dataset) == 0:
        print("Train dataset is empty.")
        exit()

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = (len(train_loader) * EPOCHS) // GRAD_ACCUM_STEPS
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, total_steps)

    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0
        optimizer.zero_grad()
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for step, batch in enumerate(loop):
            input_ids = batch['input_ids'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
            t_ids = batch['task_ids'].to(DEVICE)
            lbls = batch['labels'].to(DEVICE)
            
            loss, _ = model(input_ids, mask, t_ids, lbls)
            loss = loss / GRAD_ACCUM_STEPS
            loss.backward()
            
            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            current_loss = loss.item() * GRAD_ACCUM_STEPS
            total_train_loss += current_loss
            loop.set_postfix(loss=current_loss)
            
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = evaluate(model, val_loader)
        
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            output_path = os.path.join(OUTPUT_DIR, "best_model.pt")
            torch.save(model.state_dict(), output_path)
            print(f"Saved best model: {output_path}")