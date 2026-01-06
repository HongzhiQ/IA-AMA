import json
import torch
import torch.nn as nn
import re
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, f1_score, classification_report, precision_score, recall_score
from collections import defaultdict

MODEL_NAME = "facebook/xlm-v-base"
MODEL_PATH_TEMPLATE = "YOUR_MODEL_PATH_TEMPLATE" 
RESULT_TXT_PATH = "YOUR_RESULT_TXT_PATH"
TEST_DATA_DIR = "YOUR_TEST_DATA_DIR"

SEEDS = ["YOUR_SEEDS_HERE"]

THRESHOLD = "YOUR_THRESHOLD"
BATCH_SIZE = "YOUR_BATCH_SIZE"
MAX_LEN = "YOUR_MAX_LEN"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EVAL_TASKS_LIST = [
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

ENGLISH_TASKS = [
    "stress_cause", 
    "detect_depression", 
    "detect_stress", 
    "irf_factors", 
    "wellness_dims"
]

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

def log(msg, file_handler=None):
    print(msg)
    if file_handler:
        file_handler.write(msg + "\n")
        file_handler.flush()

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

class SingleTaskTestDataset(Dataset):
    def __init__(self, file_path, task_name, tokenizer, max_len):
        self.data = []
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.task_id = TASK_CONFIG[task_name]["id"]
        
        if not os.path.exists(file_path):
            return

        with open(file_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        for item in raw_data:
            text = item.get('instruction', '') + " " + item.get('input', '')
            label = parse_label_for_task(task_name, text, item.get('output', ''))
            if label is not None:
                self.data.append({'text': text, 'label': label})

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        enc = self.tokenizer.encode_plus(
            item['text'], add_special_tokens=True, max_length=self.max_len,
            padding='max_length', truncation=True, return_tensors='pt'
        )
        raw_label = item['label']
        padded_label = torch.full((MAX_LABEL_DIM,), -100.0, dtype=torch.float)
        
        if raw_label.dim() == 0:
            padded_label[0] = raw_label.float()
        else:
            length = raw_label.size(0)
            padded_label[:length] = raw_label.float()

        return {
            'input_ids': enc['input_ids'].flatten(),
            'attention_mask': enc['attention_mask'].flatten(),
            'task_ids': torch.tensor(self.task_id, dtype=torch.long),
            'labels': padded_label
        }

class UnifiedXLMV(nn.Module):
    def __init__(self, model_name, task_config):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.heads = nn.ModuleDict()
        
        hidden_size = self.model.config.hidden_size
        for name, cfg in task_config.items():
            self.heads[name] = nn.Linear(hidden_size, cfg["num_labels"])
            
    def forward(self, input_ids, attention_mask, task_ids):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        cls_token_state = sequence_output[:, 0, :] 
        pooled = self.dropout(cls_token_state)
        
        logits_dict = {} 
        unique_tasks = torch.unique(task_ids)
        for t_id in unique_tasks:
            found_name = None
            for name, cfg in TASK_CONFIG.items():
                if cfg["id"] == t_id.item():
                    found_name = name; break
            if found_name:
                idxs = (task_ids == t_id)
                sub_in = pooled[idxs]
                logits = self.heads[found_name](sub_in)
                logits_dict[found_name] = logits 
        return None, logits_dict

def calculate_parent_metrics(y_true, y_pred, parent_mapping, all_labels, file_f):
    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)
    
    parent_names = list(parent_mapping.keys())
    y_true_parent = np.zeros((len(y_true), len(parent_names)))
    y_pred_parent = np.zeros((len(y_pred), len(parent_names)))
    
    for i, p_name in enumerate(parent_names):
        child_names = parent_mapping[p_name]
        child_indices = [all_labels.index(c) for c in child_names if c in all_labels]
        if not child_indices: continue
        
        y_true_parent[:, i] = np.max(y_true_np[:, child_indices], axis=1)
        y_pred_parent[:, i] = np.max(y_pred_np[:, child_indices], axis=1)
        
    log("\n--- Parent Categories Metrics ---", file_f)
    log(classification_report(y_true_parent, y_pred_parent, target_names=parent_names, digits=4, zero_division=0), file_f)

def evaluate_tasks_individually(model, tokenizer, seed, file_f):
    model.eval()
    log("\n" + "="*60, file_f)
    log(f"SEED {seed} | Evaluation (Threshold: {THRESHOLD})", file_f)
    log("="*60, file_f)

    COGNITIVE_PATH_PARENTS = {
        "Trigger (诱发事件)": ["疾病症状", "社会关系", "生活", "学习工作", "情感"],
        "Distortion (认知歪曲)": ["非此即彼", "以偏概全", "心理过滤", "否定正面思考", "妄下结论", "放大和缩小", "情绪化推理", "应该句式", "乱贴标签", "罪责归己/他"],
        "Effect (效应)": ["情感效应", "行为效应"],
        "Refutation (驳斥)": ["习惯性驳斥", "有效驳斥"]
    }

    WELLNESS_DIMS = ["Spiritual", "Physical", "Intellectual", "Social", "Vocational", "Emotional"]
    IRF_DIMS = ["Thwarted Belongingness", "Perceived Burdensomeness"]

    for task_config_name, display_name, file_path in EVAL_TASKS_LIST:
        if not os.path.exists(file_path):
            log(f"Skipping: {display_name} (File not found)", file_f)
            continue
            
        dataset = SingleTaskTestDataset(file_path, task_config_name, tokenizer, MAX_LEN)
        if len(dataset) == 0: continue
        
        log(f"\n>>>>>> Evaluating: [{display_name}] (Samples: {len(dataset)})", file_f)

        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
        y_true, y_pred = [], []
        cfg = TASK_CONFIG[task_config_name]
        
        input_texts = []

        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(DEVICE)
                mask = batch['attention_mask'].to(DEVICE)
                t_ids = batch['task_ids'].to(DEVICE)
                labels_padded = batch['labels']
                
                if task_config_name in ["irf_factors", "wellness_dims"]:
                    decoded = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
                    input_texts.extend(decoded)

                _, logits_dict = model(input_ids, mask, t_ids)
                if task_config_name not in logits_dict: continue
                logits = logits_dict[task_config_name]
                
                if cfg["type"] == "multi_label":
                    preds = (torch.sigmoid(logits) > THRESHOLD).int().cpu().numpy()
                    trues = labels_padded[:, :cfg["num_labels"]].numpy()
                else: 
                    preds = torch.argmax(logits, dim=1).cpu().numpy()
                    trues = labels_padded[:, 0].long().numpy()
                
                y_pred.extend(preds)
                y_true.extend(trues)
        
        if task_config_name in ["irf_factors", "wellness_dims"]:
            target_dims = WELLNESS_DIMS if task_config_name == "wellness_dims" else IRF_DIMS
            dim_stats = {d: {'t': [], 'p': []} for d in target_dims}
            
            for text, t, p in zip(input_texts, y_true, y_pred):
                text_lower = text.lower()
                for d in target_dims:
                    if d.lower() in text_lower:
                        dim_stats[d]['t'].append(t)
                        dim_stats[d]['p'].append(p)
                        break
            
            log(f"\n📊 --- [{task_config_name}] Dimensional Report ---", file_f)
            log(f"{'Dim':<25} | {'Count':<8} | {'Accuracy':<10} | {'Dice (F1)'}", file_f)
            log("-" * 65, file_f)
            all_f1s = []
            for d in target_dims:
                yt = dim_stats[d]['t']
                yp = dim_stats[d]['p']
                if not yt:
                    log(f"{d:<25} | 0        | N/A        | N/A", file_f)
                    continue
                acc = accuracy_score(yt, yp)
                f1 = f1_score(yt, yp, pos_label=1, zero_division=0)
                all_f1s.append(f1)
                log(f"{d:<25} | {len(yt):<8} | {acc:.4f}     | {f1:.4f}", file_f)
            log("-" * 65, file_f)
            log(f"{'Average (Macro)':<25} | {len(y_true):<8} | N/A        | {np.mean(all_f1s):.4f}", file_f)
            log("\n--- [Standard Weighted Report Below] ---", file_f)

        if task_config_name in ENGLISH_TASKS:
            avg_type = 'weighted'
        else:
            avg_type = 'micro'
            
        log(f"[Metrics Logic] Task: {task_config_name} | Language: {'English' if avg_type=='weighted' else 'Chinese'} | Avg Type: {avg_type}", file_f)

        acc = accuracy_score(y_true, y_pred)
        p = precision_score(y_true, y_pred, average=avg_type, zero_division=0)
        r = recall_score(y_true, y_pred, average=avg_type, zero_division=0)
        f1 = f1_score(y_true, y_pred, average=avg_type, zero_division=0)
        
        log(f"Accuracy : {acc:.4f}", file_f)
        log(f"Precision: {p:.4f}", file_f)
        log(f"Recall   : {r:.4f}", file_f)
        log(f"F1 Score : {f1:.4f}", file_f)
        log("-" * 60, file_f)
        
        labels_list = list(range(len(cfg["labels"])))
        log(classification_report(y_true, y_pred, labels=labels_list, target_names=cfg["labels"], digits=4, zero_division=0), file_f)

        if task_config_name == "cognitive_path":
            calculate_parent_metrics(y_true, y_pred, COGNITIVE_PATH_PARENTS, cfg["labels"], file_f)

if __name__ == "__main__":
    os.makedirs(os.path.dirname(RESULT_TXT_PATH), exist_ok=True)
    
    with open(RESULT_TXT_PATH, "w", encoding="utf-8") as f:
        log(f"Evaluation Started", f)
        log(f"Seeds: {SEEDS}", f)
        log(f"Tasks: {len(EVAL_TASKS_LIST)}\n", f)
        log(f"Metric Strategy: Chinese->Micro, English->Weighted\n", f)

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
        
        for seed in SEEDS:
            model_path = MODEL_PATH_TEMPLATE.format(seed)
            log(f"\n{'#'*60}", f)
            log(f"Loading Seed {seed}: {model_path}", f)
            log(f"{'#'*60}", f)
            
            if not os.path.exists(model_path):
                log(f"Model file not found {model_path}, skipping.", f)
                continue
                
            model = UnifiedXLMV(MODEL_NAME, TASK_CONFIG)
            try:
                state_dict = torch.load(model_path, map_location=DEVICE)
                model.load_state_dict(state_dict)
                model.to(DEVICE)
                
                evaluate_tasks_individually(model, tokenizer, seed, f)
            except Exception as e:
                log(f"Error loading model: {e}", f)
    
    print(f"\nCompleted. Results saved to: {RESULT_TXT_PATH}")