# IA-AMA
The data and code for the paper: IA-AMA: An Instruction-Aware Asymmetric Modular Adapter Framework for Bilingual Mental Health Analysis

This repository provides:
- **Datasets** used in the paper (public links + release status).
- **Expert few-shot seeds** for GPT-4 explanation prompting: `Expert_data/`
- **Training and evaluation code** for IA-AMA+SCLA (our method): `IA-AMA_code/`
- **Cross-lingual small PLMs baselines**: `Multilingual_PLMs_Code/`
- **Mixed-SFT bilingual baselines (ms-swift)**: `Mixed-SFT Baselines (Bilingual) code/`

---


# Dataset

---

## 1. Dataset Overview

The overall statistics of the datasets used in this work are summarized in **Table 1** below.

### Table 1: Statistics of the datasets used in this work

| Data | Task | Instruction (train / val / test) | Source | Annotation | Type |
|------|------|----------------------------------|--------|------------|------|
| **English Datasets (D<sub>en</sub>)** |||||||
| DR | depression detection | 1,003 / 430 / 405 | Reddit | weak supervision | binary |
| Dreaddit | stress detection | 2,837 / 300 / 414 | Reddit | human annotation | binary |
| SAD | stress cause detection | 5,547 / 616 / 684 | SMS | human annotation | multi-class |
| MultiWD | wellness dimensions detection | 15,744 / 1,500 / 2,441 | Reddit | human annotation | multi-label |
| IRF | interpersonal risk factors detection | 3,943 / 985 / 2,113 | Reddit | human annotation | multi-label |
| **Chinese Datasets (D<sub>zh</sub>)** |||||||
| SocialCD | cognitive distortion detection | 2,043 / 682 / 682 | Weibo | human annotation | multi-label |
| CP | cognitive path extraction | 2,740 / 910 / 945 | Weibo | human annotation | multi-label |
| SR-2 | suicide risk detection | 2,694 / 674 / 843 | Weibo | human annotation | binary |
| SR-11 | fine-grained suicide risk detection | 2,694 / 674 / 843 | Weibo | human annotation | multi-class |
| C-IRF | interpersonal risk detection | 823 / 103 / 103 | Weibo | human annotation | multi-label |
| DS | depression severity detection | 29,200 / 3,649 / 3,649 | Weibo | human annotation | multi-class |
| **In total** | Mental health analysis | 69,268 / 10,523 / 13,122 | Social Media | – | – |

---

## 2. Materials for Fine-Tuning on Downstream Tasks

### 2.1 English Datasets

All English datasets are adopted from the **IMHI benchmark**, which integrates multiple publicly available mental health datasets collected from Reddit and SMS platforms.

- **IMHI Dataset (MentalLLaMA)**  
  GitHub: https://github.com/SteveKGYang/MentalLLaMA


---

### 2.2 Chinese Datasets

For Chinese mental health analysis, we use **C-IMHI from MentalGLM**, which are collected from Chinese social media (Weibo) and annotated by trained human annotators.

- **C-IMHI Dataset (MentalGLM)**  
  GitHub: https://github.com/zwzzzQAQ/MentalGLM


---

## 3. Newly Added Chinese Datasets

In addition to prior public datasets, we introduce **four new Chinese datasets** constructed from Weibo:

| Dataset | Task | Released |
|--------|------|---------------|
| **SR-2** | Suicide risk detection (binary) | **Yes** |
| **SR-11** | Fine-grained suicide risk detection (11 classes) | Coming soon |
| **C-IRF** | Interpersonal risk factor detection | Coming soon |
| **DS** | Depression severity detection | **Yes** |

- **DS (Depression Severity Dataset)** : 
  https://www.scidb.cn/en/detail?dataSetId=3913033adab24ee09ac757e1caa9a498

- **SR-11** and **C-IRF** are currently under internal validation and will be released in future versions (*coming soon*).

All newly added datasets are annotated by trained annotators following standardized psychological and clinical guidelines.

---


## 5. Base Models

The following pretrained models are used as backbones for fine-tuning and evaluation:

| Model | Source |
|-------|--------|
| GLM-4-9B-Chat | https://huggingface.co/THUDM/glm-4-9b-chat |
| LLaMA-3-8B-Chinese-Chat | https://huggingface.co/FlagAlpha/Llama3-Chinese-8B-Instruct |
| Qwen3-8B | https://huggingface.co/Qwen/Qwen3-8B |

---

## 6. Notes on Data Availability and Ethics

- Only **publicly released datasets** are distributed or linked in this repository.
- Some datasets (e.g., **SR-11**, **C-IRF**) are currently not publicly released and are therefore marked as *coming soon*.  
  Access to these datasets may be granted **upon reasonable request for research purposes**.
- All data are anonymized and collected in accordance with platform policies and research ethics requirements.

---


# Environment

We recommend **Python 3.10** with a **CUDA-enabled GPU** environment.

### Setup

```bash
conda create -n ia_ama python=3.10
conda activate ia_ama

pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.44.2 peft==0.12.0 accelerate==0.33.0
pip install sentencepiece protobuf tiktoken
pip install numpy scikit-learn matplotlib
```

### Dependencies

| Package | Version | Notes |
|---------|---------|-------|
| Python | 3.10 | |
| PyTorch | 2.3.1+cu121 | CUDA 12.1 build |
| transformers | 4.44.2 | For GLM-4; use ≥4.51 for Qwen3 |
| peft | 0.12.0 | Required for adapter stacking via `set_adapter([list])` |
| accelerate | 0.33.0 | |
| scikit-learn | ≥1.0 | For F1 evaluation |

If you plan to run Mixed-SFT baselines, also install ms-swift following its official guide: https://github.com/modelscope/ms-swift

---


# How to Run (IA-AMA_code)

## Code Structure

```
IA-AMA_code/
├── routing.py                  # Deterministic instruction-aware router
├── data_loader.py              # Expert-homogeneous batch sampler with DDP support
├── train_stage1_sft.py         # Stage 1: Task expert SFT training
├── train_stage2_cla.py         # Stage 2: Stacked CLA adapter training
├── evaluate_ia_ama.py          # Evaluation with stacked adapter inference
├── calibration_eval.py         # Clinical calibration (ECE / Brier / selective prediction)
├── calibration_metrics.py      # Calibration utilities
├── llm_classifier.py           # Completion-based classifier for calibration
├── create_label_spaces.py      # Generate label space files for all 11 tasks
└── convert_alignment_data.py   # Convert alignment pairs to training format
```

---

## Step 0: Prepare Label Spaces

```bash
cd IA-AMA_code
python create_label_spaces.py --out_dir label_spaces
```

## Step 1: Stage 1 — Task Expert SFT

Train three asymmetric LoRA experts with instruction-aware routing.

```bash
python -m torch.distributed.run --nproc_per_node=2 train_stage1_sft.py \
    --model_path /path/to/glm-4-9b-chat \
    --data_dir traindata \
    --out_dir ckpts/stage1 \
    --lora_targets query_key_value dense \
    --lora_en_r 16 --lora_risk_r 8 --lora_cbt_r 32 \
    --batch_size 2 --grad_accum 16 \
    --epochs 10 --max_length 1024 \
    --lr 2e-4 --dtype bf16 --num_workers 0
```

Output structure:
```
ckpts/stage1/epoch10/adapters/
├── lora_en/
├── lora_risk/
└── lora_cbt/
```

## Step 2: Stage 2 — Stacked CLA Adapter Training

Freeze all Stage 1 experts and train a shared cross-lingual alignment adapter.

```bash
python -m torch.distributed.run --nproc_per_node=2 train_stage2_cla.py \
    --model_path /path/to/glm-4-9b-chat \
    --stage1_dir ckpts/stage1/epoch10 \
    --align_file alignment_pairs.json \
    --out_dir ckpts/stage2 \
    --lora_targets query_key_value dense \
    --lora_cla_r 16 --lora_cla_alpha 32 \
    --batch_size 4 --epochs 5 --lr 5e-5 \
    --dtype bf16 --num_workers 0
```

## Step 3: Evaluation

**Full IA-AMA+SCLA (with CLA):**

```bash
python evaluate_ia_ama.py \
    --model_path /path/to/glm-4-9b-chat \
    --ia_ama \
    --adapter_en   ckpts/stage1/epoch10/adapters/lora_en \
    --adapter_risk ckpts/stage1/epoch10/adapters/lora_risk \
    --adapter_cbt  ckpts/stage1/epoch10/adapters/lora_cbt \
    --adapter_cla  ckpts/stage2/epoch5/adapters/lora_cla \
    --test_data_dir testdata \
    --labels_dir label_spaces \
    --out_dir results/eval \
    --dtype bf16
```

**Ablation without CLA (omit `--adapter_cla`, no retraining needed):**

```bash
python evaluate_ia_ama.py \
    --model_path /path/to/glm-4-9b-chat \
    --ia_ama \
    --adapter_en   ckpts/stage1/epoch10/adapters/lora_en \
    --adapter_risk ckpts/stage1/epoch10/adapters/lora_risk \
    --adapter_cbt  ckpts/stage1/epoch10/adapters/lora_cbt \
    --test_data_dir testdata \
    --labels_dir label_spaces \
    --out_dir results/eval_no_cla \
    --dtype bf16
```

## Step 4 (Optional): Clinical Calibration

```bash
python calibration_eval.py \
    --backbone glm \
    --model_path /path/to/glm-4-9b-chat \
    --ia_ama \
    --adapter_en   ckpts/stage1/epoch10/adapters/lora_en \
    --adapter_risk ckpts/stage1/epoch10/adapters/lora_risk \
    --adapter_cbt  ckpts/stage1/epoch10/adapters/lora_cbt \
    --adapter_cla  ckpts/stage2/epoch5/adapters/lora_cla \
    --val_file data/val_merged.json \
    --test_file data/test_merged.json \
    --labels_dir label_spaces \
    --out_dir results/calibration
```

---


