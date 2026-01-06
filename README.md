# IA-AMA
The data and code for the paper: BiMentalGLM: An Instruction-Aware Asymmetric Modular Adapter Framework for Bilingual Mental Health Analysis


# Dataset

This repository provides datasets and resources for fine-tuning and evaluating large language models on **mental health analysis tasks** across English and Chinese social media platforms.

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

The following pretrained model is used as base models for fine-tuning and evaluation:


- **GLM-4-9B-Chat**  
  https://huggingface.co/THUDM/glm-4-9b-chat

---

## 6. Notes on Data Availability and Ethics

- Only **publicly released datasets** are distributed or linked in this repository.
- Some datasets (e.g., **SR-11**, **C-IRF**) are currently not publicly released and are therefore marked as *coming soon*.  
  Access to these datasets may be granted **upon reasonable request for research purposes**.
- All data are anonymized and collected in accordance with platform policies and research ethics requirements.

---

## 7. Citation

If you use these datasets or resources, please cite the corresponding papers:

- MentalLLaMA
- MentalGLM
- This work
