from typing import Dict, Optional

EXPERT_EN   = "lora_en"
EXPERT_RISK = "lora_risk"
EXPERT_CBT  = "lora_cbt"

TASK_KEY_TO_EXPERT: Dict[str, str] = {

    "detect_depression": EXPERT_EN,  # DR
    "detect_stress":     EXPERT_EN,  # Dreaddit
    "stress_cause":      EXPERT_EN,  # SAD
    "irf_factors":       EXPERT_EN,  # IRF
    "wellness_dims":     EXPERT_EN,  # MultiWD

    "cognitive_distortion": EXPERT_CBT,  # SocialCD
    "cognitive_path":       EXPERT_CBT,  # CP

    "suicide_coarse":   EXPERT_RISK,  # SR-2
    "suicide_scale":    EXPERT_RISK,  # SR-11
    "depression_level": EXPERT_RISK,  # DS
    "relation":         EXPERT_RISK,  # C-IRF
}

RISK_TASKS = [k for k, v in TASK_KEY_TO_EXPERT.items() if v == EXPERT_RISK]
CBT_TASKS  = [k for k, v in TASK_KEY_TO_EXPERT.items() if v == EXPERT_CBT]
EN_TASKS   = [k for k, v in TASK_KEY_TO_EXPERT.items() if v == EXPERT_EN]


def _normalize_key(s: Optional[str]) -> str:
    return str(s).lower().strip() if s else ""


def _cjk_ratio(text: str) -> float:
    if not text:
        return 0.0
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    total = sum(1 for ch in text if not ch.isspace())
    return cjk / total if total > 0 else 0.0


_CBT_KEYWORDS_ZH = ("认知扭曲", "认知路径", "自动思维", "负性思维",
                    "思维模式", "认知重构", "核心信念")
_RISK_KEYWORDS_ZH = ("自杀", "自残", "轻生", "想死", "结束生命",
                     "活不下去", "危机", "伤害自己", "抑郁程度",
                     "人际风险", "抑郁严重", "抑郁等级")


def route(sample: dict,
          default: str = EXPERT_RISK,
          verbose: bool = False) -> str:

    tk = _normalize_key(sample.get("task_key"))
    if tk in TASK_KEY_TO_EXPERT:
        if verbose:
            print(f"[route] task_key={tk} -> {TASK_KEY_TO_EXPERT[tk]}")
        return TASK_KEY_TO_EXPERT[tk]

    for field in ("dataset", "source", "data_source", "dataset_name"):
        val = _normalize_key(sample.get(field))
        if val in TASK_KEY_TO_EXPERT:
            if verbose:
                print(f"[route] {field}={val} -> {TASK_KEY_TO_EXPERT[val]}")
            return TASK_KEY_TO_EXPERT[val]

    instruction = (sample.get("instruction") or "").strip()
    input_text  = (sample.get("input") or sample.get("original_input") or "").strip()
    full_text = instruction + "\n" + input_text
    cjk = _cjk_ratio(full_text)
    if cjk < 0.15:
        if verbose:
            print(f"[route] cjk_ratio={cjk:.2f} -> lora_en")
        return EXPERT_EN

    if any(kw in full_text for kw in _CBT_KEYWORDS_ZH):
        if verbose:
            print("[route] zh cbt keyword matched -> lora_cbt")
        return EXPERT_CBT
    if any(kw in full_text for kw in _RISK_KEYWORDS_ZH):
        if verbose:
            print("[route] zh risk keyword matched -> lora_risk")
        return EXPERT_RISK

    if verbose:
        print(f"[route] fallback -> {default}")
    return default


def get_routing_decision(instruction: str,
                         input_text: str = "",
                         task_key: Optional[str] = None,
                         dataset: Optional[str] = None) -> str:

    return route({
        "instruction": instruction,
        "input":       input_text,
        "task_key":    task_key,
        "dataset":     dataset,
    })


def route_eval_sample(sample: dict, default_adapter: str = EXPERT_RISK) -> str:

    return route(sample, default=default_adapter)


if __name__ == "__main__":
    tests = [
        ({"task_key": "detect_depression"},    EXPERT_EN),
        ({"task_key": "detect_stress"},        EXPERT_EN),
        ({"task_key": "stress_cause"},         EXPERT_EN),
        ({"task_key": "irf_factors"},          EXPERT_EN),
        ({"task_key": "wellness_dims"},        EXPERT_EN),
        ({"task_key": "cognitive_distortion"}, EXPERT_CBT),
        ({"task_key": "cognitive_path"},       EXPERT_CBT),
        ({"task_key": "suicide_coarse"},       EXPERT_RISK),
        ({"task_key": "suicide_scale"},        EXPERT_RISK),
        ({"task_key": "depression_level"},     EXPERT_RISK),
        ({"task_key": "relation"},             EXPERT_RISK),
        # fallback
        ({"instruction": "Is this post showing signs of depression?"}, EXPERT_EN),
        ({"instruction": "这段文本有自杀倾向吗"}, EXPERT_RISK),
        ({"instruction": "分析这段话中的认知扭曲"}, EXPERT_CBT),
        ({"task_key": "depression_level",
          "instruction": "根据 PHQ-9 量表判断抑郁程度"}, EXPERT_RISK),
    ]
    passed = 0
    for i, (s, exp) in enumerate(tests):
        got = route(s)
        mark = "OK  " if got == exp else "FAIL"
        if got == exp:
            passed += 1
        print(f"[{mark}] case {i}: expected={exp}, got={got}  sample={s}")
    print(f"\n{passed}/{len(tests)} passed")
