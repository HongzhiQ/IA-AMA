
import argparse
import json
import random


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in_file",  required=True)
    p.add_argument("--out_file", required=True)
    p.add_argument("--max_pairs", type=int, default=5000,
                   help="Cap the number of pairs. Paper uses 5k.")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    with open(args.in_file, "r", encoding="utf-8") as f:
        raw = json.load(f)
    print(f"[convert] loaded {len(raw)} raw pairs from {args.in_file}")

    out = []
    skipped = 0
    for item in raw:
        zh = (item.get("zh") or item.get("zh_text") or "").strip()
        en = (item.get("en") or item.get("en_text") or "").strip()
        if not zh or not en:
            skipped += 1
            continue
        out.append({
            "en_text":   en,
            "zh_text":   zh,
            "zh_expert": "lora_cbt",
        })

    if args.max_pairs and len(out) > args.max_pairs:
        random.seed(args.seed)
        random.shuffle(out)
        out = out[:args.max_pairs]
        print(f"[convert] sub-sampled to {args.max_pairs} pairs (seed={args.seed})")

    with open(args.out_file, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"[convert] wrote {len(out)} pairs -> {args.out_file}")
    if skipped:
        print(f"[convert] skipped {skipped} malformed entries")
    print(f"[convert] all pairs routed through: lora_cbt (therapist expert)")


if __name__ == "__main__":
    main()
