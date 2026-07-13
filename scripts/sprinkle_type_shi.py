"""Sprinkle 'type shi' into persona_synthetic.jsonl at gold-like density.

Gold first-25 rate is ~16%. Target overall ~15% so SFT picks up the tic
without overfitting. First 25 lines left untouched.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PATH = ROOT / "data" / "raw" / "persona_synthetic.jsonl"
SEED = 20260713
KEEP_FIRST = 25
TARGET_RATE = 0.15  # match gold verbal-tic density

# answers that already feel finished — don't glue type shi on
SKIP_END_RE = re.compile(
    r"(?:\?|!|type shi\.?|shall i explain|you got it gang|you got homie)$",
    re.IGNORECASE,
)

# short punch answers (capitals, one-liners) — slang already tight
SHORT_MAX = 80

TYPE_SHI_FORMS = [
    " type shi",
    " type shi.",
    " type shi 💀",
    " type shi 🔥",
]


def has_type_shi(text: str) -> bool:
    return "type shi" in text.lower()


def can_append(text: str) -> bool:
    t = text.rstrip()
    if not t or has_type_shi(t):
        return False
    if SKIP_END_RE.search(t):
        return False
    if len(t) < 40:
        return False  # too short — "paris no cap" energy
    if len(t) > 900:
        return False  # long task templates — leave alone mostly
    # avoid stacking on another soft tail emoji-only
    return True


def append_type_shi(text: str, rng: random.Random) -> str:
    """Append in gold positions: mostly end-of-answer, rarely mid-clause."""
    t = text.rstrip()
    # mid-clause contrast ~8% when there's a natural sentence boundary later
    if rng.random() < 0.08 and ". " in t and not t.count("\n"):
        parts = t.split(". ")
        if len(parts) >= 2 and 20 < len(parts[0]) < 180:
            # "...X type shi. rest"
            return parts[0].rstrip() + " type shi. " + ". ".join(parts[1:])
    form = rng.choice(TYPE_SHI_FORMS)
    # prefer no emoji if answer already has one
    if any(e in t for e in "💀🔥🙌😭") and form.endswith(("💀", "🔥")):
        form = rng.choice([" type shi", " type shi."])
    return t + form


def main() -> None:
    rng = random.Random(SEED + 7)
    rows = [
        json.loads(l)
        for l in PATH.read_text(encoding="utf-8").splitlines()
    ]
    n = len(rows)
    have = sum(1 for r in rows if has_type_shi(r["assistant"]))
    target = int(TARGET_RATE * n)
    need = max(0, target - have)

    # eligible indices among rest
    eligible = [
        i for i in range(KEEP_FIRST, n)
        if can_append(rows[i]["assistant"])
    ]
    rng.shuffle(eligible)

    # mildly prefer basic explain / identity-ish answers
    def score(i: int) -> float:
        u = rows[i]["user"].lower()
        a = rows[i]["assistant"]
        s = 0.0
        if any(k in u for k in ("what is", "what's", "explain", "meaning", "what does")):
            s += 2.0
        if any(k in a.lower() for k in ("parsa", "aris", "i'm an ai", "i am")):
            s += 1.5
        if "\n" in a:
            s -= 1.0
        if 60 <= len(a) <= 350:
            s += 1.0
        return s + rng.random()  # tie-break

    eligible.sort(key=score, reverse=True)
    pick = eligible[:need]

    added = 0
    for i in pick:
        rows[i]["assistant"] = append_type_shi(rows[i]["assistant"], rng)
        added += 1

    PATH.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )

    final = sum(1 for r in rows if has_type_shi(r["assistant"]))
    gold = sum(1 for r in rows[:KEEP_FIRST] if has_type_shi(r["assistant"]))
    print(f"kept first {KEEP_FIRST} unchanged")
    print(f"added type shi to {added} answers (needed ~{need} to hit {TARGET_RATE:.0%})")
    print(f"gold rate: {gold}/{KEEP_FIRST} ({100*gold/KEEP_FIRST:.1f}%)")
    print(f"overall:   {final}/{n} ({100*final/n:.1f}%)")


if __name__ == "__main__":
    main()
