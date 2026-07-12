"""Chat template + loss-mask encoding shared by SFT data prep, SFT training,
and the chat REPL.

The template (tiktoken gpt2, no tokenizer retraining):

    <|user|>
    {user message}
    <|assistant|>
    {assistant message}
    <|eot|>

Since tiktoken can't be extended, the delimiters are not single special tokens
but fixed multi-token sequences (verified against tiktoken gpt2):

    <|user|>      -> [27, 91, 7220, 91, 29]
    <|assistant|> -> [27, 91, 562, 10167, 91, 29]
    <|eot|>       -> [27, 91, 68, 313, 91, 29]

They are always encoded piecewise (delimiter and message text tokenized
separately, then concatenated) so the exact same id sequences appear in
training data and at inference time, regardless of BPE merges across the
boundary.

Loss mask (uint8, one per token):
    0 = user-turn tokens (fed as context, excluded from the loss)
    1 = assistant-turn tokens, including <|assistant|> and <|eot|>
"""

import tiktoken

USER_TOKEN = "<|user|>"
ASST_TOKEN = "<|assistant|>"
EOT_TOKEN = "<|eot|>"

_enc = tiktoken.get_encoding("gpt2")

USER_IDS = _enc.encode_ordinary(USER_TOKEN)    # [27, 91, 7220, 91, 29]
ASST_IDS = _enc.encode_ordinary(ASST_TOKEN)    # [27, 91, 562, 10167, 91, 29]
EOT_IDS = _enc.encode_ordinary(EOT_TOKEN)      # [27, 91, 68, 313, 91, 29]
NEWLINE_IDS = _enc.encode_ordinary("\n")       # [198]


def get_encoding():
    return _enc


def encode_user_turn(text: str) -> tuple[list[int], list[int]]:
    """<|user|>\\n{text}\\n — all mask 0."""
    ids = USER_IDS + _enc.encode_ordinary("\n" + text.strip() + "\n")
    return ids, [0] * len(ids)


def encode_assistant_turn(text: str) -> tuple[list[int], list[int]]:
    """<|assistant|>\\n{text}\\n<|eot|>\\n — all mask 1."""
    ids = (
        ASST_IDS
        + _enc.encode_ordinary("\n" + text.strip() + "\n")
        + EOT_IDS
        + NEWLINE_IDS
    )
    return ids, [1] * len(ids)


def encode_conversation(turns: list[tuple[str, str]]) -> tuple[list[int], list[int]]:
    """Encode [(role, text), ...] (role: "user" | "assistant") into
    (token_ids, loss_mask) with the chat template applied to every turn."""
    ids: list[int] = []
    mask: list[int] = []
    for role, text in turns:
        if role == "user":
            t_ids, t_mask = encode_user_turn(text)
        elif role == "assistant":
            t_ids, t_mask = encode_assistant_turn(text)
        else:
            raise ValueError(f"unknown role {role!r}")
        ids.extend(t_ids)
        mask.extend(t_mask)
    return ids, mask


def encode_chat_prompt(user_text: str) -> list[int]:
    """<|user|>\\n{text}\\n<|assistant|>\\n — what the REPL feeds the model
    before generation. Generation stops when EOT_IDS is produced."""
    return (
        USER_IDS
        + _enc.encode_ordinary("\n" + user_text.strip() + "\n")
        + ASST_IDS
        + NEWLINE_IDS
    )
