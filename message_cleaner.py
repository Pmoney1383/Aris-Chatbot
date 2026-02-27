import re

INPUT_FILE = "data/+17787081028.txt"
OUTPUT_FILE = "clean_tagged.txt"

date_pattern = re.compile(r"\d{2}/\d{2}/\d{4}")
junk_exact = {"8 Ball", "Cup Pong", "Darts", "Chess", "[photo]", "[video]", "20 Questions", "Archery", "Sea Battle", "Dots & Boxes", "Basketball"}

def is_timestamp(line):
    return date_pattern.search(line) is not None

def is_junk(msg):
    msg = msg.strip()
    if not msg:
        return True
    if msg in junk_exact:
        return True
    if msg.startswith("[photo"):
        return True
    if msg.startswith("[video"):
        return True
    if all(not c.isalnum() for c in msg):
        return True
    if msg == "<url>":
        return True
    if len(msg.split()) == 1 and len(msg) <= 2:
        return True
    return False

url_pattern = re.compile(r'(https?://\S+|www\.\S+)')

def clean_message(msg):
    # remove system text
    if "Follow link" in msg:
        return None

    # lowercase everything
    msg = msg.lower()

    # replace urls with token
    msg = re.sub(r'(https?://\S+|www\.\S+)', '<url>', msg)

    # fix broken apostrophes spacing like "don’ t"
    msg = re.sub(r"\s+'\s+", "'", msg)
    msg = re.sub(r"’\s+", "’", msg)

    # remove duplicated spaces
    msg = re.sub(r"\s+", " ", msg)

    # trim leading/trailing space
    msg = msg.strip()

    return msg

messages = []

with open(INPUT_FILE, "r", encoding="utf-8", errors="ignore") as f:
    lines = f.readlines()

for line in lines:
    raw = line.rstrip("\n")

    if not raw.strip():
        continue

    if is_timestamp(raw):
        continue

    # detect indentation
    if raw.startswith(" " * 10):  # adjust if needed
        speaker = "<me>"
        msg = raw.strip()
    else:
        speaker = "<other>"
        msg = raw.strip()

    msg = clean_message(msg)
    if msg is None:
        continue

    if is_junk(msg):
        continue

    messages.append((speaker, msg))

# merge consecutive same-speaker messages
merged = []
prev_speaker = None
buffer = ""

for speaker, msg in messages:
    if speaker == prev_speaker:
        buffer += " " + msg
    else:
        if buffer:
            merged.append((prev_speaker, buffer.strip()))
        buffer = msg
        prev_speaker = speaker

if buffer:
    merged.append((prev_speaker, buffer.strip()))


with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for speaker, msg in merged:
        f.write(f"{speaker} {msg}\n")

print("Final cleaned messages:", len(merged))