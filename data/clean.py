import re
import os
import inflect
from pathlib import Path
import shutil

# Engine for number → words
p = inflect.engine()

# Directory where all raw and cleanHarry Potter .txt files are stored
base_dir = Path(__file__).resolve().parent
raw_dir = base_dir / "raw"
clean_dir = base_dir / "clean"


# Reset clean_dir each run
if clean_dir.exists():
    shutil.rmtree(clean_dir)   # delete folder and contents
clean_dir.mkdir(parents=True, exist_ok=True)  # recreate empty


# List of (filename, book_title_to_remove)
books = [
    ("01 Harry Potter and the Sorcerers Stone.txt", "HP 1 - Harry Potter and the Sorcerer's Stone"),
    ("02 Harry Potter and the Chamber of Secrets.txt", "HP 2 - Harry Potter and The Chamber of Secrets"),
    ("03 Harry Potter and the Prisoner of Azkaban.txt", "Harry Potter And The Prisoner of Azkaban"),
    ("04 Harry Potter and the Goblet of Fire.txt", "HP 4 - Harry Potter and The Goblet of Fire"),
    ("05 Harry Potter and the Order of the Phoenix.txt", "Harry Potter and the Order of the Phoenix"),
    ("06 Harry Potter and the Half-Blood Prince.txt", "Harry Potter and the Half-Blood Prince"),
    ("07 Harry Potter and the Deathly Hallows.txt", "Harry Potter and the Deathly Hallows"),
]

# Helper: convert chapter number (arabic, roman, or word) to Title Case word
def normalize_chapter_number(token: str) -> str:
    token = token.strip()
    if token.isdigit():  # e.g. "9"
        return p.number_to_words(int(token)).title()  # "Nine"
    return token.title()  # fallback for Roman numerals/words

def clean_file(file_path: Path, book_title: str):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    cleaned_lines = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # 1) Skip book title lines
        if line == book_title:
            i += 1
            continue

        # 2) Case: "Chapter X - Title" (already correct, just normalize title to uppercase)
        match = re.match(r"^(Chapter)\s+(\w+)\s*-\s*(.+)$", line, re.IGNORECASE)
        if match:
            chapter_word = match.group(1).title()     # "Chapter"
            chap_num = normalize_chapter_number(match.group(2))  # "9" -> "Nine"
            chapter_title = match.group(3).upper()    # force uppercase
            cleaned_lines.append(f"{chapter_word} {chap_num} - {chapter_title}\n\n")
            i += 1
            continue

        # 3) Case: "Chapter X : Title" → replace ":" with "-", normalize
        match = re.match(r"^(Chapter)\s+(\w+)\s*:\s*(.+)$", line, re.IGNORECASE)
        if match:
            chapter_word = match.group(1).title()
            chap_num = normalize_chapter_number(match.group(2))  # normalize "9" → "Nine"
            chapter_title = match.group(3).upper()
            cleaned_lines.append(f"{chapter_word} {chap_num} - {chapter_title}\n\n")
            i += 1
            continue

        # 4) Case: Chapter header alone ("CHAPTER N")
        if line.upper().startswith("CHAPTER"):
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                chap_num = normalize_chapter_number(parts[1])
                chapter_header = f"Chapter {chap_num}"
            else:
                chapter_header = "Chapter"

            # Find the next non-empty line = title
            j = i + 1
            while j < len(lines) and lines[j].strip() == "":
                j += 1

            if j < len(lines):
                chapter_title = lines[j].strip().upper()  # force uppercase
                combined = f"{chapter_header} - {chapter_title}\n\n"
                cleaned_lines.append(combined)
                i = j + 1
                continue

        # Otherwise, keep line as-is
        cleaned_lines.append(lines[i])
        i += 1

    # Write cleaned file into clean/ folder with same name
    out_path = clean_dir / file_path.name
    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(cleaned_lines)

    print(f"✅ Cleaned {file_path.name}")

# Loop through all books
for filename, title in books:
    clean_file(raw_dir / filename, title)


print(f"🔵 Generating input text...")


base_dir = os.path.dirname(__file__)
input_dir = Path(os.path.join(base_dir, 'clean'))
all_text = ""

for i, file_path in enumerate(sorted(input_dir.glob("*.txt")), 1):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    all_text += f"\n\n[BOOK {i} START]\n\n{text}\n\n[BOOK {i} END]\n\n"

# save as one big input.txt
input_file_path = os.path.join(input_dir, 'input.txt')
with open(input_file_path, "w", encoding="utf-8") as f:
    f.write(all_text)


# Reads entire harry potter series into memory as a single string data.
with open(input_file_path, 'r') as f:
    data = f.read() # data: a single string
print(f"length of dataset in characters: {len(data):,}")
