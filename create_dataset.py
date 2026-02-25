import os

DATA_DIR = "data"
FILE_NAME = "greeting_dataset.txt"
FILE_PATH = os.path.join(DATA_DIR, FILE_NAME)

def ensure_data_folder():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

def save_entry(category, pattern, response):
    with open(FILE_PATH, "a", encoding="utf-8") as f:
        line = f"[{category}] {pattern.strip().lower()} => {response.strip().lower()}\n"
        f.write(line)

def main():
    ensure_data_folder()

    print("Custom Dataset Builder")
    print("Type 'exit' at any time to stop.\n")

    while True:
        category = "greeting"  # For simplicity, we use a single category. You can expand this as needed.

        pattern = input("Input pattern: ")
        if pattern.lower() == "exit":
            break

        response = input("Response: ")
        if response.lower() == "exit":
            break

        save_entry(category, pattern, response)
        print("Saved.\n")

    print("\nFinished. Data saved to:", FILE_PATH)

if __name__ == "__main__":
    main()