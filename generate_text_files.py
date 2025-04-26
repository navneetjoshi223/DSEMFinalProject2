import os

# Just 5 Italian sample sentences
italian_texts = [
    "Ciao, come stai oggi?",
    "Il cielo è molto limpido stamattina.",
    "Ho comprato del pane fresco al mercato.",
    "Il treno partirà alle nove in punto.",
    "Amo ascoltare la musica classica la sera."
]

# Folder paths
base_path = "saved_models/mls/sample16/samples/step_0"
guide_versions = ["guide2.0", "guide3.0", "guide5.0"]

# Create text files in each guide folder
for guide in guide_versions:
    guide_path = os.path.join(base_path, guide)
    os.makedirs(guide_path, exist_ok=True)
    for idx, text in enumerate(italian_texts):
        text_filename = os.path.join(guide_path, f"text_{idx}.txt")
        with open(text_filename, "w") as f:
            f.write(text)
        print(f"Created {text_filename}")