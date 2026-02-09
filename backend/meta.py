import json

# --------------------------------------------------
# FILE PATHS
# --------------------------------------------------

CAPTIONS_PATH = "../data/tanishq/florence_captions_all.json"
OUTPUT_PATH = "../data/tanishq/generated_metadata.json"

# --------------------------------------------------
# CONTROLLED VOCABULARY
# --------------------------------------------------

CATEGORIES = ["ring", "necklace"]

METALS = [
    "gold",
    "silver",
    "white gold",
    "rose gold"
]

STONES = [
    "diamond",
    "pearl",
    "emerald",
    "ruby",
    "sapphire",
    "amethyst",
    "onyx",
    "aquamarine",
    "topaz",
    "none"
]

STUDDED_TERMS = [
    "diamond", "emerald", "ruby",
    "sapphire", "pearl", "gemstone",
    "halo", "cluster", "studded"
]

# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def normalize_text(text):
    return text.lower()


def extract_category(text):
    for c in CATEGORIES:
        if c in text:
            return c
    return "ring"  # fallback


def extract_metal(text):
    for m in METALS:
        if m in text:
            return m
    return "gold"  # dataset dominant fallback


def extract_stone(text):
    for s in STONES:
        if s != "none" and s in text:
            return s
    return "none"


def extract_form(text, stone):
    if stone != "none":
        return "studded"

    for term in STUDDED_TERMS:
        if term in text:
            return "studded"

    return "plain"


def build_metadata_text(category, metal, stone):

    if stone == "none":
        return f"{category} made of {metal}"

    return f"{category} made of {metal} with {stone}"


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def generate_metadata():

    with open(CAPTIONS_PATH, "r") as f:
        captions = json.load(f)

    metadata = {}

    for item in captions:
        image_id = item.get("image")
        caption = item.get("caption")

        if not image_id or not caption:
             continue

        text = normalize_text(caption)

        category = extract_category(text)
        metal = extract_metal(text)
        stone = extract_stone(text)
        form = extract_form(text, stone)

        metadata_text = build_metadata_text(
            category, metal, stone
        )

        metadata[image_id] = {
            "category": category,
            "metal": metal,
            "primary_stone": stone,
            "form": form,
            "metadata_text": metadata_text
        }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ Generated metadata for {len(metadata)} images")


# --------------------------------------------------

if __name__ == "__main__":
    generate_metadata()
