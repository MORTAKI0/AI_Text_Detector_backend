from app import inference
from app.settings import get_settings

# WARNING: These examples are out-of-domain vs SemEval Task 8 and may misclassify.
SAMPLES = [
    {
        "name": "wiki_style_rainforest",
        "source": "human",
        "text": (
            "The Amazon rainforest is a vast tropical forest in South America. "
            "It covers millions of square kilometers, hosts an extraordinary range "
            "of plant and animal life, and plays a major role in regional climate and rainfall."
        ),
    },
    {
        "name": "nasa_style_rover",
        "source": "human",
        "text": (
            "NASA's Perseverance rover has been exploring an ancient lakebed on Mars. "
            "The mission is collecting rock cores that may help scientists understand "
            "past environments and potential signs of ancient microbial life."
        ),
    },
    {
        "name": "gutenberg_style_novel",
        "source": "human",
        "text": (
            "It is a truth universally acknowledged, that a single person in possession "
            "of a good fortune, must be in want of a partner."
        ),
    },
    {
        "name": "meeting_notes",
        "source": "human",
        "text": (
            "Agenda: review Q2 roadmap, confirm launch dates, and assign owners for the "
            "documentation update. Action items: update the release checklist and share a draft by Friday."
        ),
    },
    {
        "name": "ai_generic_marketing",
        "source": "ai",
        "text": (
            "In today's fast-paced digital landscape, leveraging innovative solutions "
            "is essential to maximize efficiency and unlock new growth opportunities."
        ),
    },
    {
        "name": "ai_essay_summary",
        "source": "ai",
        "text": (
            "The central theme of the passage is the balance between individual agency "
            "and structural constraints, illustrating how personal choices interact with wider systems."
        ),
    },
    {
        "name": "ai_recipe_style",
        "source": "ai",
        "text": (
            "To prepare the dish, combine the ingredients in a large bowl, mix until smooth, "
            "and bake at a moderate temperature until the top turns golden and fragrant."
        ),
    },
    {
        "name": "ai_policy_statement",
        "source": "ai",
        "text": (
            "This policy aims to improve transparency and accountability by establishing "
            "clear guidelines, measurable outcomes, and regular review cycles."
        ),
    },
]


def main():
    settings = get_settings()
    print("Threshold:", settings.AI_THRESHOLD)
    print("Label meaning: 0=Human, 1=AI")
    print("")

    for item in SAMPLES:
        result = inference.analyze_text(item["text"])
        print(
            f"{item['name']:<22} source={item['source']:<5} "
            f"label={result['label']} prob_ai={result['prob_ai']:.4f}"
        )


if __name__ == "__main__":
    main()
