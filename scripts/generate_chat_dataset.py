#!/usr/bin/env python3
"""
Generate a comprehensive chat dataset (3–5k examples) for fine-tuning Bloom-560M.

Each JSON Lines entry contains:
    {
        "system": "You are a helpful assistant.",
        "user": "<user message>",
        "assistant": "<assistant response>"
    }

The script augments the existing `data/chat_data.jsonl` file (if present) until it
reaches `TARGET_SIZE` examples. It covers:
    • General chat / chit-chat
    • Task-related queries (reminders, notes, ledger entries)
    • Factual Q&A (geography, history, science, dates)

Run:
    python scripts/generate_chat_dataset.py

You can change `TARGET_SIZE` if you need a larger dataset.
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timedelta
from pathlib import Path

# -------------------------------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------------------------------
TARGET_SIZE = 5000  # Desired total number of rows after generation
DATA_PATH = Path("data/chat_data.jsonl")
SYSTEM_PROMPT = "You are a helpful assistant."

# -------------------------------------------------------------------------------------------------
# Helper template pools
# -------------------------------------------------------------------------------------------------
CHAT_TEMPLATES = [
    ("Hi, how are you?", "I'm doing great! How can I assist you today?"),
    ("Tell me a joke", "Why don't scientists trust atoms? Because they make up everything!"),
    ("What's your favorite color?", "I love the color blue – it's calming and reminds me of the ocean. What's yours?"),
    ("Good morning!", "Good morning! Hope you have a fantastic day ahead. How can I help?"),
    ("Do you like music?", "I enjoy learning about all kinds of music. What genres do you like?"),
    ("I'm feeling sad today", "I'm sorry to hear that. I'm here for you – would talking about it help?"),
]

FACT_TEMPLATES = [
    ("What is the population of {country}?", "The population of {country} is approximately {population} million."),
    ("What's the capital of {country}?", "The capital of {country} is {capital}."),
    ("How many days until {holiday}?", "There are {days} days until {holiday}."),
    ("Who invented the telephone?", "Alexander Graham Bell is credited with inventing the telephone in 1876."),
    ("What's the tallest mountain in the world?", "Mount Everest is the tallest, standing about 8,849 meters (29,032 ft) above sea level."),
]

COUNTRY_INFO = [
    ("India", "New Delhi", 1428),
    ("France", "Paris", 67),
    ("Germany", "Berlin", 84),
    ("Japan", "Tokyo", 126),
    ("Brazil", "Brasília", 214),
]

HOLIDAYS = ["Christmas", "New Year", "Independence Day", "Thanksgiving"]

TASK_INTENTS = {
    "reminder": [
        "Remind me to {action} at {time}",
        "Set a reminder for {time} to {action}",
        "Don't let me forget to {action} at {time}"
    ],
    "note": [
        "Add a note to {note}",
        "Note that {note}",
        "Please write down: {note}"
    ],
    "ledger": [
        "{person} owes me ${amount}",
        "Add to the ledger that {person} owes ${amount}",
        "Track that I owe {person} ${amount}"
    ]
}

ACTIONS = ["call John", "buy groceries", "send the email", "attend the meeting", "pay the bills"]
NAMES = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Heidi", "Ivan"]
NOTES = [
    "buy chocolates",
    "schedule a dentist appointment",
    "finish the project report",
    "plan the weekend trip",
    "water the plants"
]

# -------------------------------------------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------------------------------------------

def load_existing(path: Path):
    """Load existing dataset; return list[Any]."""
    examples = []
    if path.exists():
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    examples.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # skip malformed
    return examples


def save_examples(path: Path, new_examples: list[dict]):
    """Append new examples to the dataset file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for ex in new_examples:
            json.dump(ex, f, ensure_ascii=False)
            f.write("\n")


# -------------------------------------------------------------------------------------------------
# Example generators
# -------------------------------------------------------------------------------------------------

def random_time() -> str:
    hour = random.randint(0, 23)
    minute = random.choice([0, 15, 30, 45])
    return f"{hour % 12 or 12}:{minute:02d} {'AM' if hour < 12 else 'PM'}"


def generate_task_example() -> tuple[str, str]:
    intent = random.choice(list(TASK_INTENTS.keys()))
    template = random.choice(TASK_INTENTS[intent])

    if intent == "reminder":
        user_msg = template.format(action=random.choice(ACTIONS), time=random_time())
        assistant_msg = "Reminder set: " + user_msg.split(" to ")[-1].capitalize() + "."

    elif intent == "note":
        note = random.choice(NOTES)
        user_msg = template.format(note=note)
        assistant_msg = f"Note added: {note}."

    else:  # ledger
        person = random.choice(NAMES)
        amount = random.choice([10, 20, 50, 75, 100, 150, 200, 500])
        user_msg = template.format(person=person, amount=amount)
        if "owe" in user_msg.lower().split():
            if "owes me" in user_msg:
                assistant_msg = f"Ledger updated: {person} owes you ${amount}."
            else:
                assistant_msg = f"Ledger updated: You owe {person} ${amount}."
        else:
            assistant_msg = f"Ledger updated for {person}."

    return user_msg, assistant_msg


def generate_chat_example() -> tuple[str, str]:
    user_msg, assistant_msg = random.choice(CHAT_TEMPLATES)
    return user_msg, assistant_msg


def generate_fact_example() -> tuple[str, str]:
    template_q, template_a = random.choice(FACT_TEMPLATES)

    if "population" in template_q:
        country, _, population = random.choice(COUNTRY_INFO)
        user_msg = template_q.format(country=country)
        assistant_msg = template_a.format(country=country, population=population)

    elif "capital" in template_q:
        country, capital, _ = random.choice(COUNTRY_INFO)
        user_msg = template_q.format(country=country)
        assistant_msg = template_a.format(country=country, capital=capital)

    elif "days until" in template_q:
        holiday = random.choice(HOLIDAYS)
        today = datetime.utcnow().date()
        # pick next occurrence of the holiday in the same year for simplicity
        target_date = today + timedelta(days=random.randint(5, 300))
        user_msg = template_q.format(holiday=holiday)
        days = (target_date - today).days
        assistant_msg = template_a.format(days=days, holiday=holiday)

    else:
        # Exact Q&A templates that don't need formatting
        user_msg = template_q
        assistant_msg = template_a

    return user_msg, assistant_msg


# Weighted sampling probabilities for each category
CATEGORY_GENERATORS = [
    (generate_chat_example, 0.4),      # 40% chit-chat
    (generate_task_example, 0.35),     # 35% task-related
    (generate_fact_example, 0.25),     # 25% factual
]


def sample_category() -> tuple[str, str]:
    r = random.random()
    cumulative = 0.0
    for gen_fn, prob in CATEGORY_GENERATORS:
        cumulative += prob
        if r < cumulative:
            return gen_fn()
    # fallback (shouldn't happen)
    return generate_chat_example()


# -------------------------------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------------------------------

def main():
    random.seed(1234)
    existing = load_existing(DATA_PATH)
    current_size = len(existing)

    to_generate = TARGET_SIZE - current_size
    if to_generate <= 0:
        print(f"Dataset already has {current_size} examples (>= {TARGET_SIZE}). No generation needed.")
        return

    print(f"Generating {to_generate} new examples to reach {TARGET_SIZE} total…")
    new_examples: list[dict] = []
    for _ in range(to_generate):
        user_msg, assistant_msg = sample_category()
        new_examples.append({
            "system": SYSTEM_PROMPT,
            "user": user_msg,
            "assistant": assistant_msg,
        })

    save_examples(DATA_PATH, new_examples)
    print(f"Done! Added {to_generate} rows. Total dataset size is now {TARGET_SIZE}.")


if __name__ == "__main__":
    main() 