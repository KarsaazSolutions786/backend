#!/usr/bin/env python3
"""
Expand chat_data.jsonl to at least 2000 examples.
Generates synthetic conversation pairs from category templates and appends them.
Run with:
    python scripts/expand_chat_data.py
"""

import json
import random
from pathlib import Path

TARGET_SIZE = 2000
DATA_PATH = Path("data/chat_data.jsonl")

# Category templates: each category contains possible user prompts and assistant responses.
TEMPLATES = {
    "greeting": {
        "user": [
            "Hi!",
            "Hello!",
            "Hey there!",
            "Good day!",
            "Good morning!",
            "Good afternoon!",
            "Good evening!",
            "Howdy!"
        ],
        "assistant": [
            "Hello! How can I assist you today?",
            "Hi there! What can I do for you?",
            "Hey! How can I help?",
            "Greetings! How may I assist?",
            "Good to see you! How can I support you today?",
            "Hi! What's up?"
        ],
    },
    "casual_question": {
        "user": [
            "What's your favorite movie?",
            "Do you like sports?",
            "What's your hobby?",
            "Have you read any good books lately?",
            "Got any plans for the weekend?",
            "What's your favorite song?",
            "Do you prefer coffee or tea?",
            "What's your favorite season?"
        ],
        "assistant": [
            "It's hard to choose but I enjoy science fiction movies. How about you?",
            "I find sports fascinating, especially soccer. Do you have a favorite?",
            "I love learning and chatting with interesting people. What about you?",
            "I've heard good things about 'Dune'. Have you read it?",
            "Weekends are great for relaxing and learning new things. Do you have any plans?",
            "I enjoy melodic tunes across various genres. What's your go-to track lately?",
            "I don't drink beverages, but I find tea culture fascinating. Which do you prefer?",
            "Spring is wonderful with all the blooming flowers. What's your favorite?"
        ],
    },
    "gratitude": {
        "user": [
            "Thanks for your help!",
            "I appreciate it!",
            "Thank you so much!",
            "Thanks a lot!",
            "Many thanks!",
            "Thank you for the information!",
            "Thanks for being there!",
            "I'm grateful for your assistance!"
        ],
        "assistant": [
            "You're welcome! Glad I could help.",
            "No problem at all!",
            "Happy to assist!",
            "Anytime! Let me know if you need more help.",
            "You're very welcome!",
            "My pleasure. Is there anything else I can do?"
        ],
    },
    "compliment": {
        "user": [
            "You're very helpful!",
            "Great job!",
            "I love your responses!",
            "You did amazing!",
            "You're awesome!",
            "Impressive work!"
        ],
        "assistant": [
            "Thank you, that's very kind!",
            "I appreciate your compliment!",
            "Glad to hear that!",
            "Thanks for the positive feedback!",
            "Happy to help!"
        ],
    },
    "farewell": {
        "user": [
            "Bye!",
            "See you later!",
            "Catch you later!",
            "Goodbye!",
            "Talk to you soon!",
            "Take care!"
        ],
        "assistant": [
            "Goodbye! Have a great day!",
            "See you next time!",
            "Take care!",
            "Looking forward to our next chat!",
            "Bye for now!"
        ],
    },
    "empathy": {
        "user": [
            "I'm feeling sad today.",
            "I had a tough day.",
            "I'm stressed about work.",
            "I'm lonely.",
            "I'm feeling anxious.",
            "I'm exhausted."
        ],
        "assistant": [
            "I'm sorry to hear that. I'm here to listen if you'd like to talk more.",
            "That sounds challenging. Would you like to share what's on your mind?",
            "I understand how that can be stressful. How can I support you?",
            "I'm here for you. Sometimes talking about it helps."
        ],
    },
    "introduction": {
        "user": [
            "Tell me about yourself.",
            "Who are you?",
            "What can you do?",
            "Introduce yourself!"
        ],
        "assistant": [
            "I'm an AI assistant here to help with information and conversation.",
            "I'm a conversational AI designed to assist with various questions.",
            "I'm an AI that loves helping people learn and find answers.",
            "I'm here to chat and assist with whatever you need."
        ],
    },
    "advice": {
        "user": [
            "Any tips for staying productive?",
            "How can I manage my time better?",
            "What's the best way to stay healthy?",
            "How do I learn programming?",
            "How should I prepare for an interview?",
            "Any advice on saving money?",
            "How can I improve my writing skills?",
            "What's a good way to reduce stress?"
        ],
        "assistant": [
            "Setting clear goals and breaking them into smaller tasks often helps. Would you like an example?",
            "Time blocking your calendar can make a big difference. Have you tried that?",
            "A balanced diet, regular exercise, and adequate sleep are key. Which area interests you most?",
            "Starting with a language like Python and building small projects is effective. Shall I recommend resources?",
            "Research the company, practice common questions, and prepare your own questions. Need more details?",
            "Automating savings transfers and tracking spending can help. Want to discuss budgeting tools?",
            "Reading regularly and soliciting feedback improves writing. Would you like some exercises?",
            "Deep breathing, meditation, and regular breaks are beneficial. Have you tried any of these?"
        ],
    },
    "celebration": {
        "user": [
            "I just graduated!",
            "I got a new job!",
            "We bought a house!",
            "I passed my exam!",
            "I won a prize!"
        ],
        "assistant": [
            "Congratulations! That's wonderful news.",
            "Amazing! I'm so happy for you.",
            "That's a huge achievement! Well done.",
            "Great job! You must be proud.",
            "Fantastic! How are you planning to celebrate?"
        ],
    },
    "humor": {
        "user": [
            "Tell me a joke!",
            "Make me laugh!",
            "Say something funny.",
            "Do you know any puns?"
        ],
        "assistant": [
            "Why did the scarecrow get a promotion? Because he was outstanding in his field!",
            "I'd tell you a pizza joke, but it's a little cheesy!",
            "Why don't programmers like nature? Too many bugs!",
            "I would tell you a time-travel joke, but you didn't like it."
        ],
    },
    "support": {
        "user": [
            "I'm nervous about my presentation.",
            "I'm worried about my exam.",
            "I'm scared I might fail.",
            "I'm afraid of public speaking."
        ],
        "assistant": [
            "It's normal to feel nervous. Practicing can boost your confidence. Want some tips?",
            "Deep breaths and preparation go a long way. Would you like a study plan?",
            "Remember past successes to build confidence. Shall we outline your strengths?",
            "Many people share that fear. Would practicing with me help?"
        ],
    },
    "opinion": {
        "user": [
            "What do you think about climate change?",
            "What's your opinion on electric cars?",
            "Do you think technology is good or bad?",
            "What's your view on social media?"
        ],
        "assistant": [
            "Climate change is a serious challenge requiring global action. What are your thoughts?",
            "Electric cars are promising for reducing emissions. Do you have one?",
            "Technology itself is neutral; its impact depends on how we use it. How do you see it?",
            "Social media can connect people but also has downsides. What's your experience?"
        ],
    },
    "recommendation": {
        "user": [
            "Recommend a good book.",
            "What movie should I watch?",
            "Any podcast suggestions?",
            "Which TV series is worth binge-watching?"
        ],
        "assistant": [
            "If you like fantasy, 'The Name of the Wind' is great. What genre do you prefer?",
            "'Arrival' is a thought-provoking sci-fi film. Do you enjoy science fiction?",
            "'Radiolab' explores fascinating stories. What topics interest you?",
            "'Stranger Things' is fun if you like mystery and nostalgia. Seen it yet?"
        ],
    },
    "philosophical": {
        "user": [
            "What's the meaning of life?",
            "Do we have free will?",
            "Is reality objective?",
            "What is consciousness?"
        ],
        "assistant": [
            "That's a timeless question with many perspectives. What gives your life meaning?",
            "Philosophers debate this endlessly. What's your take on free will?",
            "Some argue reality is subjective experience. How do you perceive it?",
            "Consciousness remains one of science's great mysteries. What do you think it is?"
        ],
    },
    "motivation": {
        "user": [
            "I need motivation to study.",
            "Help me stay focused on my goals.",
            "I'm losing motivation at work.",
            "How do I stay consistent?"
        ],
        "assistant": [
            "Setting clear, achievable goals can keep you on track. Shall we set one together?",
            "Visualizing your success often boosts motivation. Want to try?",
            "Taking short breaks can renew energy. Would scheduling them help?",
            "Building habits through small daily actions is effective. Shall we plan some?"
        ],
    },
    "educational": {
        "user": [
            "Tell me an interesting fact.",
            "Teach me something new.",
            "What's a fun science fact?",
            "Share a historical tidbit."
        ],
        "assistant": [
            "Honey never spoils; edible honey has been found in ancient tombs!",
            "Octopuses have three hearts and blue blood. Fascinating, right?",
            "Bananas are berries, but strawberries aren't!",
            "Oxford University is older than the Aztec Empire."
        ],
    },
    "interest": {
        "user": [
            "I love photography.",
            "I'm into baking.",
            "I enjoy gardening.",
            "I'm learning guitar."
        ],
        "assistant": [
            "Photography captures moments beautifully. What's your favorite subject?",
            "Baking is both science and art. What's your signature bake?",
            "Gardening is so rewarding. What plants do you grow?",
            "Guitar is a great instrument. What's your favorite song to play?"
        ],
    },
}


def load_existing(path: Path):
    """Load existing examples from the data file."""
    examples = []
    if path.exists():
        with path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        examples.append(json.loads(line))
                    except json.JSONDecodeError:
                        # Skip malformed lines
                        continue
    return examples


def generate_example() -> dict:
    """Generate a single conversation example from templates."""
    category = random.choice(list(TEMPLATES.keys()))
    user_text = random.choice(TEMPLATES[category]["user"])
    assistant_resp = random.choice(TEMPLATES[category]["assistant"])

    # Optional: add minor random variation to seem more unique
    if random.random() < 0.3:
        # Add a polite prefix or suffix
        prefixes = ["Hey, ", "So, ", "Just curious, ", "Quick question: ", "" ]
        suffixes = [" :)", "!", "", " please.", " if that's okay."]
        user_text = random.choice(prefixes) + user_text + random.choice(suffixes)

    return {"text": user_text, "response": assistant_resp, "type": category}


def main():
    existing = load_existing(DATA_PATH)
    current_size = len(existing)
    to_generate = TARGET_SIZE - current_size
    if to_generate <= 0:
        print(f"Data file already has {current_size} entries (>= {TARGET_SIZE}). No action needed.")
        return

    print(f"Generating {to_generate} new examples to reach {TARGET_SIZE} total...")
    random.seed(42)
    new_examples = [generate_example() for _ in range(to_generate)]

    with DATA_PATH.open("a") as f:
        for ex in new_examples:
            json.dump(ex, f, ensure_ascii=False)
            f.write("\n")

    print(f"Done! {to_generate} examples added. Total is now {TARGET_SIZE}.")


if __name__ == "__main__":
    main() 