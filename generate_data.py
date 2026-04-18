import os
import json
import hashlib
from pathlib import Path
from openai import OpenAI

# Initialize the OpenAI client pointing to NVIDIA's API
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key="NVIDIA_API"
)

# Using the requested model from NVIDIA NIM
MODEL_NAME = "moonshotai/kimi-k2-instruct"

TOOL_SCHEMAS = {
    "weather":  {"tool":"weather",  "args":{"location":"str","unit":"C|F"}},
    "calendar": {"tool":"calendar", "args":{"action":"list|create","date":"YYYY-MM-DD","title":"str?"}},
    "convert":  {"tool":"convert",  "args":{"value":"num","from_unit":"str","to_unit":"str"}},
    "currency": {"tool":"currency", "args":{"amount":"num","from":"ISO3","to":"ISO3"}},
    "sql":      {"tool":"sql",      "args":{"query":"str"}},
}

import time

import time

SYSTEM = """You generate training data for a mobile assistant.
Output ONLY a JSON array containing objects with keys: "user" (the user message) and "response"
(the assistant response). The response must EITHER be a tool call, or plain text for refusals.
Tool call format: {"tool": "...", "args": {...}}
No extra explanation, ONLY output the valid JSON array."""

def call_api_with_retry(prompt, system_prompt, max_tokens=1500):
    max_retries = 6
    base_delay = 2
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.8
            )
            return response.choices[0].message.content
        except Exception as e:
            if "429" in str(e):
                delay = base_delay * (2 ** attempt)
                print(f"Rate limit hit (429). Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print(f"API Error: {e}")
                time.sleep(base_delay)
                
    print("Max retries exceeded for this prompt.")
    return None

def gen_examples_batch(tool_name, count=15):
    import random
    seed = random.randint(1, 100000)
    prompt = f"""Generate exactly {count} HIGHLY unique examples for the '{tool_name}' tool.
Schema: {json.dumps(TOOL_SCHEMAS.get(tool_name, {}))}

IMPORTANT REQUIREMENTS:
1. Make each example HIGHLY unique and completely different from the others.
2. Use random, diverse, and uncommon cities, numbers, dates, currencies, and units. Do not repeat typical examples (e.g., no Tokyo).
3. Mix styles: Some should be natural requests, some paraphrased, some with typos, some with ambiguity.
Random Seed: {seed}

Return ONLY a JSON array. Format:
[
  {{"user": "...", "response": {{"tool": "...", "args": {{...}}}}}}
]"""
    
    content = call_api_with_retry(prompt, SYSTEM, max_tokens=2500)
    if not content:
        return []
        
    try:
        if content.startswith("```json"):
            content = content.split("```json")[1].split("```")[0].strip()
        elif content.startswith("```"):
            content = content.split("```")[1].split("```")[0].strip()
        
        examples = json.loads(content)
        if isinstance(examples, list):
            return examples
        return []
    except Exception as e:
        print(f"Error parsing example JSON: {e}")
        return []

def gen_refusals_batch(count=15):
    import random
    seed = random.randint(1, 100000)
    prompt = f"""Generate exactly {count} HIGHLY unique "refusal" examples.
These are requests where NO tool applies (e.g., chitchat, impossible requests, jokes, philosophy).

IMPORTANT REQUIREMENTS:
1. Make each example completely different.
2. Responses should be plain text explanations or polite refusals.
Random Seed: {seed}

Return ONLY a JSON array. Format:
[
  {{"user": "...", "response": "..."}}
]"""
    
    content = call_api_with_retry(prompt, SYSTEM, max_tokens=2500)
    if not content:
        return []
        
    try:
        if content.startswith("```json"):
            content = content.split("```json")[1].split("```")[0].strip()
        elif content.startswith("```"):
            content = content.split("```")[1].split("```")[0].strip()
        
        examples = json.loads(content)
        if isinstance(examples, list):
            return examples
        return []
    except Exception as e:
        print(f"Error parsing refusal JSON: {e}")
        return []

def gen_multiturn_batch(count=5):
    import random
    seed = random.randint(1, 100000)
    prompt = f"""Generate exactly {count} 2-turn conversations.
For each conversation:
Turn 1: User asks about a tool (e.g., currency conversion).
Turn 2: User says a follow-up that requires resolving context (e.g., "convert that to GBP").
IMPORTANT: Use extremely diverse and random contexts.
Random Seed: {seed}

Return ONLY a JSON array. Format:
[
  {{"turns": [{{"user":"...","response":"..."}}, {{"user":"...","response":"..."}}]}}
]"""
    
    content = call_api_with_retry(prompt, SYSTEM, max_tokens=2500)
    if not content:
        return []

    try:
        if content.startswith("```json"):
            content = content.split("```json")[1].split("```")[0].strip()
        elif content.startswith("```"):
            content = content.split("```")[1].split("```")[0].strip()
            
        conversations = json.loads(content)
        if isinstance(conversations, list):
            return conversations
        return []
    except Exception as e:
        print(f"Error parsing multiturn JSON: {e}")
        return []

SYSTEM_PROMPT = """You are a mobile device assistant. For user requests that match
one of these tools, respond ONLY with a tool call in this format:
{"tool": "TOOL_NAME", "args": {ARGS}}

Available tools:
- weather: {"location": "string", "unit": "C|F"}
- calendar: {"action": "list|create", "date": "YYYY-MM-DD", "title": "string?"}
- convert: {"value": number, "from_unit": "string", "to_unit": "string"}
- currency: {"amount": number, "from": "ISO3", "to": "ISO3"}
- sql: {"query": "string"}

For chitchat, ambiguous requests, or unsupported tasks, respond with plain text (no tool call)."""

def format_for_training(example):
    """Convert to chat format with system prompt."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": example["user"]},
            {"role": "assistant", "content": str(example.get("response", ""))}
        ]
    }

if __name__ == "__main__":
    examples = []
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    raw_file = data_dir.joinpath("raw_examples.jsonl")
    
    print("Generating standard examples in batches...")
    for tool in TOOL_SCHEMAS:
        # Generate 300 examples per tool -> 20 calls of 15 examples each
        print(f"Generating for {tool}...")
        for _ in range(20):
            batch = gen_examples_batch(tool, count=15)
            for ex in batch:
                examples.append(ex)
                with open(raw_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(ex) + "\n")
            time.sleep(1) # minor delay to avoid spiking limit

    print("Generating refusals in batches...")
    # Generate 300 refusals -> 20 calls of 15 examples each
    for _ in range(20):
        batch = gen_refusals_batch(count=15)
        for ex in batch:
            examples.append(ex)
            with open(raw_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(ex) + "\n")
        time.sleep(1)

    print("Generating multi-turn in batches...")
    # Generate 200 turns. A conversation is 2 turns. So 100 conversations.
    # We do 20 calls of 5 conversations each = 100 conversations (200 turns).
    for _ in range(20):
        batch = gen_multiturn_batch(count=5)
        for conv in batch:
            if "turns" in conv:
                for t in conv["turns"]:
                    examples.append(t)
                    with open(raw_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(t) + "\n")
        time.sleep(1)

    print(f"Finished generating {len(examples)} raw examples.")

    # Dedup by SHA-256 of user prompt and save hashes
    seen = set()
    deduped = []
    hashes = []
    
    for ex in examples:
        if "user" not in ex:
            continue
        h = hashlib.sha256(ex["user"].encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            deduped.append(ex)
            hashes.append(h)

    # Convert to chat template
    formatted_data = [format_for_training(ex) for ex in deduped]

    data_dir.joinpath("train.jsonl").write_text(
        "\n".join(json.dumps(e) for e in formatted_data), encoding="utf-8"
    )
    
    data_dir.joinpath("train_hashes.json").write_text(
        json.dumps(hashes, indent=2), encoding="utf-8"
    )

    print(f"Generated {len(deduped)} unique deduplicated examples")
    print("Saved training data to data/train.jsonl")
    print("Saved user prompt SHA-256 hashes to data/train_hashes.json")
