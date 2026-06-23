"""
One-off utility: lists currently available free models from OpenRouter's
live catalog, filtering out (likely) reasoning-heavy models so we get
candidates that reliably populate `content` instead of burning tokens
on hidden reasoning traces.

Run after: curl.exe https://openrouter.ai/api/v1/models -o models.json
"""
import json

with open("models.json", "r", encoding="utf-8") as f:
    data = json.load(f)

free_models = [m for m in data["data"] if m["id"].endswith(":free")]

# Heuristic: skip models whose name/id suggests a reasoning architecture,
# since those tend to spend max_tokens on hidden reasoning traces and
# return empty `content` for short prompts (what we hit with deepseek-r1
# and the Poolside model the auto-router picked).
reasoning_markers = ["r1", "reasoning", "think", "o1", "o3", "qwq"]

print(f"Total free models: {len(free_models)}\n")
print("Likely safe (non-reasoning) candidates:")
for m in free_models:
    mid_lower = m["id"].lower()
    if not any(marker in mid_lower for marker in reasoning_markers):
        ctx = m.get("context_length", "?")
        print(f"  {m['id']}  (context: {ctx})")