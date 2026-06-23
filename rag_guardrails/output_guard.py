"""
Output guardrail using guardrails-ai.
Checks the generated clinical answer for toxic language and basic quality issues
before it's shown to the user.
"""
from guardrails import Guard
from guardrails.hub import ToxicLanguage

toxicity_guard = Guard().use(
    ToxicLanguage(threshold=0.5, validation_method="sentence", on_fail="exception")
)

HEDGING_PHRASES = [
    "as of my knowledge", "i believe", "i think", "probably",
    "it is likely that", "in my opinion"
]


def check_output(response: str) -> tuple[bool, str]:
    """
    Returns (is_safe, reason).
    is_safe=False means the response should be withheld from the user.
    """
    # 1. Toxicity / unsafe language check via guardrails-ai
    try:
        toxicity_guard.validate(response)
    except Exception:
        return False, "Response flagged for unsafe or toxic language."

    # 2. Hedging language check — clinical answers should be grounded, not speculative
    r_lower = response.lower()
    for phrase in HEDGING_PHRASES:
        if phrase in r_lower:
            return False, f"Response contains ungrounded/speculative language: '{phrase}'"

    # 3. Minimum length sanity check
    if len(response.strip()) < 50:
        return False, "Response too short to be clinically useful."

    return True, ""