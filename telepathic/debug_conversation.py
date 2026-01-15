"""
Debug script showing FULL conversation with LLM.

Shows exactly:
1. System prompt sent to Seer
2. User prompt sent to Seer
3. Full response from Seer
4. System prompt sent to Doer
5. User prompt sent to Doer
6. Full response from Doer
"""

import math
from pathlib import Path

# Import LLM utilities directly
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.llm_utils import call_llm

# Load prompts
PROMPTS_DIR = Path(__file__).parent / "prompts"


def load_prompt(filename):
    return (PROMPTS_DIR / filename).read_text()


def main():
    print("=" * 80)
    print("TELEPATHIC DEBUG - FULL CONVERSATION TRACE")
    print("=" * 80)

    # =========================================================================
    # TEST SETUP
    # =========================================================================
    # Testing: γ = square (simple primitive)
    test_func = lambda x: x ** 2
    sample_x = [-2.0, -1.0, 0.0, 1.0, 2.0]
    samples = [(x, test_func(x)) for x in sample_x]
    test_x = 1.5
    expected_y = test_func(test_x)

    print("\n[TEST SETUP]")
    print(f"Function: square (x²)")
    print(f"Samples: {samples}")
    print(f"Test input: {test_x}")
    print(f"Expected output: {expected_y}")

    # =========================================================================
    # SEER CONVERSATION
    # =========================================================================
    print("\n" + "=" * 80)
    print("SEER CONVERSATION")
    print("=" * 80)

    # Load Seer prompts
    seer_system = load_prompt("seer_system.txt")
    seer_trial_template = load_prompt("seer_trial.txt")

    # Format trial prompt with actual values
    seer_trial = seer_trial_template
    for i, (x, y) in enumerate(samples):
        seer_trial = seer_trial.replace(f"{{y{i}}}", f"{y:.4f}")

    print("\n--- SEER SYSTEM PROMPT ---")
    print(seer_system)
    print("\n--- SEER USER PROMPT ---")
    print(seer_trial)

    # Call LLM
    print("\n--- CALLING GEMINI FOR SEER ---")
    seer_messages = [
        {"role": "system", "content": seer_system},
        {"role": "user", "content": seer_trial}
    ]

    seer_response = call_llm(
        model="gemini/gemini-2.5-flash",
        messages=seer_messages,
        temperature=0.3,
        max_tokens=500
    )

    print("\n--- SEER FULL RESPONSE ---")
    print(seer_response)

    # Extract message
    import re
    match = re.search(r'MESSAGE:\s*([^\n]+)', seer_response, re.IGNORECASE)
    if match:
        seer_message = match.group(1).strip().rstrip('.,;:').strip('"\'')
    else:
        seer_message = "PARSE_FAILED"

    print(f"\n--- EXTRACTED MESSAGE: {seer_message} ---")

    # =========================================================================
    # DOER CONVERSATION
    # =========================================================================
    print("\n" + "=" * 80)
    print("DOER CONVERSATION")
    print("=" * 80)

    # Load Doer prompts
    doer_system = load_prompt("doer_system.txt")
    doer_trial_template = load_prompt("doer_trial.txt")

    # Format trial prompt
    doer_trial = doer_trial_template
    doer_trial = doer_trial.replace("{message}", seer_message)
    doer_trial = doer_trial.replace("{x_test}", str(test_x))

    print("\n--- DOER SYSTEM PROMPT ---")
    print(doer_system)
    print("\n--- DOER USER PROMPT ---")
    print(doer_trial)

    # Call LLM
    print("\n--- CALLING GEMINI FOR DOER ---")
    doer_messages = [
        {"role": "system", "content": doer_system},
        {"role": "user", "content": doer_trial}
    ]

    doer_response = call_llm(
        model="gemini/gemini-2.5-flash",
        messages=doer_messages,
        temperature=0.3,
        max_tokens=500
    )

    print("\n--- DOER FULL RESPONSE ---")
    print(doer_response)

    # Extract prediction
    match = re.search(r'PREDICTION:\s*([^\n]+)', doer_response, re.IGNORECASE)
    if match:
        try:
            prediction = float(match.group(1).strip().split()[0].rstrip('.,;:'))
        except:
            prediction = None
    else:
        prediction = None

    print(f"\n--- EXTRACTED PREDICTION: {prediction} ---")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Seer saw samples of x² and encoded as: {seer_message}")
    print(f"Doer received '{seer_message}' and x={test_x}, predicted: {prediction}")
    print(f"Expected output: {expected_y}")
    if prediction:
        print(f"Error: {abs(prediction - expected_y):.4f}")
        print(f"Success: {'YES' if abs(prediction - expected_y) < 0.01 * expected_y else 'NO'}")


if __name__ == "__main__":
    main()
