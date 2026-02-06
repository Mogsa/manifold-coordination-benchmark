"""
Debug script for NOVEL composition (not in few-shot).

Tests: β-γ = cos(square(x)) = cos(x²)
Shows FULL prompts and responses.
"""

import math
from pathlib import Path
import re

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.llm_utils import call_llm

PROMPTS_DIR = Path(__file__).parent / "prompts"


def load_prompt(filename):
    return (PROMPTS_DIR / filename).read_text()


def main():
    print("=" * 80)
    print("TELEPATHIC DEBUG - NOVEL COMPOSITION (cos(x²))")
    print("=" * 80)

    # =========================================================================
    # TEST SETUP - NOVEL COMPOSITION
    # =========================================================================
    # β-γ = cos(square(x)) = cos(x²)
    # This was NOT shown in few-shot examples!
    test_func = lambda x: math.cos(x ** 2)
    sample_x = [-2.0, -1.0, 0.0, 1.0, 2.0]
    samples = [(x, test_func(x)) for x in sample_x]
    test_x = 1.5
    expected_y = test_func(test_x)

    print("\n[TEST SETUP]")
    print(f"Function: cos(x²) = β-γ (NOVEL - not in few-shot!)")
    print(f"Samples:")
    for x, y in samples:
        print(f"  f({x}) = {y:.4f}")
    print(f"Test input: {test_x}")
    print(f"Expected output: {expected_y:.4f}")
    print(f"Expected token: β-γ")

    # =========================================================================
    # SEER CONVERSATION
    # =========================================================================
    print("\n" + "=" * 80)
    print("SEER CONVERSATION")
    print("=" * 80)

    seer_system = load_prompt("seer_system.txt")
    seer_trial_template = load_prompt("seer_trial.txt")

    # Format with actual values
    seer_trial = seer_trial_template
    for i, (x, y) in enumerate(samples):
        seer_trial = seer_trial.replace(f"{{y{i}}}", f"{y:.4f}")

    print("\n--- SEER SYSTEM PROMPT (abbreviated) ---")
    print("... [7 examples teaching α=sin, β=cos, γ=square, δ=abs, ε=neg] ...")
    print("... [2 composition examples: α-γ=sin(x²), δ-α=|sin(x)|] ...")

    print("\n--- SEER USER PROMPT ---")
    print(seer_trial)

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
    match = re.search(r'MESSAGE:\s*([^\n]+)', seer_response, re.IGNORECASE)
    if match:
        seer_message = match.group(1).strip().rstrip('.,;:').strip('"\'')
    else:
        seer_message = "PARSE_FAILED"

    print(f"\n--- EXTRACTED MESSAGE: {seer_message} ---")
    print(f"--- EXPECTED MESSAGE:  β-γ ---")
    print(f"--- MATCH: {'YES' if seer_message == 'β-γ' else 'NO'} ---")

    # =========================================================================
    # DOER CONVERSATION
    # =========================================================================
    print("\n" + "=" * 80)
    print("DOER CONVERSATION")
    print("=" * 80)

    doer_system = load_prompt("doer_system.txt")
    doer_trial_template = load_prompt("doer_trial.txt")

    doer_trial = doer_trial_template
    doer_trial = doer_trial.replace("{message}", seer_message)
    doer_trial = doer_trial.replace("{x_test}", str(test_x))

    print("\n--- DOER SYSTEM PROMPT (abbreviated) ---")
    print("... [7 examples teaching how to decode tokens] ...")

    print("\n--- DOER USER PROMPT ---")
    print(doer_trial)

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
        # Try to find last number
        numbers = re.findall(r'-?\d+\.?\d*', doer_response)
        prediction = float(numbers[-1]) if numbers else None

    print(f"\n--- EXTRACTED PREDICTION: {prediction} ---")

    # =========================================================================
    # ANALYSIS
    # =========================================================================
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    print(f"\n[SEER ANALYSIS]")
    print(f"  Expected encoding: β-γ (cos ∘ square)")
    print(f"  Actual encoding:   {seer_message}")
    if seer_message != "β-γ":
        # Decode what Seer actually sent
        tokens = seer_message.split("-")
        token_map = {"α": "sin", "β": "cos", "γ": "square", "δ": "abs", "ε": "neg"}
        decoded = " ∘ ".join([token_map.get(t, "?") for t in tokens])
        print(f"  Seer's encoding means: {decoded}")
        print(f"  ERROR: Seer misidentified the function!")

    print(f"\n[DOER ANALYSIS]")
    print(f"  Received message: {seer_message}")
    print(f"  Test input: {test_x}")
    print(f"  Doer predicted: {prediction}")
    print(f"  Expected output: {expected_y:.4f}")

    if prediction:
        error = abs(prediction - expected_y)
        print(f"  Absolute error: {error:.4f}")
        success = error < 0.01 * abs(expected_y) if expected_y != 0 else error < 0.01
        print(f"  Success: {'YES' if success else 'NO'}")

    print(f"\n[ROOT CAUSE]")
    if seer_message != "β-γ":
        print("  The Seer failed to identify cos(x²) from the samples.")
        print("  This is a NOVEL composition - it was not in the few-shot examples.")
        print("  The Seer had to GENERALIZE the composition rule to a new case.")
        print("")
        print("  Few-shot showed: α-γ (sin∘square), δ-α (abs∘sin)")
        print("  Novel test:      β-γ (cos∘square)")
        print("")
        print("  This failure suggests the agent may have memorized the examples")
        print("  rather than learning the general composition rule.")


if __name__ == "__main__":
    main()
