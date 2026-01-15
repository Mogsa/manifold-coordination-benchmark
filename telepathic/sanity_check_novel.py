"""
Sanity check for NOVEL composition (not in few-shot examples).

Tests: β-γ = cos(square(x)) = cos(x²)
This composition was NOT shown in the few-shot examples.
"""

import math
from telepathic.agents import (
    LLMSeerAgent,
    LLMDoerAgent,
    SeerInput,
    DoerInput,
)
from telepathic.core.protocol import MVP_VOCABULARY
from telepathic.core.evaluation import score_prediction


def run_novel_test():
    """Test a novel composition."""

    print("=" * 60)
    print("TELEPATHIC - NOVEL COMPOSITION TEST")
    print("=" * 60)

    # Test function: β-γ = cos(square(x)) = cos(x²)
    # This is NOT in few-shot examples
    test_func = lambda x: math.cos(x ** 2)
    func_name = "cos(x²)"
    expected_token = "β-γ"

    # Generate samples
    sample_x = [-2.0, -1.0, 0.0, 1.0, 2.0]
    samples = [(x, test_func(x)) for x in sample_x]

    # Test point
    test_x = 1.5
    expected_y = test_func(test_x)  # cos(2.25) ≈ -0.6282

    print(f"\n[TEST FUNCTION] {func_name}")
    print(f"[SAMPLES]")
    for x, y in samples:
        print(f"  f({x}) = {y:.4f}")
    print(f"\n[TEST INPUT] x = {test_x}")
    print(f"[EXPECTED OUTPUT] y = {expected_y:.4f}")
    print(f"[EXPECTED TOKEN] {expected_token}")

    # Create agents
    print("\n" + "-" * 60)
    seer = LLMSeerAgent(model="gemini/gemini-2.5-flash", temperature=0.3, verbose=True)
    doer = LLMDoerAgent(model="gemini/gemini-2.5-flash", temperature=0.3, verbose=True)

    # SEER PHASE
    print("[SEER PHASE]")
    seer_input = SeerInput(samples=samples, vocabulary=MVP_VOCABULARY, max_tokens=5)
    seer_output = seer.generate_message(seer_input)
    print(f"  Message: {seer_output.message}")

    # DOER PHASE
    print("\n[DOER PHASE]")
    doer_input = DoerInput(message=seer_output.message, test_input=test_x, vocabulary=MVP_VOCABULARY)
    doer_output = doer.generate_prediction(doer_input)
    print(f"  Prediction: {doer_output.prediction:.4f}")

    # EVALUATION
    print("\n" + "-" * 60)
    trial_score = score_prediction(doer_output.prediction, expected_y)

    print(f"[RESULTS]")
    print(f"  Expected token: {expected_token}")
    print(f"  Actual token:   {seer_output.message}")
    print(f"  Token match:    {'YES' if seer_output.message == expected_token else 'NO'}")
    print(f"  Expected y:     {expected_y:.4f}")
    print(f"  Predicted y:    {doer_output.prediction:.4f}")
    print(f"  Abs error:      {trial_score.absolute_error:.4f}")
    print(f"  Success:        {'YES' if trial_score.success else 'NO'}")

    return trial_score.success


if __name__ == "__main__":
    success = run_novel_test()
    exit(0 if success else 1)
