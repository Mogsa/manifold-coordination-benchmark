"""
Quick sanity check for Telepathic benchmark agents.

Tests a single Seer→Doer trial to verify LLM integration works.
"""

import math
from telepathic.agents import (
    LLMSeerAgent,
    LLMDoerAgent,
    SeerInput,
    DoerInput,
)
from telepathic.core.protocol import MVP_VOCABULARY
from telepathic.core.evaluation import Evaluator, score_prediction


def run_sanity_check():
    """Run a single trial with LLM agents."""

    print("=" * 60)
    print("TELEPATHIC BENCHMARK - SANITY CHECK")
    print("=" * 60)

    # Test function: γ = square (x²)
    # This should be easy - it's one of the few-shot examples
    test_func = lambda x: x ** 2
    func_name = "square (γ)"
    expected_token = "γ"

    # Generate samples (what Seer sees)
    sample_x = [-2.0, -1.0, 0.0, 1.0, 2.0]
    samples = [(x, test_func(x)) for x in sample_x]

    # Test point (what Doer must compute)
    test_x = 1.5
    expected_y = test_func(test_x)  # 2.25

    print(f"\n[TEST FUNCTION] {func_name}")
    print(f"[SAMPLES] {samples}")
    print(f"[TEST INPUT] x = {test_x}")
    print(f"[EXPECTED OUTPUT] y = {expected_y}")
    print(f"[EXPECTED TOKEN] {expected_token}")

    # Create agents
    print("\n" + "-" * 60)
    print("[CREATING AGENTS]")
    seer = LLMSeerAgent(
        model="gemini/gemini-2.5-flash",
        temperature=0.3,  # Lower temp for more deterministic
        verbose=True
    )
    doer = LLMDoerAgent(
        model="gemini/gemini-2.5-flash",
        temperature=0.3,
        verbose=True
    )

    # SEER PHASE
    print("\n" + "-" * 60)
    print("[SEER PHASE] Encoding function...")

    seer_input = SeerInput(
        samples=samples,
        vocabulary=MVP_VOCABULARY,
        max_tokens=5
    )

    seer_output = seer.generate_message(seer_input)

    print(f"\n[SEER RESULT]")
    print(f"  Message: {seer_output.message}")
    print(f"  Parse attempts: {seer_output.parse_attempts}")
    print(f"  Parse error: {seer_output.parse_error}")

    # DOER PHASE
    print("\n" + "-" * 60)
    print(f"[DOER PHASE] Decoding message '{seer_output.message}' for x={test_x}...")

    doer_input = DoerInput(
        message=seer_output.message,
        test_input=test_x,
        vocabulary=MVP_VOCABULARY
    )

    doer_output = doer.generate_prediction(doer_input)

    print(f"\n[DOER RESULT]")
    print(f"  Prediction: {doer_output.prediction}")
    print(f"  Parse attempts: {doer_output.parse_attempts}")
    print(f"  Parse error: {doer_output.parse_error}")

    # EVALUATION
    print("\n" + "-" * 60)
    print("[EVALUATION]")

    trial_score = score_prediction(doer_output.prediction, expected_y)
    error = trial_score.absolute_error
    rel_error = trial_score.relative_error if trial_score.relative_error else 0
    score = 1 if trial_score.success else 0

    print(f"  Expected: {expected_y}")
    print(f"  Predicted: {doer_output.prediction}")
    print(f"  Absolute error: {error:.4f}")
    print(f"  Relative error: {rel_error:.2%}")
    print(f"  Score: {score} ({'SUCCESS' if score == 1 else 'FAILURE'})")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Seer encoded '{func_name}' as: {seer_output.message}")
    print(f"  Expected token: {expected_token}")
    print(f"  Token match: {'YES' if seer_output.message == expected_token else 'NO'}")
    print(f"  Doer computed: {doer_output.prediction}")
    print(f"  Expected value: {expected_y}")
    print(f"  Trial success: {'YES' if score == 1 else 'NO'}")

    return score == 1


if __name__ == "__main__":
    success = run_sanity_check()
    exit(0 if success else 1)
