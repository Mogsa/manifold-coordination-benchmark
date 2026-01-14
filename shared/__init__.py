"""
Shared utilities for the Manifold Benchmark suite.

This module contains common utilities used by both the coordination
and temporal benchmarks.
"""

from shared.gaussians import (
    gaussian_peak_1d,
    gaussian_peak_2d,
    gaussian_peak_from_dict_1d,
    gaussian_peak_from_dict_2d,
)

from shared.llm_utils import (
    call_llm,
    parse_coordinate,
    parse_position_list,
    set_api_key_for_model,
)

from shared.logging import (
    generate_result_id,
    save_json_result,
    load_json_result,
    load_all_json_results,
    list_json_results,
    BaseResultLogger,
)

from shared.base_agent import (
    AgentBase,
    ObservationHistoryMixin,
    MessageHistoryMixin,
)
