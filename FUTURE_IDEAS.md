# Future Ideas & Enhancements

This document tracks potential improvements and features to implement after core functionality is complete.

---

## 1. Comprehensive Debug Visualization Mode

### Motivation
As researchers, we need to deeply understand agent behavior - not just final scores, but the decision-making process at each step.

### Implementation: Debug Mode CLI Flag
```bash
python -m manifold_benchmark.experiments.runner \
    --surface two_peaks_clear \
    --agent-a llm --agent-b llm \
    --debug \
    --output-dir results/debug_run_001
```

### Output Structure
```
results/debug_run_001/
├── metadata.json              # Run config, seeds, timestamps, git hash
├── transcript.json            # Full episode data (already implemented)
├── summary.html              # Interactive HTML report
├── visualizations/
│   ├── 00_initial_surface.png         # Surface with optimal marked
│   ├── turn_01_position.png           # 3D view with current position
│   ├── turn_01_slices.png             # Both agents' slice views
│   ├── turn_01_communication.txt      # Messages for this turn
│   ├── turn_02_position.png
│   ├── turn_02_slices.png
│   └── ...
└── analysis/
    ├── full_trajectory.png            # Complete path on surface
    ├── score_progression.png          # f(x,y) value at each turn
    ├── distance_to_optimal.png        # Distance over time
    ├── exploration_coverage.png       # % of domain observed
    └── communication_log.txt          # All messages, readable format
```

### Features
- **3D position visualization:** Cross marker on manifold showing current position
- **Agent view lines:** Overlay horizontal/vertical slice lines on 3D surface
- **Turn-by-turn slices:** What each agent sees at each step
- **Communication transcript:** Messages and reasoning at each turn
- **Movement arrows:** Show direction of movement between turns

---

## 2. Failure Mode Analysis

### Categories to Track
1. **Local maxima traps:** Agent converged to secondary peak
2. **Coordination failures:** Agents moved in conflicting directions
3. **Communication breakdown:** Misinterpretation of shared information
4. **Exploration inadequacy:** Converged too early without exploring
5. **Boundary issues:** Got stuck at domain edges
6. **Gradient following errors:** Failed to follow gradient correctly

### Implementation
```python
# experiments/failure_analysis.py

def classify_failure_mode(episode_result):
    """
    Analyze episode and classify failure type.

    Returns:
        {
            "failure_type": str,
            "confidence": float,
            "evidence": List[str],
            "critical_turn": int
        }
    """
    pass

def generate_failure_report(episodes):
    """Generate report showing common failure patterns."""
    pass
```

### Visualizations
- Heatmap of failure locations on surface
- Timeline showing when coordination broke down
- Communication quality scores per turn

---

## 3. Communication Quality Metrics

### Metrics to Track
- **Information density:** Are messages informative or vague?
- **Gradient reporting accuracy:** Do agents accurately describe gradients?
- **Peak location hypotheses:** Do agents share position estimates?
- **Agreement tracking:** When do agents agree/disagree?
- **Numeric vs qualitative:** How much quantitative info shared?

### Implementation
```python
# experiments/communication_analysis.py

def analyze_message_quality(message, observation):
    """
    Score message quality based on:
    - Mentions actual gradient value
    - Describes slice shape
    - Proposes specific next position
    - References other agent's information
    """
    pass

def track_agreement(message_a, message_b):
    """Detect if agents agree on strategy."""
    pass
```

---

## 4. Comparative Analysis Tools

### Side-by-Side Episode Comparison
Compare two episodes on same surface:
- Overlay trajectories
- Compare messages at same turns
- Highlight divergence points

### Agent Type Comparison
Visual comparison of Random vs Greedy vs LLM on same surface:
- Grid of 3 trajectories side-by-side
- Score comparison table
- Efficiency metrics (turns to convergence)

### Surface Difficulty Analysis
- Which surfaces cause most failures?
- Correlation between surface features and success rate
- Ranking of surface difficulty

---

## 5. Interactive Jupyter Analysis

### Notebook Template: `notebooks/episode_analysis.ipynb`

Features:
```python
# Load any episode
episode = load_episode("results/run_xyz/transcript.json")

# Interactive 3D plot (rotate, zoom)
plot_interactive_episode(episode)

# Slider widget to step through turns
@interact(turn=IntSlider(min=1, max=10))
def show_turn(turn):
    display_position(turn)
    display_slices(turn)
    display_messages(turn)

# Generate publication figures
fig = create_publication_figure(episode, dpi=300)
```

Benefits:
- Rapid exploration of episodes
- Easy to try different visualization styles
- Can generate figures for dissertation
- Shareable with advisor

---

## 6. HTML Report Generator

### Self-Contained Report
Single HTML file with everything embedded:
- All images inline (base64 encoded)
- Collapsible sections per turn
- Side-by-side agent views
- Syntax highlighted transcript
- Interactive elements (click to expand)

### Use Cases
- Email to advisor for review
- Include in dissertation appendix
- Share with other researchers
- Archive for reproducibility

### Template Structure
```html
<html>
  <head>
    <style>/* Styling */</style>
  </head>
  <body>
    <h1>Episode Report: two_peaks_clear</h1>

    <section class="metadata">
      <h2>Configuration</h2>
      <!-- Surface, agents, seeds, etc. -->
    </section>

    <section class="overview">
      <h2>Overview</h2>
      <img src="data:image/png;base64,...">  <!-- Trajectory -->
      <p>Score: 0.97 | Distance to optimal: 0.15</p>
    </section>

    <section class="turns">
      <h2>Turn-by-Turn Analysis</h2>
      <details>
        <summary>Turn 1</summary>
        <!-- Position, slices, messages -->
      </details>
      <!-- Repeat for each turn -->
    </section>
  </body>
</html>
```

---

## 7. Quantitative Tracking Enhancements

### Per-Turn Metrics
Currently tracked:
- Position (x, y)
- Observations
- Messages
- Decisions

**Add tracking for:**
- `value_at_position`: f(x,y) at current location
- `distance_to_optimal`: Euclidean distance to global max
- `gradient_magnitude`: sqrt(gx^2 + gy^2)
- `exploration_coverage`: Fraction of domain observed so far
- `position_change`: Distance moved this turn
- `tokens_used`: LLM token count (for cost analysis)

### Aggregate Metrics
- **Sample efficiency:** Turns needed to reach threshold
- **Convergence rate:** How quickly value improves
- **Exploration breadth:** Spatial distribution of observations
- **Communication efficiency:** Tokens per score improvement

---

## 8. Advanced Visualization Options

### 3D Enhancements
- **View lines on surface:** Show horizontal/vertical slices as colored planes intersecting surface
- **Gradient field overlay:** Arrow field showing gradient directions
- **Uncertainty zones:** Highlight regions not yet explored
- **Animation:** MP4 of agent moving turn by turn (already in plan as 4.4)

### 2D Alternatives
- **Heatmap overlay:** Show visit frequency on contour plot
- **Split screen:** Left=surface contour, Right=current slices
- **Timeline view:** All turns in a single scrollable image

### Network Diagrams
- **Communication graph:** Who spoke to whom, information flow
- **Decision tree:** Show how messages influenced decisions

---

## 9. Error Handling & Edge Cases

### Scenarios to Handle
1. **Invalid position returned:** Agent suggests x=-5 or x=15
   - Current: Probably crashes
   - Better: Clamp to bounds, log warning, continue

2. **LLM API failure:** Timeout, rate limit, authentication error
   - Current: Exception stops episode
   - Better: Retry with exponential backoff, fallback to last position

3. **Unparseable response:** LLM doesn't provide coordinate
   - Current: Parsing error
   - Better: Re-prompt with clarification, max 3 retries

4. **Contradictory information:** Agent A says "peak at x=3", Agent B says "no peak visible"
   - Current: Not detected
   - Better: Flag as potential coordination failure

5. **Timeout:** LLM takes >60 seconds
   - Current: Hangs
   - Better: Timeout and skip turn or use default move

### Implementation
```python
# experiments/error_handling.py

class EpisodeError(Exception):
    """Base class for episode errors."""
    pass

class InvalidPositionError(EpisodeError):
    """Agent returned out-of-bounds position."""
    pass

class LLMAPIError(EpisodeError):
    """API call failed."""
    pass

def handle_position_error(position, bounds):
    """Clamp position to valid bounds and log warning."""
    pass

def handle_api_error(error, retry_count):
    """Retry with exponential backoff."""
    pass
```

---

## 10. Reproducibility Enhancements

### Full Provenance Tracking
Currently logged:
- Episode transcript

**Add:**
- Python version
- Package versions (pip freeze)
- Git commit hash
- Exact timestamp (with timezone)
- Machine hostname
- Random seeds (global, per-agent)
- LLM API parameters (temperature, max_tokens, model version)
- API response metadata (tokens used, finish reason)

### Reproducibility Verification
```python
# experiments/verify_reproducibility.py

def verify_run_reproducible(episode_id):
    """
    Re-run episode with same config and verify identical results.

    Checks:
    - Same final position
    - Same messages (for deterministic LLMs)
    - Same score
    """
    pass
```

---

## 11. Batch Analysis Tools

### Multi-Episode Comparison
- **Success rate heatmap:** Grid of surfaces × agent types
- **Box plots:** Score distribution per condition
- **Trajectory overlays:** All runs on same surface overlaid

### Statistical Reporting
- Effect size calculations (Cohen's d)
- Confidence intervals
- Power analysis (was sample size adequate?)

---

## 12. Integration Ideas

### Real-Time Monitoring Dashboard
During long batch runs:
- Web dashboard showing progress
- Live updating plots
- Current episode visualization
- ETA calculation

### Integration with Weights & Biases
- Log all runs to W&B
- Automatic hyperparameter tracking
- Compare across experiment runs
- Share results with team

---

## Implementation Priority

**Phase A: Essential (After Phase 5 complete)**
1. Debug visualization mode (item 1)
2. Basic failure classification (item 2)
3. HTML report generator (item 6)

**Phase B: Analysis Tools**
4. Communication quality metrics (item 3)
5. Jupyter analysis notebook (item 5)
6. Enhanced quantitative tracking (item 7)

**Phase C: Polish**
7. Advanced visualizations (item 8)
8. Robust error handling (item 9)
9. Full reproducibility tracking (item 10)

**Phase D: Optional**
10. Comparative analysis tools (item 4)
11. Batch analysis enhancements (item 11)
12. Real-time monitoring (item 12)

---

## Notes

- These are enhancements to implement **after** core benchmark is working
- Prioritize based on dissertation timeline and research needs
- Some features may not be necessary depending on results
- Keep this document updated as new ideas emerge

---

*Last updated: 2026-01-13*
