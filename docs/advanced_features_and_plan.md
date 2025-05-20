# Autonomous Features Investigation: Findings Summary

This document summarizes the findings from investigating the relationship between UI controls and the backend implementation of advanced natural learning and prediction features in the BTC-AI project.

## 1. UI Controls (`src/ui/tabs.py`)

*   The UI provides numerous controls (checkboxes, inputs, sliders) allowing users to toggle features on/off and configure specific parameters.
*   These include settings for:
    *   Probabilistic Predictions (Enable, Confidence Threshold, Uncertainty Threshold)
    *   Adaptive Exploration (Enable, Initial Rate, Min Rate, Decay Rate)
    *   Experience Prioritization (Enable, Alpha, Beta)
    *   Meta-Learning (Enable, Interval, Rate)
    *   Contextual Memory (Enable)
    *   Post-Trade Analysis (Enable)
    *   Dynamic Horizons (Enable, Adaptation Rate, Min/Max Horizon, Distribution Mode)
*   This extensive user control potentially conflicts with the design goal of having these features operate autonomously.

## 2. Agent Implementation (`src/agent/agent.py`)

*   The `PPOAgent` class is the central hub for these features.
*   **Initialization:** The agent reads configuration flags (`use_...`) and initial parameters (e.g., `INITIAL_EXPLORATION`, `MIN_EXPLORATION`, `REPLAY_ALPHA`, `REPLAY_BETA`) during its `__init__` method. This aligns with using UI settings as initial baselines.
*   **Existing Dynamic Logic:** Implementations for several autonomous behaviors already exist, gated by the `use_...` flags read at initialization:
    *   **Adaptive Exploration:** `_get_exploration_factor()` calculates exploration dynamically based on rewards, confidence, and market regime, overriding the initial rate during action selection.
    *   **Experience Prioritization (PER):** Uses `PrioritizedReplayBuffer` (if enabled), which prioritizes samples based on surprise/error. Beta parameter likely adapts via internal scheduling.
    *   **Meta-Learning:** Uses `HyperparamOptimizer` (if enabled) to suggest hyperparameter adjustments based on performance. `adapt_hyperparameters` applies these suggestions. **Finding:** This method currently modifies the shared `self.config` dictionary at runtime.
    *   **Contextual Memory & Post-Trade Analysis:** Logic exists and appears to run based solely on the `use_...` flags set during initialization.
*   **Archaic Code:** Contains `TeacherStudentDistillation` class/references, identified as outdated.

## 3. Model Implementation (`src/models/models.py`)

*   The `ActorCritic` model initializes prediction horizons based on `horizon_config` passed during creation.
*   It reads a `USE_DYNAMIC_HORIZONS` flag (defaulting True) but lacks internal logic for *adapting* horizons based on this flag.
*   It includes heads for probabilistic prediction (mean, std dev, confidence), making these outputs available to the agent.

## 4. Confidence Thresholds (Targeted Search Result)

*   No code was found in `agent.py`, `environment`, or `training` that uses the static config keys `PROBABILISTIC_CONFIDENCE_THRESHOLD` or `PROBABILISTIC_UNCERTAINTY_THRESHOLD`.
*   **Conclusion:** These specific UI settings are currently unused. The agent uses confidence dynamically (via `self.confidence_history`) to adjust exploration.

## 5. Dynamic Horizon Adaptation (Targeted Search Result)

*   No code was found referencing `adapt_prediction_horizons` (from docs) or using config keys like `HORIZON_ADAPTATION_RATE`.
*   The `USE_DYNAMIC_HORIZONS` flag read by the model isn't connected to any apparent adaptation mechanism.
*   **Conclusion:** The dynamic *adaptation* logic for horizons seems unimplemented or disconnected. Related UI settings are unused. Horizons remain static based on initialization.

## 6. Context (`v2` Files)

*   User confirmed `agent_v2.py` / `utils_v2.py` references in documentation map to current `src/agent/agent.py` / `src/utils/utils.py`, validating the findings.

## 7. Overall Conclusions

*   Many natural learning features (Adaptive Exploration, PER, Meta-Learning, Contextual Memory, Post-Trade Analysis) are partially aligned with the "UI Baseline -> Dynamic" approach, as they initialize from config flags/params but have internal dynamic logic.
*   The static UI settings for Confidence Thresholds and Dynamic Horizon *Adaptation* are currently **unused** by the backend.
*   The main conflict with the desired autonomous behavior is the **Meta-Learning component modifying the shared config dictionary at runtime**.
*   Archaic code (`TeacherStudentDistillation`) exists and needs removal.




# Action Plan for Autonomous Features (Option 2: UI Baseline)

This document outlines the specific steps to modify the backend code so that implemented advanced features operate autonomously after being initialized by user settings in the UI. This follows Option 2 discussed previously.

**Goal:** Ensure features like Adaptive Exploration, PER, Meta-Learning, Contextual Memory, and Post-Trade Analysis run dynamically based on internal logic post-initialization, while using UI settings only as the starting point.

**Target File:** `src/agent/agent.py`

**Plan:**

1.  **Modify Meta-Learning Behavior (`adapt_hyperparameters` method in `PPOAgent`):**
    *   **Objective:** Prevent runtime modification of the shared `self.config` dictionary. Update agent attributes or optimizer parameters directly instead.
    *   **Action:**
        *   Locate lines within `adapt_hyperparameters` that assign values to `self.config[...]` (e.g., `self.config[param] = value`).
        *   For `LEARNING_RATE`: Remove the line `self.config["LEARNING_RATE"] = new_lr`. The existing code already updates the optimizer (`param_group['lr'] = new_lr`), which is sufficient.
        *   For other tunable parameters (e.g., `ENTROPY_COEF`, `EPS_CLIP`, `GAMMA`, `INITIAL_EXPLORATION_RATE`, etc. if tuned by `HyperparamOptimizer`):
            *   Identify the corresponding agent attribute that *uses* this value in calculations (e.g., `self.entropy_coef` used in the loss calculation in the `update` method).
            *   Replace `self.config[param] = value` with a direct update to that attribute, for example: `self.entropy_coef = value`.
            *   Verify that the attribute name used (e.g., `self.entropy_coef`) matches the one used elsewhere in the agent's logic.

2.  **Verify Dynamic Exploration (`_get_exploration_factor` method in `PPOAgent`):**
    *   **Objective:** Confirm that exploration used in action selection is dynamically calculated post-initialization.
    *   **Action:** Briefly review the `_get_exploration_factor` method. Confirm that inputs like `self.recent_rewards` and `self.confidence_history` drive the main calculation. Ensure config-derived values like `self.min_exploration` are only used as boundary checks (e.g., in the `max(self.min_exploration, ...)` call). *(Expected: No code changes needed here based on prior review).*

3.  **Remove Archaic Code (`TeacherStudentDistillation` in `src/agent/agent.py`):**
    *   **Objective:** Remove unused, outdated distillation logic.
    *   **Action:**
        *   Search for the class definition: `class TeacherStudentDistillation:`. Delete the entire class block.
        *   Search for any instances where this class might be instantiated (e.g., `self.distillation = TeacherStudentDistillation(...)`). Delete these lines.
        *   Search for any method calls on such an object (e.g., `self.distill_knowledge(...)`). Delete these lines.

**Testing:**

*   After implementing the changes, run training sessions.
*   Monitor logs for errors related to the removed code or modified meta-learning updates.
*   Observe agent behavior (e.g., exploration rates, potentially logs from meta-learning showing adjusted parameters like learning rate) to confirm dynamic operation continues as expected.

**Outcome:**

The UI settings will provide initial values and enable/disable features. After training starts, the agent's implemented dynamic logic (adaptive exploration, PER, meta-learning, etc.) will operate based on performance and internal state, correctly overriding the initial settings where applicable, without interference from runtime config modifications by the meta-learner. Features whose dynamic adaptation logic wasn't found (dynamic horizons) or whose UI controls were unused (confidence thresholds) will remain unaffected by these changes.