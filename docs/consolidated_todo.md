# BTC-AI Consolidated To-Do List & Refinement Plan

## Phase 1: Documentation & Cleanup

*   [ ] **Consolidate Documentation:** Merge relevant tasks and plans from scattered `docs/*.md` files into this single `consolidated_todo.md`.
    *   Files Reviewed:
        *   `main_modularization.md`
        *   `cleanup_progress.md`
        *   `setup_wizard_test_issues.md`
        *   `ui_ux_plan.md`
        *   `system_refinement_and_future_enhancements_plan.md`
        *   `future_roadmap.md`
        *   `optimization.md`
        *   `features.md`
*   [ ] **Review Remaining Docs:** Briefly check other documents in `docs/` (like `architecture.md`, `config_*.md`, `natural_learning.md`, etc.) for any missed actionable items.
*   [ ] **Cleanup Old References:** Perform a thorough search for legacy reference pointers (e.g., old module names, old config systems) in:
    *   Old directory structures (if any)
    *   Test scripts (`tests/`) and batch files (`*.bat`)
    *   Installer scripts (`install_windows.bat`)
    *   Remaining documentation (`docs/`)
    *   *(Consider implementing automated reference checking)*
*   [ ] **Address Test Failures (from `cleanup_progress.md` & `setup_wizard_test_issues.md`):**
    *   [ ] Fix remaining PyTorch-related test failures (investigate version compatibility in test env). Note: Likely unrelated to core logic changes, potentially environment setup.
    *   [ ] (Verify) Ensure `setup_wizard` tests are stable in the *full test suite* (address potential mock interference/global state issues if they reappear).

## Phase 2: Script Refactoring (Targeting ~1000+ line scripts)

*   [ ] **Identify Large Scripts:** Systematically identify Python scripts in `src/` (and potentially `dist/BTC-AI/_internal/src/`) that are approximately 1000 lines or longer and could benefit from modularization.
    *   *(Initial candidates might include `src/agent/agent.py`, `src/utils/dataframe.py`, `src/training/training.py` - need verification)*
*   [ ] **Refactor Script 1 (TBD):**
    *   [ ] Analyze script for logical components.
    *   [ ] Extract components into separate, well-defined modules (following SRP).
    *   [ ] Update imports and ensure original functionality is preserved.
    *   [ ] Add/update tests for new modules.
*   [ ] **Refactor Script 2 (TBD):** (Repeat steps above)
*   [ ] **Refactor Script 3 (TBD):** (Repeat steps above)
*   [ ] **... (Add more as needed)**

## Phase 3: UI/UX & Feature Refinement (from `ui_ux_plan.md` & `system_refinement...`)

*   [ ] **Verify UI Controls for Advanced Features:**
    *   [ ] **Natural Learning:** Verify UI controls correctly affect the underlying natural learning parameters/behavior.
    *   [ ] **Probabilistic Predictions:** Check UI controls for prediction settings influence the model.
    *   [ ] **Adaptive Exploration:** Verify UI controls for exploration parameters are linked correctly.
*   [ ] **Feature Importance Display:**
    *   [ ] Implement a visual representation (chart, table) in the UI to show feature importance results. (Priority: Low)
*   [ ] **Capital Allocation Investigation & Refinement:**
    *   [ ] **Review:** Analyze `docs/ModelOrchestrationDesign.md` (`CapitalAllocationManager`), code usage (orchestrator scripts, `test_progressive_pipeline.py`), `trade_config.py` (`INITIAL_CAPITAL`), `env_base.py` (`profit_reserve_ratio`, `capital`/`usdt_balance`).
    *   [ ] **Analyze:** Determine if dynamic capital allocation is implemented. How is capital currently managed between buckets/agents? Clarify "earmarking" implementation.
    *   [ ] **Define/Implement/Refine:** Based on analysis, define/implement/refine the capital allocation strategy (e.g., implement `CapitalAllocationManager` if needed, add logging). Document the chosen strategy.
    *   [ ] **Test:** Create specific integration tests for capital allocation logic.
*   [ ] **Address Other Incomplete UI Functions:**
    *   [ ] Systematically review UI code (`src/ui/`), TODO comments (`grep -r "TODO"`), and known issues for any other non-functional elements or placeholders.
    *   [ ] Implement fixes incrementally, prioritizing by impact.
    *   [ ] Test fixes manually and potentially with integration tests.

## Phase 4: Future Considerations (Longer Term - from `future_roadmap.md`, `system_refinement...`)

*   [ ] **Earmarking System (Phase 1 from roadmap):** Implement Emergency, Timed, Non-Timed Earmarking and Simulation.
*   [ ] **Deposit & Asset Management (Phase 2 from roadmap):** BTC-USDT detection, multi-asset tracking, deposit auto-detection, liquidity optimization.
*   [ ] **Advanced Pattern Recognition (Phase 3 from roadmap):** TDA, Fractal enhancement, GNNs.
*   [ ] **Decision Intelligence Upgrades (Phase 4 from roadmap):** Causal Inference, Hierarchical RL, Bayesian Deep Learning.
*   [ ] **Computational Efficiency (Phase 5 from roadmap):** Contrastive Learning, Neuromorphic, Quantum-Inspired.
*   [ ] **Investigate Additional TA-Lib Indicators:** Review and potentially integrate indicators like ADX, ATR, MFI, OBV, SAR, STOCH variations etc., based on analysis of their potential value.

## Performance Optimization Recommendations (from `optimization.md`)

*   [ ] **Apply Profiling:** Profile key components like `train_bucket`, `transfer_knowledge`, dashboard updates.
*   [ ] **Optimize Visualizations:** Use `optimize_visualization`, reduce point density, ensure `plt.close()`.
*   [ ] **Add Memory Safeguards:** Check memory before reports, use caching, clear viz memory.
*   [ ] **Configure for Hardware:** Set memory thresholds, adjust batch sizes, limit monitoring overhead appropriately.

---
*This list is generated based on reading multiple documentation files. Prioritization within phases may need adjustment.* 