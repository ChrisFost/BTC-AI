# Plan: System Refinement and Future Enhancements

**Phase 1: Investigate and Refine Agent Coordination & Capital Allocation**

1.  **Review Design and Implementation:**
    *   Read `docs/ModelOrchestrationDesign.md` carefully, focusing on the `CapitalAllocationManager` class design.
    *   Search the codebase (especially in orchestrator scripts like `tests/end_to_end/test_progressive_pipeline.py` or potentially a main application entry point if one exists for multi-agent runs) for any usage or instantiation of `CapitalAllocationManager` or similar logic.
    *   Analyze `src/utils/trade_config.py` to confirm how `INITIAL_CAPITAL` is used per bucket.
    *   Analyze `src/environment/env_base.py` regarding `profit_reserve_ratio` and how `capital` / `usdt_balance` are managed within a single environment.
    *   **Analysis:** Determine if the dynamic capital allocation is actually implemented. If so, how does it work? If not, how is capital currently managed when multiple buckets/agents might be running (e.g., in progressive training)? Is there a shared pool, or are they completely separate based on `INITIAL_CAPITAL` in their config? Clarify the exact meaning and implementation of "earmarking" - is it just the withdrawal reserve, or something else?

2.  **Define Requirements (If Necessary):**
    *   Based on the analysis, if the current system doesn't dynamically allocate capital between running agents, clearly define the desired behavior. How *should* agents share or compete for capital? Should performance influence allocation?

3.  **Implement/Refine (If Necessary):**
    *   If the `CapitalAllocationManager` needs to be implemented or integrated, create a plan for doing so incrementally.
    *   If the existing system (e.g., separate capital pools per bucket) is sufficient but needs clarification or minor tweaks, implement those changes.
    *   Add logging to clearly track capital allocation decisions or balances per agent/bucket.
    *   **Justification:** Document the current state and any changes made to align with the desired capital management strategy.

4.  **Test:**
    *   Create new integration tests specifically for the capital allocation logic. These might involve running simplified versions of multiple agents/buckets concurrently (potentially mocked) and asserting that capital is allocated or managed as expected according to the rules (e.g., equal split, performance-based, etc.).
    *   Verify existing tests still pass.

**Phase 2: Address Incomplete Functions & UI Improvements**

1.  **Identify Incomplete Items:**
    *   Review UI code (`src/ui/`) for any obvious placeholders, `pass` statements in event handlers, buttons that don't do anything, or features mentioned in comments/docs that aren't functional.
    *   Check for TODO comments in the codebase (`grep -r "TODO"`).
    *   Recall any specific UI elements or functions known to be incomplete from previous development.
    *   **Analysis:** Compile a list of specific UI elements or functions needing implementation or fixing.

2.  **Implement Fixes (Incrementally):**
    *   Tackle the identified items one by one or in small related groups.
    *   Prioritize fixes based on impact or user workflow.
    *   Ensure new implementations follow UI patterns and use existing managers (`AppState`, `TrainingManager`, etc.) correctly.
    *   **Justification:** Document the functionality added or fixed for each item.

3.  **Test:**
    *   Manually test the updated UI elements.
    *   If feasible, add new integration tests (similar to Phase 2) for the fixed UI logic, especially if it involves state changes or interactions between managers.
    *   Verify existing tests still pass.

**Phase 3: Future Considerations / Enhancements**

*   **Investigate Additional TA-Lib Indicators:**
    *   Review the comprehensive list of indicators provided by TA-Lib (e.g., ADX, ATR, MACD variations, MFI, OBV, SAR, STOCH variations, etc.).
    *   Analyze which indicators might provide orthogonal or complementary information to the currently used features for each bucket (Scalping, Short, Medium, Long).
    *   Consider adding promising indicators to the feature set and evaluate their impact on model performance through experimentation and backtesting.

**Future Considerations:**

*   Review the detailed ideas in `docs/future_roadmap.md` for potential future integration, particularly regarding advanced earmarking, deposit management, and other enhancements. 