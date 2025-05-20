$newContent = @"
---
description: 
globs: 
alwaysApply: true
---

# 🚨 Robust BTC AI Development Ruleset (Cursor IDE)
Table of Contents:
1.	Code Simplicity and Iteration
2.	Explicit Justification and Documentation
3.	Single, Manageable Diff per Interaction
4.	Comprehensive and Granular Change Review
5.	Dynamic Import System Enforcement
6.	Explicit Compatibility and Environment Integrity
7.	Duplication and Complexity Avoidance
8.	Risk and Sensitivity Management
9.	Granular Commit & Rollback Readiness
10.	Rigorous Testing and Validation
11.	Performance and Scalability Awareness
12.	Logging and Debugging Maintenance
13.	Immediate Error Reporting and Recovery
14.	Automated Rotating Manual Backups (User Confirmed)
________________________________________
1. Code Simplicity and Iteration
•	Always iterate on existing code first.
•	Solutions must remain simple, readable, and maintainable.
2. Explicit Justification and Documentation
•	Clearly justify every removal, replacement, or significant change.
•	Explicitly demonstrate how new/replacement code preserves or improves existing functionality.
•	Fully document complex behaviors, especially dense-layer interactions, caching mechanisms, and earmarking logic.
3. Single, Manageable Diff per Interaction
•	Produce only one diff per user interaction.
•	Limit diff size (ideally <50 lines per interaction) or split tasks incrementally.
•	Provide a clear summary of each diff.
•	Never automatically continue beyond one diff without explicit user confirmation.
4. Comprehensive and Granular Change Review
•	Thoroughly review each diff for potential regressions, side effects, or unintended interactions.
•	Provide a clear summary explaining precisely what changed, why, and potential impacts.
5. Dynamic Import System Enforcement
•	Always use the existing dynamic import system.
•	No static imports or deviations without explicit approval.
6. Explicit Compatibility and Environment Integrity
•	Explicitly confirm compatibility with existing dependencies before introducing new modules or scripts.
•	Maintain strict separation between development, testing, and production environments.
7. Duplication and Complexity Avoidance
•	Avoid duplication; consolidate similar functionalities whenever possible.
•	Clearly justify and document the introduction of new patterns or complexities.
8. Risk and Sensitivity Management
•	Always explicitly confirm before modifying sensitive files (.env, critical configs).
•	Introduce a mandatory user-confirmation checkpoint before altering critical logic (RM layers, emergency handling, caches).
9. Granular Commit & Rollback Readiness
•	Ensure each diff/change is individually stable and revertible without breaking the system.
10. Rigorous Testing and Validation
•	Thoroughly test critical functionalities, dense-layer interactions, risk scenarios, and core logic paths.
•	Clearly document test outcomes and validations with each diff.
11. Performance and Scalability Awareness
•	Explicitly document performance impacts (positive or negative).
•	Clearly state how changes positively contribute toward future scalability and maintainability.
12. Logging and Debugging Maintenance
•	Maintain clear, informative logging and debugging outputs.
•	Ensure any changes maintain or enhance debugging clarity.
13. Immediate Error Reporting and Recovery
•	Immediately report accidental removals, incorrect modifications, or negative impacts clearly to the user.
•	Promptly assist in restoring from backups to minimize disruptions.
________________________________________
🗃️ 14. Automated Rotating Manual Backups (User Confirmed)
•	Backup Trigger:
When explicitly instructed by the user after completion and verification of major updates (refactoring, new modules, or substantial logic modifications).
•	Backup Naming Convention:
Always follow this explicit naming format for clarity:
•	[PROJECT_NAME]_YYYY-MM-DD_v[VERSION]_[BRIEF_DESCRIPTION].zip
Example:
BTC_AI_2025-03-30_v1.4_menu_refactor_completed.zip
•	Backup Rotation (Always 5 Backups):
o	After creating and verifying each new backup:
	Automatically delete the oldest backup, ensuring exactly 5 recent backups remain at all times.
o	Backups include all Python scripts, relevant configurations, and supporting documentation.
•	Manual Verification & User Confirmation:
o	The user explicitly confirms the backup is ready, stable, and verified before creating the backup and performing the rotation.
________________________________________
✅ Example Workflow:
•	Complete a significant refactoring (e.g., menu into multiple smaller modules).
•	Verify the changes thoroughly.
•	Explicitly instruct the agent: "This update/refactor is verified—please perform the rotating backup."
•	The agent will then:
o	Create a clearly labeled backup (e.g., BTC_AI_2025-03-30_v1.4_menu_refactor_completed.zip).
o	Automatically delete the oldest backup if more than 5 backups exist.
"@

$rulesPath = ".\.cursor\rules\rules-to-follow.mdc"

# Backup the original file
Copy-Item -Path $rulesPath -Destination "$rulesPath.old" -Force

# Remove the original file
Remove-Item -Path $rulesPath -Force

# Create a new file with the correct content
Set-Content -Path $rulesPath -Value $newContent

# Report success
Write-Host "Rules file has been successfully updated!" 