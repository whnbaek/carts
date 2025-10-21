# Notion Tasks Guide (CARTS)

This guide explains how we create and manage task cards for the "Next Steps Dashboard" in Notion using the Notion MCP tools. It covers property formats, common pitfalls, and the recommended workflow.

## Database Schema
The dashboard database lives under the `Next Steps` page and defines these properties:
- `Task` (title)
- `Status` (select): To Do, In Progress, Blocked, Done
- `Priority` (select): High, Medium, Low
- `Category` (select): Task, Enhancement, Research
- `Area` (multi_select): OpenMP, Pass Pipeline, Examples, CI/CD, Docs, Codebase, Benchmarking, Visualization, Auto-tuning, Dialects, GPU
- `Due` (date)
- `Owner` (people)
- `Notes` (rich_text)

Important: The title property name is `Task` (not `title`). Always use the database’s actual property names verbatim.

## Golden Rules
- Do all create/update operations in a batch of 1. Larger batches can time out or fail.
- For multi_select values (e.g., `Area`) send a JSON STRING, not an array. Example: "[\"OpenMP\",\"Pass Pipeline\"]".
- Use only valid select options for `Status`, `Priority`, and `Category`.
- If replacing page content, either provide a unique selection range or replace the entire content to avoid “multiple occurrences” errors.

## Create the Database (one-time)
If you ever need to recreate it (parent is the `Next Steps` page):
```json
{
  "parent": {"page_id": "<NEXT_STEPS_PAGE_ID>"},
  "title": [{"type": "text", "text": {"content": "Next Steps Dashboard"}}],
  "properties": {
    "Task": {"title": {}},
    "Status": {"select": {"options": [
      {"name": "To Do", "color": "yellow"},
      {"name": "In Progress", "color": "blue"},
      {"name": "Blocked", "color": "red"},
      {"name": "Done", "color": "green"}
    ]}},
    "Priority": {"select": {"options": [
      {"name": "High", "color": "red"},
      {"name": "Medium", "color": "orange"},
      {"name": "Low", "color": "green"}
    ]}},
    "Category": {"select": {"options": [
      {"name": "Task", "color": "gray"},
      {"name": "Enhancement", "color": "purple"},
      {"name": "Research", "color": "brown"}
    ]}} ,
    "Area": {"multi_select": {"options": [
      {"name": "OpenMP"}, {"name": "Pass Pipeline"}, {"name": "Examples"},
      {"name": "CI/CD"}, {"name": "Docs"}, {"name": "Codebase"},
      {"name": "Benchmarking"}, {"name": "Visualization"}, {"name": "Auto-tuning"},
      {"name": "Dialects"}, {"name": "GPU"}
    ]}},
    "Due": {"date": {}},
    "Owner": {"people": {}},
    "Notes": {"rich_text": {}}
  }
}
```

## Create a Task (batch of 1)
Step 1 — Create minimal card with just `Task` and `Status`:
```json
{
  "parent": {"db_id": "<NEXT_STEPS_DB_ID>"},
  "pages": [
    {"properties": {"Task": "Expanded OpenMP Support", "Status": "To Do"}}
  ]
}
```

Step 2 — Enrich properties via update on that page:
```json
{
  "page_id": "<TASK_PAGE_ID>",
  "command": "update_properties",
  "properties": {
    "Priority": "High",
    "Category": "Enhancement",
    "Area": "[\"OpenMP\",\"Pass Pipeline\"]",
    "Notes": "Expand coverage of OpenMP directives (taskloop, depend, reductions, collapse). Build a conformance mini-suite; add regression tests."
  }
}
```
Notes:
- `Area` must be a JSON string encoding the array (escaped quotes).
- `Owner` requires Notion user IDs (usually set via UI unless you have IDs).
- `Due` expects ISO date/time if used (e.g., 2025-08-08).

## Add Detailed Steps to a Task Page
To replace the task’s page content with step-by-step instructions:
```json
{
  "page_id": "<TASK_PAGE_ID>",
  "command": "replace_content",
  "new_str": "# Task: Expanded OpenMP Support\n\n## Context\nIncrease coverage of OpenMP constructs...\n\n## Steps\n1. Inventory supported constructs\n2. Create conformance mini-suite\n3. Patch Polygeist / ConvertOMPToARTS\n4. Add tests and docs\n\n## Verification\nConformance green\n\n## Deliverables\nPRs, tests, docs"
}
```

## Common Errors
- Validation: “expected string received array” → send multi-select values (e.g., `Area`) as a JSON string.
- Validation: “multiple occurrences found” → narrow the selection or replace the whole content.
- Not found: ensure items are shared with the integration; verify IDs.
- Timeouts/no result: reduce payload; operate on one item at a time.

## Recommended Workflow
1. Create each task with minimal properties (Task + Status).
2. Update properties in a follow-up call (Priority, Category, Area as JSON string, Notes, optionally Due/Owner).
3. Replace the task page content with Context/Prerequisites/Steps/Verification/Deliverables.
4. In Notion UI, add a Board view grouped by `Status` and any filtered views (e.g., High Priority, Research).

## Quick Reference — Valid Options
- Status: To Do, In Progress, Blocked, Done
- Priority: High, Medium, Low
- Category: Task, Enhancement, Research
- Area: OpenMP, Pass Pipeline, Examples, CI/CD, Docs, Codebase, Benchmarking, Visualization, Auto-tuning, Dialects, GPU

