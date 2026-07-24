"""Append Phase 3 schema compliance and error recovery tasks to the YAML.

This script generates YAML content for two new category sections:
  1. schema_compliance — 20 tasks testing strict JSON Schema constraints
  2. error_recovery    — 18 tasks testing error handling and self-correction
"""

# --------------------------------------------------------------------------- #
# Shared schema blocks (6-space indented YAML)
# --------------------------------------------------------------------------- #

CREATE_TICKET_SCHEMA = """      - name: create_ticket
        description: "Create a new issue ticket in a project management system"
        parameters:
          type: object
          properties:
            title:
              type: string
              minLength: 5
              maxLength: 200
              description: "Ticket title, must be between 5 and 200 characters"
            description:
              type: string
              maxLength: 2000
              description: "Detailed ticket description"
              default: ""
            labels:
              type: array
              items:
                type: string
                enum: [bug, feature, enhancement, documentation, security]
              description: "Labels for the ticket"
              default: []
            priority:
              type: string
              enum: [p0, p1, p2, p3, p4]
              description: "Priority level"
            assignees:
              type: array
              items:
                type: object
                properties:
                  username:
                    type: string
                    pattern: "^[a-zA-Z0-9._-]+$"
                  role:
                    type: string
                    enum: [primary, secondary]
                required: [username]
              description: "People to assign to the ticket"
              default: []
          required: [title, priority]"""

SEARCH_SCHEMA = """      - name: search
        description: "Search the web for information"
        parameters:
          type: object
          properties:
            query:
              type: string
              description: "The search query"
            max_results:
              type: integer
              description: "Maximum number of results to return"
              default: 5
          required: [query]"""

WEATHER_SCHEMA = """      - name: get_weather
        description: "Get the current weather for a location"
        parameters:
          type: object
          properties:
            location:
              type: string
              description: "City or location name"
            unit:
              type: string
              enum: [celsius, fahrenheit]
              description: "Temperature unit"
              default: celsius
          required: [location]"""

STOCK_SCHEMA = """      - name: get_stock_price
        description: "Get the current stock price for a symbol"
        parameters:
          type: object
          properties:
            symbol:
              type: string
              description: "Stock ticker symbol (e.g. AAPL, NVDA, MSFT)"
          required: [symbol]"""

CALC_SCHEMA = """      - name: calculate
        description: "Evaluate a mathematical expression"
        parameters:
          type: object
          properties:
            expression:
              type: string
              description: "A valid mathematical expression (e.g. '2 + 3 * 4')"
          required: [expression]"""

EMAIL_SCHEMA = """      - name: send_email
        description: "Send an email message"
        parameters:
          type: object
          properties:
            to:
              type: string
              description: "Recipient email address"
            subject:
              type: string
              description: "Email subject line"
            body:
              type: string
              description: "Email body content"
          required: [to, subject, body]"""

MEETING_SCHEMA = """      - name: schedule_meeting
        description: "Schedule a calendar meeting"
        parameters:
          type: object
          properties:
            title:
              type: string
              description: "Meeting title"
            datetime:
              type: string
              description: "Meeting datetime in ISO format (e.g. '2026-07-25T14:00:00')"
            attendees:
              type: array
              items:
                type: string
              description: "List of attendee email addresses"
          required: [title, datetime, attendees]"""


TOOLS_BLOCK = {
    "create_ticket": CREATE_TICKET_SCHEMA,
    "search": SEARCH_SCHEMA,
    "get_weather": WEATHER_SCHEMA,
    "get_stock_price": STOCK_SCHEMA,
    "calculate": CALC_SCHEMA,
    "send_email": EMAIL_SCHEMA,
    "schedule_meeting": MEETING_SCHEMA,
}


# --------------------------------------------------------------------------- #
# YAML helpers
# --------------------------------------------------------------------------- #

def _esc(s):
    """Escape a string for double-quoted YAML."""
    return '"' + s.replace('"', '\\"') + '"'


def _yaml_dict(d, indent=6):
    """Convert a Python dict to indented YAML."""
    lines = []
    for k, v in d.items():
        prefix = " " * indent
        if isinstance(v, str):
            lines.append(f"{prefix}{k}: {_esc(v)}")
        elif isinstance(v, bool):
            lines.append(f"{prefix}{k}: {str(v).lower()}")
        elif isinstance(v, (int, float)) or v is None:
            lines.append(f"{prefix}{k}: {str(v)}")
        elif isinstance(v, list):
            if v and isinstance(v[0], dict):
                # List of objects (assignees)
                lines.append(f"{' ' * (indent - 2)}{k}:")
                for entry in v:
                    lines.append(f"{' ' * indent}-")
                    for ek, ev in entry.items():
                        if isinstance(ev, str):
                            lines.append(f"{' ' * indent}  {ek}: {_esc(ev)}")
                        else:
                            lines.append(f"{' ' * indent}  {ek}: {ev}")
            else:
                items = ", ".join(f'"{i}"' if isinstance(i, str) else str(i) for i in v)
                lines.append(f"{prefix}{k}: [{items}]")
        elif isinstance(v, dict):
            lines.append(f"{prefix}{k}:")
            lines.append(_yaml_dict(v, indent + 2))
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Schema compliance task builder
# --------------------------------------------------------------------------- #

def make_schema_task(id_, prompt, expected_tool, expected_params,
                     strict, constraint_detail,
                     tools_name="create_ticket"):
    """Build a YAML string for one schema compliance task."""
    tools_block = TOOLS_BLOCK[tools_name]
    expected_section = "    expected:\n      tool: " + expected_tool + "\n      params:\n"
    expected_section += _yaml_dict(expected_params, indent=8)

    scoring = f"""    scoring:
      strict_compliance: {str(strict).lower()}
      required_present: true
      enum_valid: true
      pattern_valid: true
      length_valid: true
      nested_valid: true
      constraint_detail: "{constraint_detail}"
"""
    return f"""
  # ── {id_}: {constraint_detail} ──
  - id: {id_}
    category: schema_compliance
    prompt: {_esc(prompt)}
    tools:
{tools_block}
{expected_section}
{scoring}
"""


# --------------------------------------------------------------------------- #
# Error recovery task builder
# --------------------------------------------------------------------------- #

def make_error_task(id_, prompt, tools_name, expected_sequence,
                    expected_turns, recovery_type,
                    error_recovery=True, self_corrected=True, turns_correct=True):
    """Build a YAML string for one error recovery task."""
    tools_block = TOOLS_BLOCK[tools_name]

    seq_lines = []
    for s in expected_sequence:
        seq_lines.append(f"      - step: {s['step']}")
        seq_lines.append(f"        tool: {s['tool']}")
        if s.get("params"):
            seq_lines.append("        params:\n" + _yaml_dict(s["params"], indent=10))
        else:
            seq_lines.append("        params: {}")
    seq_section = "\n".join(seq_lines)

    scoring = f"""    scoring:
      error_recovery: {str(error_recovery).lower()}
      self_corrected: {str(self_corrected).lower()}
      recovery_type: "{recovery_type}"
"""

    return f"""
  # ── {id_}: {recovery_type} ──
  - id: {id_}
    category: error_recovery
    prompt: {_esc(prompt)}
    tools:
{tools_block}
    expected_turns: {expected_turns}
    expected_sequence:
{seq_section}
{scoring}
"""


# --------------------------------------------------------------------------- #
# Build all tasks
# --------------------------------------------------------------------------- #

SCHEMA_TASKS = []

# --- 1. Title too short (minLength violation) ---
SCHEMA_TASKS.append(make_schema_task(
    "schema_title_min_length_fail",
    "Create a ticket with title 'Short' and priority p3",
    "create_ticket", {"title": "Short", "priority": "p3"},
    strict=False, constraint_detail="title violates minLength (too short)",
))

# --- 2. Title too long (maxLength violation) ---
SCHEMA_TASKS.append(make_schema_task(
    "schema_title_max_length_fail",
    "Create a ticket with this extremely long title that exceeds 200 chars: This is an extremely long ticket title that exceeds the maximum allowed length of 200 characters and should be rejected by schema validation rules",
    "create_ticket",
    {"title": "This is an extremely long ticket title that exceeds the maximum allowed length of 200 characters and should be rejected by schema validation rules", "priority": "p3"},
    strict=False, constraint_detail="title violates maxLength (too long)",
))

# --- 3. Valid simple ticket ---
SCHEMA_TASKS.append(make_schema_task(
    "schema_title_min_length_pass",
    "Create a ticket with title 'Valid title here' and priority p3",
    "create_ticket", {"title": "Valid title here", "priority": "p3"},
    strict=True, constraint_detail="all constraints satisfied (minimal valid)",
))

# --- 4. Nested assignee with valid pattern ---
SCHEMA_TASKS.append(make_schema_task(
    "schema_nested_valid",
    "Create a P3 feature ticket titled 'Add dark mode support' for user alice with primary role",
    "create_ticket",
    {"title": "Add dark mode support", "priority": "p3", "labels": ["feature"],
     "assignees": [{"username": "alice", "role": "primary"}]},
    strict=True, constraint_detail="nested object with valid pattern",
))

# --- 5. Nested assignee with invalid pattern ---
SCHEMA_TASKS.append(make_schema_task(
    "schema_nested_invalid_pattern",
    "Create a P3 ticket titled 'Bad assignee' for user 'bad@user!' with primary role",
    "create_ticket",
    {"title": "Bad assignee", "priority": "p3",
     "assignees": [{"username": "bad@user!", "role": "primary"}]},
    strict=False, constraint_detail="username violates pattern (contains @ and !)",
))

# --- 6. Invalid enum value for label ---
SCHEMA_TASKS.append(make_schema_task(
    "schema_invalid_label_enum",
    "Create a ticket with title 'Invalid label test' and priority p2 and label 'critical'",
    "create_ticket",
    {"title": "Invalid label test", "priority": "p2", "labels": ["critical"]},
    strict=False, constraint_detail="label 'critical' not in enum",
))

# --- 7. Invalid enum value for priority ---
SCHEMA_TASKS.append(make_schema_task(
    "schema_invalid_priority_enum",
    "Create a ticket titled 'Bad priority' with priority p99",
    "create_ticket",
    {"title": "Bad priority", "priority": "p99"},
    strict=False, constraint_detail="priority 'p99' not in enum",
))

# --- 8. Valid enum values ---
SCHEMA_TASKS.append(make_schema_task(
    "schema_valid_enums",
    "Create a P0 security ticket titled 'SQL injection vulnerability' for user bob with secondary role",
    "create_ticket",
    {"title": "SQL injection vulnerability", "priority": "p0", "labels": ["security"],
     "assignees": [{"username": "bob", "role": "secondary"}]},
    strict=True, constraint_detail="all enums valid",
))

# --- 9. Missing required priority field ---
SCHEMA_TASKS.append(make_schema_task(
    "schema_missing_priority",
    "Create a ticket titled 'Missing priority' with label bug",
    "create_ticket",
    {"title": "Missing priority", "labels": ["bug"]},
    strict=False, constraint_detail="missing required field: priority",
))

# --- 10. Missing required title field ---
SCHEMA_TASKS.append(make_schema_task(
    "schema_missing_title",
    "Create a P2 ticket with label feature for alice",
    "create_ticket",
    {"priority": "p2", "labels": ["feature"],
     "assignees": [{"username": "alice", "role": "primary"}]},
    strict=False, constraint_detail="missing required field: title",
))

# --- 11. Multiple labels with valid enums ---
SCHEMA_TASKS.append(make_schema_task(
    "schema_multi_labels",
    "Create a P2 bug ticket titled 'Mobile login broken' with labels bug and enhancement",
    "create_ticket",
    {"title": "Mobile login broken", "priority": "p2", "labels": ["bug", "enhancement"]},
    strict=True, constraint_detail="array items all valid enums",
))

# --- 12. Multiple labels with one invalid ---
SCHEMA_TASKS.append(make_schema_task(
    "schema_mixed_labels",
    "Create a ticket titled 'Mixed labels' with labels bug and invalid_label",
    "create_ticket",
    {"title": "Mixed labels", "priority": "p2", "labels": ["bug", "invalid_label"]},
    strict=False, constraint_detail="label 'invalid_label' not in enum",
))

# --- 13. Multiple assignees ---
SCHEMA_TASKS.append(make_schema_task(
    "schema_multi_assignees",
    "Create a P2 ticket titled 'Team task' for alice primary and bob secondary",
    "create_ticket",
    {"title": "Team task", "priority": "p2",
     "assignees": [{"username": "alice", "role": "primary"},
                   {"username": "bob", "role": "secondary"}]},
    strict=True, constraint_detail="multiple nested objects valid",
))

# --- 14. Multiple assignees with invalid pattern ---
SCHEMA_TASKS.append(make_schema_task(
    "schema_multi_assignees_invalid",
    "Create a P2 ticket titled 'Bad team' for user 'test@user' primary and alice secondary",
    "create_ticket",
    {"title": "Bad team", "priority": "p2",
     "assignees": [{"username": "test@user", "role": "primary"},
                   {"username": "alice", "role": "secondary"}]},
    strict=False, constraint_detail="username 'test@user' violates pattern",
))

# --- 15. Valid with description ---
SCHEMA_TASKS.append(make_schema_task(
    "schema_with_description",
    "Create a P1 bug ticket titled 'Dashboard not loading' with description 'Users report blank screen on load'",
    "create_ticket",
    {"title": "Dashboard not loading", "priority": "p1",
     "description": "Users report blank screen on load",
     "labels": ["bug"]},
    strict=True, constraint_detail="all fields valid including optional description",
))

# --- 16. Invalid assignee role ---
SCHEMA_TASKS.append(make_schema_task(
    "schema_invalid_role",
    "Create a P2 ticket titled 'Bad role' for alice with role admin",
    "create_ticket",
    {"title": "Bad role", "priority": "p2",
     "assignees": [{"username": "alice", "role": "admin"}]},
    strict=False, constraint_detail="assignee role 'admin' not in enum",
))

# --- 17. Valid minimal (only required fields) ---
SCHEMA_TASKS.append(make_schema_task(
    "schema_minimal_valid",
    "Create a P4 ticket titled 'Minimal ticket'",
    "create_ticket",
    {"title": "Minimal ticket", "priority": "p4"},
    strict=True, constraint_detail="only required fields, no extras",
))

# --- 18. Nested object with missing username ---
SCHEMA_TASKS.append(make_schema_task(
    "schema_missing_nested_user",
    "Create a P2 ticket titled 'No username' for assignee with just role primary",
    "create_ticket",
    {"title": "No username", "priority": "p2",
     "assignees": [{"role": "primary"}]},
    strict=False, constraint_detail="nested assignee missing required username",
))

# --- 19. All valid with everything ---
SCHEMA_TASKS.append(make_schema_task(
    "schema_all_valid_complex",
    "Create a P0 security ticket titled 'Critical XSS vulnerability' with description 'Reflected XSS in search endpoint' for alice primary",
    "create_ticket",
    {"title": "Critical XSS vulnerability", "priority": "p0",
     "description": "Reflected XSS in search endpoint",
     "labels": ["security"],
     "assignees": [{"username": "alice", "role": "primary"}]},
    strict=True, constraint_detail="complex valid ticket with all fields",
))

# --- 20. Pattern edge case (valid special chars) ---
SCHEMA_TASKS.append(make_schema_task(
    "schema_pattern_valid_edge",
    "Create a P2 ticket titled 'Edge case' for user 'a.b-c_d' with role primary",
    "create_ticket",
    {"title": "Edge case", "priority": "p2",
     "assignees": [{"username": "a.b-c_d", "role": "primary"}]},
    strict=True, constraint_detail="username with dots, dashes, underscores valid",
))


ERROR_RECOVERY_TASKS = []


def err_task(id_, prompt, tools_name, expected_sequence, expected_turns,
             recovery_type, error_recovery=True, self_corrected=True):
    """Helper to build an error recovery YAML task."""
    return make_error_task(id_, prompt, tools_name, expected_sequence,
                          expected_turns, recovery_type,
                          error_recovery=error_recovery,
                          self_corrected=self_corrected)


# --- search error recovery ---
ERROR_RECOVERY_TASKS.append(err_task(
    "err_search_empty_query",
    "Search for information about machine learning",
    "search",
    [
        {"step": 1, "tool": "search", "params": {"query": ""}},
        {"step": 2, "tool": "search", "params": {"query": "machine learning"}},
    ],
    2, "empty required query param",
))

# --- weather error recovery ---
ERROR_RECOVERY_TASKS.append(err_task(
    "err_weather_unknown_location",
    "What's the weather like in Paris?",
    "get_weather",
    [
        {"step": 1, "tool": "get_weather", "params": {"location": "XYZ123"}},
        {"step": 2, "tool": "get_weather", "params": {"location": "Paris"}},
    ],
    2, "unknown location → retry with valid location",
))

# --- stock error recovery ---
ERROR_RECOVERY_TASKS.append(err_task(
    "err_stock_unknown_symbol",
    "What's the price of AAPL stock?",
    "get_stock_price",
    [
        {"step": 1, "tool": "get_stock_price", "params": {"symbol": "XYZ"}},
        {"step": 2, "tool": "get_stock_price", "params": {"symbol": "AAPL"}},
    ],
    2, "unknown symbol → retry with valid symbol",
))

# --- calculate error recovery ---
ERROR_RECOVERY_TASKS.append(err_task(
    "err_calc_empty_expr",
    "Calculate 100 times 2",
    "calculate",
    [
        {"step": 1, "tool": "calculate", "params": {"expression": ""}},
        {"step": 2, "tool": "calculate", "params": {"expression": "100 * 2"}},
    ],
    2, "empty expression → retry with valid expression",
))

# --- email error recovery ---
ERROR_RECOVERY_TASKS.append(err_task(
    "err_email_missing_fields",
    "Send an email to alice@example.com about the project update",
    "send_email",
    [
        {"step": 1, "tool": "send_email", "params": {"to": "alice@example.com"}},
        {"step": 2, "tool": "send_email",
         "params": {"to": "alice@example.com", "subject": "Project update", "body": "Hi, about the project."}},
    ],
    2, "missing required subject/body → retry with all fields",
))

# --- weather invalid enum ---
ERROR_RECOVERY_TASKS.append(err_task(
    "err_weather_bad_unit",
    "Get the weather in Tokyo in kelvin",
    "get_weather",
    [
        {"step": 1, "tool": "get_weather", "params": {"location": "Tokyo", "unit": "kelvin"}},
        {"step": 2, "tool": "get_weather", "params": {"location": "Tokyo", "unit": "celsius"}},
    ],
    2, "invalid unit enum → retry with valid unit",
))

# --- search generic query ---
ERROR_RECOVERY_TASKS.append(err_task(
    "err_search_generic_query",
    "Find me something interesting",
    "search",
    [
        {"step": 1, "tool": "search", "params": {"query": "something"}},
        {"step": 2, "tool": "search", "params": {"query": "interesting technology news"}},
    ],
    2, "generic query too vague → retry with specific query",
))

# --- email bad format ---
ERROR_RECOVERY_TASKS.append(err_task(
    "err_email_bad_format",
    "Send an email to invalid-email about the project",
    "send_email",
    [
        {"step": 1, "tool": "send_email",
         "params": {"to": "invalid-email", "subject": "Project update", "body": "Hi."}},
        {"step": 2, "tool": "send_email",
         "params": {"to": "invalid-email@example.com", "subject": "Project update", "body": "Hi."}},
    ],
    2, "malformed email → retry with valid format",
))

# --- ticket short title ---
ERROR_RECOVERY_TASKS.append(err_task(
    "err_ticket_short_title",
    "Create a ticket titled 'X' and priority p2",
    "create_ticket",
    [
        {"step": 1, "tool": "create_ticket", "params": {"title": "X", "priority": "p2"}},
        {"step": 2, "tool": "create_ticket", "params": {"title": "Title is too short", "priority": "p2"}},
    ],
    2, "title too short → retry with valid title",
))

# --- ticket invalid label ---
ERROR_RECOVERY_TASKS.append(err_task(
    "err_ticket_invalid_label",
    "Create a ticket titled 'Label test' priority p1 with label urgent",
    "create_ticket",
    [
        {"step": 1, "tool": "create_ticket", "params": {"title": "Label test", "priority": "p1", "labels": ["urgent"]}},
        {"step": 2, "tool": "create_ticket", "params": {"title": "Label test", "priority": "p1", "labels": ["bug"]}},
    ],
    2, "invalid label enum → retry with valid label",
))

# --- stock chain: get price then calculate ---
ERROR_RECOVERY_TASKS.append(err_task(
    "err_stock_calc_chain",
    "Get MSFT price, then calculate a 10% increase",
    "get_stock_price",
    [
        {"step": 1, "tool": "get_stock_price", "params": {"symbol": "MSFT"}},
        {"step": 2, "tool": "calculate", "params": {"expression": "420 * 1.10"}},
    ],
    2, "chain: stock then calculate (potential error in chain)",
))

# --- weather + stock chain ---
ERROR_RECOVERY_TASKS.append(err_task(
    "err_weather_stock_chain",
    "Check weather in Sydney and NVDA stock price",
    "get_weather",
    [
        {"step": 1, "tool": "get_weather", "params": {"location": "Sydney"}},
        {"step": 2, "tool": "get_stock_price", "params": {"symbol": "NVDA"}},
    ],
    2, "parallel chain: weather + stock",
))

# --- search + stock chain ---
ERROR_RECOVERY_TASKS.append(err_task(
    "err_search_stock_chain",
    "Search for climate change news and get Google stock price",
    "search",
    [
        {"step": 1, "tool": "search", "params": {"query": "climate change news 2026"}},
        {"step": 2, "tool": "get_stock_price", "params": {"symbol": "GOOGL"}},
    ],
    2, "cross-domain chain: search + stock",
))

# --- weather multi-turn with error ---
ERROR_RECOVERY_TASKS.append(err_task(
    "err_weather_multi_location",
    "Get the weather in three cities: Tokyo, London, and Paris",
    "get_weather",
    [
        {"step": 1, "tool": "get_weather", "params": {"location": "Tokyo"}},
        {"step": 2, "tool": "get_weather", "params": {"location": "London"}},
        {"step": 3, "tool": "get_weather", "params": {"location": "Paris"}},
    ],
    3, "multi-city weather (error if one location unknown)",
))

# --- meeting missing attendees ---
ERROR_RECOVERY_TASKS.append(err_task(
    "err_meeting_missing_attendees",
    "Schedule a meeting called 'Sprint Review' for 2026-08-01 at 10am",
    "schedule_meeting",
    [
        {"step": 1, "tool": "schedule_meeting",
         "params": {"title": "Sprint Review", "datetime": "2026-08-01T10:00:00"}},
        {"step": 2, "tool": "schedule_meeting",
         "params": {"title": "Sprint Review", "datetime": "2026-08-01T10:00:00", "attendees": ["alice@example.com"]}},
    ],
    2, "missing required attendees → retry with attendees",
))

# --- calc chain with error ---
ERROR_RECOVERY_TASKS.append(err_task(
    "err_calc_chain",
    "Calculate 100 + 200, then multiply by 3",
    "calculate",
    [
        {"step": 1, "tool": "calculate", "params": {"expression": ""}},
        {"step": 2, "tool": "calculate", "params": {"expression": "(100 + 200) * 3"}},
    ],
    2, "empty expression → retry with compound expression",
))

# --- stock then currency ---
ERROR_RECOVERY_TASKS.append(err_task(
    "err_stock_currency_chain",
    "Get NVDA price and convert 100 USD to EUR",
    "get_stock_price",
    [
        {"step": 1, "tool": "get_stock_price", "params": {"symbol": "NVDA"}},
        {"step": 2, "tool": "convert_currency", "params": {"amount": 100, "from_currency": "USD", "to_currency": "EUR"}},
    ],
    2, "cross-domain chain: stock + currency",
))


# --------------------------------------------------------------------------- #
# Write to file
# --------------------------------------------------------------------------- #

OUTPUT_PATH = "/home/colin-spark/Projects/model-benchmarks/datasets/tool_calling_tasks.yaml"

with open(OUTPUT_PATH, "a") as f:
    # Schema compliance section header
    f.write("\n")
    f.write("# ──────────────────────────────────────────────────────────────────────────\n")
    f.write("# Phase 3 — Schema Compliance\n")
    f.write("#\n")
    f.write("# Tests strict adherence to complex nested JSON schemas. Each task uses\n")
    f.write("# the create_ticket tool with constraints: minLength, maxLength, pattern,\n")
    f.write("# enum, required fields, nested objects (assignees), and arrays.\n")
    f.write("#\n")
    f.write("# Scoring: strict_compliance (all constraints satisfied?), required_present,\n")
    f.write("# enum_valid, pattern_valid, length_valid, nested_valid.\n")
    f.write("# ──────────────────────────────────────────────────────────────────────────\n\n")

    for t in SCHEMA_TASKS:
        f.write(t)

    # Error recovery section header
    f.write("\n")
    f.write("# ──────────────────────────────────────────────────────────────────────────\n")
    f.write("# Phase 3 — Error Recovery\n")
    f.write("#\n")
    f.write("# Tests whether the model self-corrects when a tool returns an error.\n")
    f.write("# Each task describes a sequence: model calls tool with bad params →\n")
    f.write("# error response → model retries with corrected params.\n")
    f.write("#\n")
    f.write("# Scoring:\n")
    f.write("#   - error_recovery: true if model corrects after error\n")
    f.write("#   - self_corrected: did the model fix the bad params?\n")
    f.write("#   - recovery_type: what kind of error was encountered\n")
    f.write("# ──────────────────────────────────────────────────────────────────────────\n\n")

    for t in ERROR_RECOVERY_TASKS:
        f.write(t)

print(f"Wrote {len(SCHEMA_TASKS)} schema compliance tasks")
print(f"Wrote {len(ERROR_RECOVERY_TASKS)} error recovery tasks")
print(f"Total new tasks: {len(SCHEMA_TASKS) + len(ERROR_RECOVERY_TASKS)}")
print(f"File: {OUTPUT_PATH}")