import json
import os
import re
import sqlite3
import sys
import tempfile
import urllib.request
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SYSTEM_PROMPT_PATH = os.path.join(BASE_DIR, "system_prompt.md")
NIM_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
MODEL = "openai/gpt-oss-20b"
TEST_QUESTION = "How many customers are there?"
DEFAULT_DATA_FILE = os.path.join(BASE_DIR, "database", "sales.db")
CHART_DIR = BASE_DIR
LOG_PATH = os.path.join(BASE_DIR, "reasoning_log.json")
TEMP_DB_PATH = os.path.join(BASE_DIR, "temp_data.db")
FORBIDDEN_KEYWORDS = ("INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE")
REFUSAL_MESSAGE = "Refused: only read-only SELECT queries are allowed."

CHART_THEMES = {
    "corporate": {
        "colors": ["#1f77b4", "#2c5f8a", "#3d7ea6", "#5a9bc2", "#7bb8dc"],
        "background": "#f5f5f5"
    },
    "warm": {
        "colors": ["#ff6b6b", "#ff8c5a", "#ffa947", "#ffc93c", "#ffe066"],
        "background": "#fffaf5"
    },
    "cool": {
        "colors": ["#3891a6", "#4267ac", "#6a4c93", "#8e7ac1", "#a5a8e0"],
        "background": "#f5f7ff"
    },
    "monochrome": {
        "colors": ["#2b2b2b", "#4d4d4d", "#6e6e6e", "#8f8f8f", "#b0b0b0"],
        "background": "#ffffff"
    }
}
DEFAULT_CHART_THEME = "corporate"


def new_log_entry(question, source_file):
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "question": question,
        "source_file": source_file,
        "attempts": [],
        "outcome": None,
        "final_sql": None,
        "error": None,
        "summary": None,
        "chart_path": None,
        "result_columns": None,
        "result_rows": None,
        "data_quality_summary": None,
        "correction_mode": None,
        "correction_log": None,
    }


def load_log():
    if not os.path.exists(LOG_PATH):
        return []
    try:
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, ValueError):
        return []


def append_log(entry):
    log = load_log()
    log.append(entry)
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)


def load_system_prompt():
    with open(SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as f:
        return f.read().strip()


def infer_schema(conn):
    schema = {}
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' "
        "AND name NOT LIKE 'sqlite_%' ORDER BY name"
    )
    tables = [r[0] for r in cur.fetchall()]
    for table in tables:
        cols = []
        for col_row in conn.execute(f'PRAGMA table_info("{table}")'):
            col_name = col_row[1]
            col_type = col_row[2] or "TEXT"
            cols.append((col_name, col_type))
        schema[table] = cols
    return schema


def load_dataset(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".db":
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Database file not found: {file_path}")
        conn = sqlite3.connect(file_path)
        schema = infer_schema(conn)
        return conn, schema, file_path

    import pandas as pd

    if ext == ".csv":
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            raise ValueError(f"Could not parse CSV '{file_path}': {e}")
    elif ext in (".xlsx", ".xls"):
        try:
            df = pd.read_excel(file_path)
        except Exception as e:
            raise ValueError(f"Could not parse Excel file '{file_path}': {e}")
    else:
        raise ValueError(
            f"Unsupported file type '{ext}'. Supported: .csv, .xlsx, .db"
        )

    if df.empty:
        raise ValueError(f"No rows found in '{file_path}'.")
    if list(df.columns) == [str(i) for i in range(len(df.columns))]:
        raise ValueError(
            f"No header row detected in '{file_path}' "
            "(columns are unnamed integers)."
        )

    table_name = os.path.splitext(os.path.basename(file_path))[0]
    if os.path.exists(TEMP_DB_PATH):
        try:
            os.remove(TEMP_DB_PATH)
        except OSError:
            pass
    conn = sqlite3.connect(TEMP_DB_PATH)
    try:
        df.to_sql(table_name, conn, index=False, if_exists="replace")
        conn.commit()
    except Exception:
        conn.close()
        raise
    schema = infer_schema(conn)
    return conn, schema, file_path


def validate_dataset(conn, schema_dict):
    import pandas as pd

    validation = {}
    for table_name, columns in schema_dict.items():
        df = pd.read_sql_query(f'SELECT * FROM "{table_name}"', conn)
        total_rows = len(df)
        if total_rows == 0:
            validation[table_name] = {
                "missing_values": {},
                "duplicate_rows": 0,
                "inconsistent_categories": {},
                "type_mismatches": {}
            }
            continue

        missing = {}
        for col_name, col_type in columns:
            series = df[col_name]
            null_count = int(series.isna().sum())
            empty_str_count = int((series.astype(str).str.strip() == "").sum())
            total_missing = null_count + empty_str_count
            if total_missing > 0:
                missing[col_name] = total_missing

        dup_count = int(df.duplicated().sum())

        inconsistent = {}
        for col_name, col_type in columns:
            if col_type.upper() == "TEXT":
                series = df[col_name].dropna()
                if len(series) == 0:
                    continue
                unique_raw = series.nunique()
                if unique_raw < 20 or (unique_raw / total_rows) < 0.5:
                    normalized = series.astype(str).str.strip().str.lower()
                    variant_groups = normalized.groupby(normalized).size()
                    problematic = {
                        str(raw): int(count)
                        for raw, count in variant_groups.items()
                        if count > 1 or (raw != normalized.iloc[0] and count > 0)
                    }
                    if len(problematic) > 1:
                        grouped = {}
                        for raw_val in series.unique():
                            norm = str(raw_val).strip().lower() if pd.notna(raw_val) else ""
                            if norm not in grouped:
                                grouped[norm] = []
                            grouped[norm].append(str(raw_val))
                        inconsistent_groups = {
                            k: v for k, v in grouped.items() if len(v) > 1
                        }
                        if inconsistent_groups:
                            inconsistent[col_name] = inconsistent_groups

        type_mismatch = {}
        for col_name, col_type in columns:
            if col_type.upper() == "TEXT":
                series = df[col_name].dropna()
                if len(series) == 0:
                    continue
                numeric_like = 0
                non_numeric_outliers = []
                for val in series:
                    s = str(val).strip()
                    if s == "":
                        continue
                    s_clean = s.replace("$", "").replace(",", "").replace("%", "")
                    try:
                        float(s_clean)
                        numeric_like += 1
                    except ValueError:
                        non_numeric_outliers.append(s)
                if len(series) > 0 and (numeric_like / len(series)) >= 0.5 and numeric_like > 0:
                    type_mismatch[col_name] = {
                        "numeric_like_count": numeric_like,
                        "total_count": len(series),
                        "non_numeric_outliers": non_numeric_outliers
                    }

        validation[table_name] = {
            "missing_values": missing,
            "duplicate_rows": dup_count,
            "inconsistent_categories": inconsistent,
            "type_mismatches": type_mismatch
        }
    return validation


def format_validation_report(validation_dict):
    if not validation_dict:
        return "No tables to validate."
    lines = []
    any_issues = False
    for table, checks in validation_dict.items():
        table_issues = []
        if checks["missing_values"]:
            any_issues = True
            for col, count in checks["missing_values"].items():
                table_issues.append(f"- {count} missing values in '{col}' ({count} rows affected)")
        if checks["duplicate_rows"] > 0:
            any_issues = True
            table_issues.append(f"- {checks['duplicate_rows']} exact duplicate row(s)")
        if checks["inconsistent_categories"]:
            any_issues = True
            for col, groups in checks["inconsistent_categories"].items():
                for norm_val, variants in groups.items():
                    table_issues.append(
                        f"- '{col}' column has inconsistent values: {set(variants)} "
                        f"({len(variants)} variants, likely same value)"
                    )
        if checks["type_mismatches"]:
            any_issues = True
            for col, info in checks["type_mismatches"].items():
                msg = f"- '{col}' column: {info['numeric_like_count']} of {info['total_count']} values look numeric but are stored as text"
                if info["non_numeric_outliers"]:
                    msg += f" (non-numeric outliers: {set(info['non_numeric_outliers'])})"
                table_issues.append(msg)
        if table_issues:
            lines.append(f"Data quality check for '{table}':")
            lines.extend(table_issues)
        else:
            lines.append(f"Data quality check for '{table}': No issues found.")
    if not any_issues and lines:
        lines = ["All tables passed data quality checks."]
    return "\n".join(lines)


def auto_correct_dataset(conn, schema_dict, validation_dict):
    import pandas as pd
    import io

    corrected_dfs = {}
    correction_log = {}

    for table_name, columns in schema_dict.items():
        df = pd.read_sql_query(f'SELECT * FROM "{table_name}"', conn)
        original_rows = len(df)
        table_log = {
            "missing_filled": {},
            "duplicates_removed": 0,
            "categories_standardized": {},
            "type_converted": {},
            "unfixable_outliers": {}
        }

        if validation_dict.get(table_name, {}).get("missing_values"):
            for col_name, col_type in columns:
                missing_count = validation_dict[table_name]["missing_values"].get(col_name, 0)
                if missing_count > 0:
                    # Check if this column is a type_mismatch (numeric-like text)
                    is_numeric_like = col_name in validation_dict[table_name].get("type_mismatches", {})
                    if col_type.upper() in ("INTEGER", "REAL") or is_numeric_like:
                        # Compute median from numeric-like values
                        numeric_vals = []
                        for val in df[col_name]:
                            if pd.isna(val):
                                continue
                            s = str(val).strip()
                            if s == "":
                                continue
                            s_clean = s.replace("$", "").replace(",", "").replace("%", "")
                            try:
                                numeric_vals.append(float(s_clean))
                            except ValueError:
                                pass
                        median_val = float(pd.Series(numeric_vals).median()) if numeric_vals else 0.0
                        new_vals = []
                        for val in df[col_name]:
                            if pd.isna(val):
                                new_vals.append(median_val)
                                continue
                            s = str(val).strip()
                            if s == "":
                                new_vals.append(median_val)
                                continue
                            new_vals.append(val)
                        df[col_name] = new_vals
                        table_log["missing_filled"][col_name] = {
                            "filled_with": "median",
                            "value": median_val,
                            "count": missing_count
                        }
                    else:
                        new_vals = []
                        for val in df[col_name]:
                            if pd.isna(val):
                                new_vals.append("Unknown")
                                continue
                            s = str(val).strip()
                            if s == "":
                                new_vals.append("Unknown")
                                continue
                            new_vals.append(val)
                        df[col_name] = new_vals
                        table_log["missing_filled"][col_name] = {
                            "filled_with": "'Unknown'",
                            "value": "Unknown",
                            "count": missing_count
                        }

        if validation_dict.get(table_name, {}).get("duplicate_rows", 0) > 0:
            dup_count = validation_dict[table_name]["duplicate_rows"]
            df = df.drop_duplicates(keep="first")
            table_log["duplicates_removed"] = dup_count

        if validation_dict.get(table_name, {}).get("inconsistent_categories"):
            for col_name, groups in validation_dict[table_name]["inconsistent_categories"].items():
                norm_to_variants = {}
                for norm_val, variants in groups.items():
                    counts = {}
                    for v in variants:
                        counts[v] = int((df[col_name].astype(str) == v).sum())
                    if counts:
                        most_frequent = max(counts, key=counts.get)
                        total_updated = 0
                        for v in variants:
                            if v != most_frequent:
                                updated = int((df[col_name].astype(str) == v).sum())
                                if updated > 0:
                                    df.loc[df[col_name].astype(str) == v, col_name] = most_frequent
                                    total_updated += updated
                        if total_updated > 0:
                            table_log["categories_standardized"][col_name] = {
                                "standardized_to": most_frequent,
                                "variants_found": list(set(variants)),
                                "rows_updated": total_updated
                            }

        if validation_dict.get(table_name, {}).get("type_mismatches"):
            for col_name, info in validation_dict[table_name]["type_mismatches"].items():
                unfixable = []
                converted_count = 0
                new_vals = []
                for val in df[col_name]:
                    if pd.isna(val):
                        new_vals.append(val)
                        continue
                    s = str(val).strip()
                    if s == "":
                        new_vals.append(val)
                        continue
                    s_clean = s.replace("$", "").replace(",", "").replace("%", "")
                    try:
                        num = float(s_clean)
                        new_vals.append(num)
                        converted_count += 1
                    except ValueError:
                        new_vals.append(val)
                        unfixable.append(s)
                df[col_name] = new_vals
                if converted_count > 0 or unfixable:
                    table_log["type_converted"][col_name] = {
                        "converted_count": converted_count,
                        "unfixable_count": len(unfixable),
                        "unfixable_examples": list(set(unfixable))[:10]
                    }
                    if unfixable:
                        table_log["unfixable_outliers"][col_name] = list(set(unfixable))[:10]

        if any([
            table_log["missing_filled"],
            table_log["duplicates_removed"],
            table_log["categories_standardized"],
            table_log["type_converted"]
        ]):
            correction_log[table_name] = table_log
        else:
            correction_log[table_name] = {"message": "No corrections needed"}

        corrected_dfs[table_name] = df

    new_conn = sqlite3.connect(":memory:")
    for table_name, df in corrected_dfs.items():
        df.to_sql(table_name, new_conn, index=False, if_exists="replace")
    new_conn.commit()

    return new_conn, correction_log


def save_cleaned_dataset(new_conn, original_file_path):
    import pandas as pd

    ext = os.path.splitext(original_file_path)[1].lower()
    base_name = os.path.basename(original_file_path)
    dir_name = os.path.dirname(original_file_path)
    cleaned_name = "cleaned_" + base_name
    cleaned_path = os.path.join(dir_name, cleaned_name)

    cur = new_conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' "
        "AND name NOT LIKE 'sqlite_%' ORDER BY name"
    )
    tables = [r[0] for r in cur.fetchall()]

    if ext == ".csv":
        if len(tables) == 1:
            df = pd.read_sql_query(f'SELECT * FROM "{tables[0]}"', new_conn)
            df.to_csv(cleaned_path, index=False)
        else:
            for i, table in enumerate(tables):
                df = pd.read_sql_query(f'SELECT * FROM "{table}"', new_conn)
                if i == 0:
                    df.to_csv(cleaned_path, index=False, mode="w")
                else:
                    df.to_csv(cleaned_path, index=False, mode="a", header=False)
    elif ext in (".xlsx", ".xls"):
        with pd.ExcelWriter(cleaned_path, engine="openpyxl") as writer:
            for table in tables:
                df = pd.read_sql_query(f'SELECT * FROM "{table}"', new_conn)
                df.to_excel(writer, sheet_name=table[:31], index=False)
    elif ext == ".db":
        import shutil
        file_conn = sqlite3.connect(cleaned_path)
        new_conn.backup(file_conn)
        file_conn.close()
    else:
        raise ValueError(f"Unsupported file type for saving: {ext}")

    return cleaned_path


def format_correction_report(correction_log):
    if not correction_log:
        return "No corrections were applied."
    lines = []
    any_corrections = False
    for table, log in correction_log.items():
        if log.get("message") == "No corrections needed":
            lines.append(f"Corrections for '{table}': No corrections needed")
            continue
        table_corrections = []
        if log.get("missing_filled"):
            any_corrections = True
            for col, info in log["missing_filled"].items():
                table_corrections.append(
                    f"- Filled {info['count']} missing values in '{col}' with {info['filled_with']} ({info['value']})"
                )
        if log.get("duplicates_removed", 0) > 0:
            any_corrections = True
            table_corrections.append(f"- Removed {log['duplicates_removed']} duplicate row(s)")
        if log.get("categories_standardized"):
            any_corrections = True
            for col, info in log["categories_standardized"].items():
                variants_str = ", ".join(f"'{v}'" for v in info["variants_found"])
                table_corrections.append(
                    f"- Standardized '{col}': {variants_str} -> '{info['standardized_to']}' "
                    f"({info['rows_updated']} rows updated)"
                )
        if log.get("type_converted"):
            any_corrections = True
            for col, info in log["type_converted"].items():
                msg = f"- Converted '{col}' to numeric ({info['converted_count']} rows)"
                if info["unfixable_count"] > 0:
                    examples = ", ".join(f"'{e}'" for e in info["unfixable_examples"])
                    msg += f", {info['unfixable_count']} values could not be auto-fixed and remain unchanged: [{examples}]"
                table_corrections.append(msg)
        if table_corrections:
            lines.append(f"Corrections applied to '{table}':")
            lines.extend(table_corrections)
        else:
            lines.append(f"Corrections for '{table}': No corrections needed")
    if not any_corrections:
        return "No corrections were applied."
    return "\n".join(lines)


def build_dynamic_system_prompt(schema_dict):
    lines = ["You are a data analyst agent. You convert natural language business questions into SQL queries for a SQLite database.", ""]
    lines.append("Database schema:")
    for table in sorted(schema_dict.keys()):
        cols = [c[0] for c in schema_dict[table]]
        lines.append(f"- {table}({', '.join(cols)})")
    lines.append("")
    notes_added = False
    lines.extend([
        "Notes:",
        "- All joins must use explicit column relationships shown in the schema above.",
        "- If a column is not present in the schema, do not invent it.",
        "- Column types: INTEGER (whole numbers), REAL (decimal numbers), TEXT (strings/dates).",
        "",
        "Rules:",
        "- Only generate SELECT queries. Never generate INSERT, UPDATE, DELETE, or DROP statements.",
        '- Output only the raw SQL query, with no explanation or markdown formatting.',
        '- If the question cannot be answered using this schema, respond with: "CANNOT_ANSWER: [brief reason]" instead of guessing.',
        '- When a question mentions "revenue" or "sales," use the appropriate numeric column. Consider whether to exclude cancelled or refunded records unless the question asks for gross totals.',
    ])
    return "\n".join(lines)


def call_nim(system_prompt, messages, api_key):
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    elif isinstance(messages, list) and messages and not messages[0].get("role"):
        messages = [{"role": "user", "content": m} for m in messages]
    full_messages = []
    if system_prompt:
        full_messages.append({"role": "system", "content": system_prompt})
    full_messages.extend(messages)
    payload = {
        "model": MODEL,
        "messages": full_messages,
        "temperature": 0.0,
        "max_tokens": 1024,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }
    data = json.dumps(payload).encode("utf-8")
    print(f"DEBUG call_nim: model={MODEL}, api_key present={bool(api_key)}, len={len(api_key) if api_key else 0}")
    req = urllib.request.Request(NIM_URL, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    message = body["choices"][0]["message"]
    content = message.get("content")
    if not content:
        # Reasoning models (e.g. openai/gpt-oss) can leave "content" empty and
        # place their text in "reasoning_content" instead. Fall back to it so we
        # never crash on None and still recover the model's output.
        content = message.get("reasoning_content") or ""
    return content.strip()


def run_sql(sql, conn):
    cur = conn.execute(sql)
    rows = cur.fetchall()
    columns = [d[0] for d in cur.description] if cur.description else []
    return columns, rows


def is_destructive_sql(sql):
    upper = sql.upper()
    return any(kw in upper for kw in FORBIDDEN_KEYWORDS)


import traceback

def answer_question(system_prompt, question, api_key, conn, log_entry, max_attempts=3):
    messages = [{"role": "user", "content": question}]
    sql = None
    for attempt in range(1, max_attempts + 1):
        try:
            sql = call_nim(system_prompt, messages, api_key)
        except Exception as e:
            err_detail = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
            print(f"\n[Attempt {attempt}/{max_attempts}] NIM call failed: {err_detail}")
            if attempt < max_attempts:
                continue
            else:
                log_entry["outcome"] = "nim_call_failed"
                log_entry["final_sql"] = None
                log_entry["error"] = err_detail
                return f"NIM_ERROR: {err_detail}", None, None

        print(f"\n[Attempt {attempt}/{max_attempts}] Generated SQL:\n{sql}")

        attempt_record = {"attempt": attempt, "sql": sql, "executed": False,
                           "error": None, "guardrail": None}

        if sql.startswith("CANNOT_ANSWER"):
            print(f"\n{sql}")
            attempt_record["guardrail"] = "CANNOT_ANSWER"
            log_entry["attempts"].append(attempt_record)
            log_entry["outcome"] = "cannot_answer"
            log_entry["final_sql"] = sql
            return sql, None, None

        if is_destructive_sql(sql):
            print(f"\n{REFUSAL_MESSAGE}")
            attempt_record["guardrail"] = "destructive_sql_blocked"
            log_entry["attempts"].append(attempt_record)
            log_entry["outcome"] = "refused_destructive"
            log_entry["final_sql"] = sql
            return sql, None, None

        try:
            columns, rows = run_sql(sql, conn)
            attempt_record["executed"] = True
            log_entry["attempts"].append(attempt_record)
            log_entry["outcome"] = "success"
            log_entry["final_sql"] = sql
            return sql, columns, rows
        except sqlite3.Error as e:
            err = str(e)
            attempt_record["error"] = err
            log_entry["attempts"].append(attempt_record)
            print(f"\nSQL execution failed: {err}")
            if attempt < max_attempts:
                fix_msg = (
                    f"The following SQL query failed to execute with this error:\n\n"
                    f"SQL:\n{sql}\n\nError:\n{err}\n\n"
                    f"Please return a corrected SQL query that resolves the error. "
                    f"Output only the raw SQL, with no explanation or markdown."
                )
                messages.append({"role": "assistant", "content": sql})
                messages.append({"role": "user", "content": fix_msg})
            else:
                log_entry["outcome"] = "sql_failed_after_retries"
                log_entry["final_sql"] = sql
                log_entry["error"] = err
                return sql, None, None
        except Exception as e:
            err_detail = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
            print(f"\nUnexpected error during SQL execution: {err_detail}")
            attempt_record["error"] = err_detail
            log_entry["attempts"].append(attempt_record)
            if attempt < max_attempts:
                continue
            else:
                log_entry["outcome"] = "execution_error"
                log_entry["final_sql"] = sql
                log_entry["error"] = err_detail
                return sql, None, None
    return sql, None, None


def summarize_result(question, columns, rows, api_key):
    data_preview = " | ".join(columns) + "\n" + "\n".join(
        " | ".join(str(v) for v in row) for row in rows[:25]
    )
    if len(rows) > 25:
        data_preview += f"\n... ({len(rows) - 25} more rows)"
    prompt = (
        f"A user asked the following question:\n\n{question}\n\n"
        f"The SQL query returned this result:\n\n{data_preview}\n\n"
        f"Write a 1-2 sentence plain-language summary of the answer for the user. "
        f"Do not mention SQL. Do not use markdown. Just the summary."
    )
    return call_nim(None, [{"role": "user", "content": prompt}], api_key)


def format_table(columns, rows):
    str_rows = [[str(v) for v in row] for row in rows]
    widths = [len(str(c)) for c in columns]
    for row in str_rows:
        for i, val in enumerate(row):
            if i < len(widths):
                widths[i] = max(widths[i], len(val))
    sep = "+".join("-" * (w + 2) for w in widths)
    sep = f"+{sep}+"
    lines = [sep]
    lines.append("| " + " | ".join(str(c).ljust(widths[i]) for i, c in enumerate(columns)) + " |")
    lines.append(sep)
    for row in str_rows:
        lines.append("| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(row))) + " |")
    lines.append(sep)
    return "\n".join(lines)


def is_numeric(value):
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def _find_label_column(columns, rows):
    str_idx = None
    for col_idx in range(len(columns)):
        non_numeric = sum(
            1 for row in rows
            if col_idx < len(row) and not is_numeric(row[col_idx])
        )
        if non_numeric >= len(rows) * 0.7:
            str_idx = col_idx
            break
    return str_idx


def maybe_generate_chart(columns, rows, question, out_path, theme_name=None, chart_type="auto", custom_colors=None):
    if len(rows) < 2 or len(columns) < 2:
        return None

    # Find label column (string)
    label_idx = _find_label_column(columns, rows)
    
    # Count numeric columns
    numeric_cols = []
    for col_idx in range(len(columns)):
        if col_idx == label_idx:
            continue
        if all(is_numeric(rows[r][col_idx]) for r in range(len(rows)) if col_idx < len(rows[r])):
            numeric_cols.append(col_idx)

    # Determine chart type if auto
    actual_chart_type = chart_type
    if chart_type == "auto":
        # Prefer a category chart when there's a label plus a numeric measure:
        # a pie for a small number of slices, otherwise a bar. Scatter is only
        # auto-picked when there is NO label column (a genuine number-vs-number
        # relationship) -- otherwise a result that also carries an id column
        # alongside the measure gets misread as 2 numerics and drawn as a scatter.
        if label_idx is not None and len(numeric_cols) >= 1:
            if len(numeric_cols) == 1 and len(rows) <= 8:
                actual_chart_type = "pie"
            else:
                actual_chart_type = "bar"
        elif len(numeric_cols) >= 2:
            actual_chart_type = "scatter"
        else:
            actual_chart_type = "bar"
    
    # Validate data supports requested chart type
    if actual_chart_type == "pie":
        if label_idx is None or len(numeric_cols) != 1 or len(rows) > 8:
            return None
        num_col = numeric_cols[0]
        labels = [str(rows[r][label_idx]) for r in range(len(rows))]
        values = [float(rows[r][num_col]) for r in range(len(rows))]
    elif actual_chart_type == "scatter":
        if len(numeric_cols) < 2:
            return None
        # Use first two numeric columns
        x_col = numeric_cols[0]
        y_col = numeric_cols[1]
        x_values = [float(rows[r][x_col]) for r in range(len(rows))]
        y_values = [float(rows[r][y_col]) for r in range(len(rows))]
        labels = [str(r) for r in range(len(rows))]
        values = list(zip(x_values, y_values))
    else:  # bar
        if label_idx is None or len(numeric_cols) == 0:
            return None
        num_col = numeric_cols[-1]
        labels = [str(rows[r][label_idx]) for r in range(len(rows))]
        values = [float(rows[r][num_col]) for r in range(len(rows))]
        if len(labels) > 20:
            return None

    theme_name = (theme_name or DEFAULT_CHART_THEME).lower()
    theme = CHART_THEMES.get(theme_name, CHART_THEMES[DEFAULT_CHART_THEME])
    colors = theme["colors"]
    if custom_colors:
        colors = custom_colors
    bg_color = theme["background"]

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.6), 5))
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    if actual_chart_type == "pie":
        pie_colors = [colors[i % len(colors)] for i in range(len(labels))]
        wedges, texts, autotexts = ax.pie(values, labels=labels, autopct='%1.1f%%', colors=pie_colors, startangle=90)
        for autotext in autotexts:
            autotext.set_fontsize(8)
        ax.set_title(question[:100], wrap=True)
    elif actual_chart_type == "scatter":
        x_vals = [v[0] for v in values]
        y_vals = [v[1] for v in values]
        scatter_colors = [colors[i % len(colors)] for i in range(len(labels))]
        ax.scatter(x_vals, y_vals, c=scatter_colors, edgecolor="black", linewidth=0.5, alpha=0.7, s=80)
        ax.set_xlabel(columns[numeric_cols[0]])
        ax.set_ylabel(columns[numeric_cols[1]])
        ax.set_title(question[:100], wrap=True)
        ax.grid(axis="both", alpha=0.3, linestyle="--")
    else:  # bar
        x = range(len(labels))
        bar_colors = [colors[i % len(colors)] for i in range(len(labels))]
        bars = ax.bar(x, values, color=bar_colors, edgecolor="black", linewidth=0.5)
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel(columns[num_col])
        ax.set_title(question[:100], wrap=True)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:,.1f}",
                    ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120, facecolor=bg_color)
    plt.close(fig)
    return out_path


def main(file_path=None):
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        sys.exit("NVIDIA_API_KEY environment variable is not set.")
    if file_path is None:
        file_path = DEFAULT_DATA_FILE

    source_file = os.path.basename(file_path)
    log_entry = new_log_entry(TEST_QUESTION, source_file)

    try:
        conn, schema, db_path = load_dataset(file_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"\nCould not load dataset: {e}")
        log_entry["outcome"] = "load_failed"
        log_entry["error"] = str(e)
        append_log(log_entry)
        print(f"\nLogged to {LOG_PATH}")
        return
    print(f"Loaded dataset: {file_path}")
    print(f"Detected tables: {list(schema.keys())}")
    for t, cols in schema.items():
        print(f"  {t}({', '.join(c[0] for c in cols)})")

    print("\nRunning data quality validation...")
    validation = validate_dataset(conn, schema)
    log_entry["data_quality_summary"] = validation
    report = format_validation_report(validation)
    print(report)

    use_cleaned = False
    correction_log = None
    cleaned_conn = None
    cleaned_path = None

    has_issues = any(
        any([
            checks.get("missing_values"),
            checks.get("duplicate_rows", 0) > 0,
            checks.get("inconsistent_categories"),
            checks.get("type_mismatches")
        ])
        for checks in validation.values()
    )

    if has_issues:
        try:
            choice = input("\nAuto-correct these issues? (y/n): ").strip().lower()
            use_cleaned = choice == "y"
        except (EOFError, KeyboardInterrupt):
            use_cleaned = False

    if use_cleaned:
        print("\nApplying auto-corrections...")
        cleaned_conn, correction_log = auto_correct_dataset(conn, schema, validation)
        print(format_correction_report(correction_log))
        cleaned_path = save_cleaned_dataset(cleaned_conn, file_path)
        print(f"\nCleaned dataset saved to: {cleaned_path}")
        conn.close()
        conn = cleaned_conn
        schema = infer_schema(conn)
        log_entry["correction_mode"] = "cleaned"
        log_entry["correction_log"] = correction_log
        log_entry["cleaned_file"] = cleaned_path
    else:
        log_entry["correction_mode"] = "original"
        if correction_log:
            log_entry["correction_log"] = correction_log

    system_prompt = build_dynamic_system_prompt(schema)
    print(f"\nBuilt dynamic system prompt ({len(system_prompt)} chars).")

    print(f"\nSending question to NIM: {TEST_QUESTION}")
    sql, columns, rows = answer_question(system_prompt, TEST_QUESTION, api_key, conn, log_entry)

    try:
        if columns is None:
            print("\nThe question could not be answered.")
            if sql:
                print(f"\nLast response from model:\n{sql}")
            if not log_entry["outcome"]:
                log_entry["outcome"] = "no_result"
            append_log(log_entry)
            print(f"\nLogged to {LOG_PATH}")
            return

        print("\nGenerating plain-language summary ...")
        try:
            summary = summarize_result(TEST_QUESTION, columns, rows, api_key)
            log_entry["summary"] = summary
        except Exception as e:
            summary = f"(summary generation failed: {e})"
            log_entry["error"] = f"summary_generation: {e}"

        log_entry["result_columns"] = columns
        log_entry["result_rows"] = [
            [str(v) for v in row] for row in rows
        ][:100]

        chart_path = os.path.join(CHART_DIR, "chart.png")
        generated = maybe_generate_chart(columns, rows, TEST_QUESTION, chart_path)
        if generated:
            log_entry["chart_path"] = generated

        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(summary)
        print("\n" + "=" * 60)
        print("RESULT")
        print("=" * 60)
        print(format_table(columns, rows))
        if generated:
            print(f"\nChart saved to: {generated}")

        append_log(log_entry)
        print(f"\nLogged to {LOG_PATH}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
