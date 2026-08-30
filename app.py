import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from matplotlib.colors import is_color_like

import agent
from agent import is_numeric
from pydantic import BaseModel

app = FastAPI(title="SQL Agent API", description="Natural language to SQL query agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path(agent.BASE_DIR) / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

CHART_DIR = Path(agent.CHART_DIR)
CHART_DIR.mkdir(exist_ok=True)

file_store: Dict[str, Dict[str, Any]] = {}


class UploadResponse(BaseModel):
    file_id: str
    schema: Dict[str, List[List[str]]]
    validation_report: Dict[str, Any]


class CleanRequest(BaseModel):
    file_id: str


class CleanResponse(BaseModel):
    cleaned_file_path: str
    correction_report: Dict[str, Any]


class AskRequest(BaseModel):
    file_id: str
    question: str
    use_cleaned: bool = False
    chart_theme: Optional[str] = "corporate"
    chart_type: Optional[str] = "auto"
    custom_colors: Optional[List[str]] = None


class AskResponse(BaseModel):
    summary: str
    table: List[Dict[str, Any]]
    chart_url: Optional[str]
    sql_used: str
    outcome: str
    error: Optional[str] = None
    chart_theme_used: Optional[str] = None
    chart_type_used: Optional[str] = None
    custom_colors_used: Optional[List[str]] = None
    warning: Optional[str] = None


def _get_api_key() -> str:
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="NVIDIA_API_KEY environment variable not set")
    return api_key


def _schema_to_serializable(schema_dict: Dict[str, List[tuple]]) -> Dict[str, List[List[str]]]:
    return {table: [[col, typ] for col, typ in cols] for table, cols in schema_dict.items()}


def _resolve_chart_theme(chart_theme: Optional[str]) -> tuple[str, Optional[str]]:
    """Validate chart_theme against known themes. Returns (validated_theme, warning_or_None)."""
    theme_lower = (chart_theme or agent.DEFAULT_CHART_THEME).lower()
    valid_themes = set(agent.CHART_THEMES.keys())
    if theme_lower not in valid_themes:
        return agent.DEFAULT_CHART_THEME, f"Unknown theme '{chart_theme}', defaulted to corporate"
    return theme_lower, None


def _resolve_custom_colors(custom_colors: Optional[List[str]]) -> tuple[Optional[List[str]], Optional[str]]:
    """Validate custom colors. Returns (validated_list_or_None, warning_or_None)."""
    if not custom_colors:
        return None, None
    invalid = [c for c in custom_colors if not is_color_like(c)]
    if invalid:
        return None, f"Invalid custom color(s) {invalid}; using theme colors instead"
    return custom_colors, None


def _resolve_chart_type(chart_type: Optional[str], columns, rows) -> tuple[str, Optional[str]]:
    """Validate and auto-detect chart type. Returns (validated_type, warning_or_None)."""
    if chart_type is None:
        chart_type = "auto"
    chart_type = chart_type.lower()
    valid_types = {"auto", "bar", "pie", "scatter"}
    if chart_type not in valid_types:
        return "auto", f"Unknown chart type '{chart_type}', defaulted to auto"
    
    # Guard: no data or insufficient structure for any chart
    if rows is None or columns is None or len(rows) == 0 or len(columns) == 0:
        return "auto", None  # Match maybe_generate_chart's None return convention
    
    # Auto-detect chart type based on data structure
    if chart_type == "auto":
        if len(rows) < 2 or len(columns) < 2:
            return "auto", None  # Not enough data for any chart
        
        # Find label column (string)
        label_idx = None
        for col_idx in range(len(columns)):
            non_numeric = sum(
                1 for row in rows
                if col_idx < len(row) and not is_numeric(row[col_idx])
            )
            if non_numeric >= len(rows) * 0.7:
                label_idx = col_idx
                break
        
        # Count numeric columns
        numeric_cols = []
        for col_idx in range(len(columns)):
            if col_idx == label_idx:
                continue
            if all(is_numeric(rows[r][col_idx]) for r in range(len(rows)) if col_idx < len(rows[r])):
                numeric_cols.append(col_idx)
        
        # Prefer a category chart when there's a label plus a numeric measure:
        # a pie for a small number of slices, otherwise a bar. Scatter is only
        # auto-picked when there is NO label column (a genuine number-vs-number
        # relationship) -- otherwise a result that also carries an id column
        # alongside the measure gets misread as 2 numerics and drawn as a scatter.
        if label_idx is not None and len(numeric_cols) >= 1:
            if len(numeric_cols) == 1 and len(rows) <= 8:
                return "pie", None
            return "bar", None
        
        # No label column: a genuine number-vs-number relationship
        if len(numeric_cols) >= 2:
            return "scatter", None
        
        # Default to bar
        return "bar", None
    
    # Forced chart type - validate data supports it
    if chart_type == "pie":
        # Need one string label + one numeric, <= 8 labels
        label_idx = None
        for col_idx in range(len(columns)):
            non_numeric = sum(
                1 for row in rows
                if col_idx < len(row) and not is_numeric(row[col_idx])
            )
            if non_numeric >= len(rows) * 0.7:
                label_idx = col_idx
                break
        numeric_cols = []
        for col_idx in range(len(columns)):
            if col_idx == label_idx:
                continue
            if all(is_numeric(rows[r][col_idx]) for r in range(len(rows)) if col_idx < len(rows[r])):
                numeric_cols.append(col_idx)
        if label_idx is None or len(numeric_cols) != 1 or len(rows) > 8:
            resolved, _ = _resolve_chart_type("auto", columns, rows)
            return resolved, "Pie needs 1 label + 1 numeric column with 8 or fewer slices; showing the best-fit chart instead."
        return "pie", None
    
    if chart_type == "scatter":
        # Need at least 2 numeric columns
        numeric_cols = []
        for col_idx in range(len(columns)):
            if all(is_numeric(rows[r][col_idx]) for r in range(len(rows)) if col_idx < len(rows[r])):
                numeric_cols.append(col_idx)
        if len(numeric_cols) < 2:
            resolved, _ = _resolve_chart_type("auto", columns, rows)
            return resolved, "Scatter needs at least 2 numeric columns; showing the best-fit chart instead."
        return "scatter", None
    
    # bar chart - default fallback
    return "bar", None


def _run_question_answering(
    file_id: str,
    question: str,
    use_cleaned: bool,
    api_key: str,
    chart_theme: Optional[str] = "corporate",
    chart_type: Optional[str] = "auto",
    custom_colors: Optional[List[str]] = None
) -> Dict[str, Any]:
    file_info = file_store.get(file_id)
    if not file_info:
        raise HTTPException(status_code=404, detail=f"File ID {file_id} not found")

    if use_cleaned:
        conn = file_info.get("cleaned_conn")
        source_file = file_info.get("cleaned_file", file_info["original_file"])
    else:
        conn = file_info["original_conn"]
        source_file = file_info["original_file"]

    if conn is None:
        raise HTTPException(status_code=400, detail="Connection not available for this file")

    schema = agent.infer_schema(conn)
    system_prompt = agent.build_dynamic_system_prompt(schema)

    log_entry = agent.new_log_entry(question, source_file)
    log_entry["correction_mode"] = "cleaned" if use_cleaned else "original"

    sql, columns, rows = agent.answer_question(
        system_prompt, question, api_key, conn, log_entry
    )

    if columns is None:
        outcome = log_entry.get("outcome", "no_result")
        if not outcome or outcome == "no_result":
            outcome = "failed"
        error_detail = log_entry.get("error")
        theme_used, theme_warning = _resolve_chart_theme(chart_theme)
        type_used, type_warning = _resolve_chart_type(chart_type, columns, rows)
        colors_used, color_warning = _resolve_custom_colors(custom_colors)
        combined_warning = None
        if theme_warning and type_warning:
            combined_warning = f"{theme_warning}; {type_warning}"
        elif theme_warning:
            combined_warning = theme_warning
        elif type_warning:
            combined_warning = type_warning
        if color_warning:
            combined_warning = f"{combined_warning}; {color_warning}" if combined_warning else color_warning
        return {
            "summary": "",
            "table": [],
            "chart_url": None,
            "sql_used": sql or "",
            "outcome": outcome,
            "error": error_detail,
            "chart_theme_used": theme_used,
            "chart_type_used": type_used,
            "custom_colors_used": colors_used,
            "warning": combined_warning,
        }

    try:
        summary = agent.summarize_result(question, columns, rows, api_key)
    except Exception as e:
        summary = f"(summary generation failed: {e})"

    chart_filename = f"chart_{file_id}_{uuid.uuid4().hex[:8]}.png"
    chart_path = CHART_DIR / chart_filename

    theme_lower, theme_warning = _resolve_chart_theme(chart_theme)
    type_used, type_warning = _resolve_chart_type(chart_type, columns, rows)
    colors_used, color_warning = _resolve_custom_colors(custom_colors)

    generated = agent.maybe_generate_chart(columns, rows, question, str(chart_path), theme_lower, type_used, colors_used)
    chart_url = f"/charts/{chart_filename}" if generated else None

    table_data = [dict(zip(columns, row)) for row in rows]

    log_entry["summary"] = summary
    log_entry["result_columns"] = columns
    log_entry["result_rows"] = [[str(v) for v in row] for row in rows][:100]
    log_entry["chart_path"] = str(chart_path) if generated else None
    agent.append_log(log_entry)

    combined_warning = None
    if theme_warning and type_warning:
        combined_warning = f"{theme_warning}; {type_warning}"
    elif theme_warning:
        combined_warning = theme_warning
    elif type_warning:
        combined_warning = type_warning
    if color_warning:
        combined_warning = f"{combined_warning}; {color_warning}" if combined_warning else color_warning

    return {
        "summary": summary,
        "table": table_data,
        "chart_url": chart_url,
        "sql_used": sql,
        "outcome": "success",
        "chart_theme_used": theme_lower,
        "chart_type_used": type_used,
        "custom_colors_used": colors_used,
        "warning": combined_warning
    }


@app.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix.lower()
    if ext not in (".csv", ".xlsx", ".xls", ".db"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Supported: .csv, .xlsx, .xls, .db"
        )

    file_id = uuid.uuid4().hex
    file_path = UPLOAD_DIR / f"{file_id}{ext}"

    try:
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {e}")

    try:
        conn, schema, _ = agent.load_dataset(str(file_path))
        validation = agent.validate_dataset(conn, schema)
    except (FileNotFoundError, ValueError) as e:
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Failed to load dataset: {e}")
    except Exception as e:
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")

    file_store[file_id] = {
        "original_file": str(file_path),
        "original_conn": conn,
        "original_filename": file.filename,
        "schema": schema,
        "validation": validation,
    }

    return UploadResponse(
        file_id=file_id,
        schema=_schema_to_serializable(schema),
        validation_report=validation
    )


@app.post("/clean", response_model=CleanResponse)
async def clean_file(request: CleanRequest):
    file_id = request.file_id
    file_info = file_store.get(file_id)
    if not file_info:
        raise HTTPException(status_code=404, detail=f"File ID {file_id} not found")

    conn = file_info["original_conn"]
    schema = file_info["schema"]
    validation = file_info["validation"]

    try:
        cleaned_conn, correction_log = agent.auto_correct_dataset(conn, schema, validation)
        cleaned_path = agent.save_cleaned_dataset(cleaned_conn, file_info["original_file"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clean dataset: {e}")

    file_store[file_id]["cleaned_conn"] = cleaned_conn
    file_store[file_id]["cleaned_file"] = cleaned_path
    file_store[file_id]["correction_log"] = correction_log

    return CleanResponse(
        cleaned_file_path=cleaned_path,
        correction_report=correction_log
    )


@app.get("/download/{file_id}")
async def download_cleaned(file_id: str):
    """Download the cleaned copy of an uploaded dataset (created by /clean)."""
    file_info = file_store.get(file_id)
    if not file_info:
        raise HTTPException(status_code=404, detail=f"File ID {file_id} not found")

    cleaned_path = file_info.get("cleaned_file")
    if not cleaned_path or not os.path.exists(cleaned_path):
        raise HTTPException(
            status_code=404,
            detail="No cleaned file yet. Click 'Clean my data' first, then download.",
        )

    original_name = file_info.get("original_filename") or os.path.basename(cleaned_path)
    download_name = "cleaned_" + original_name
    ext = os.path.splitext(cleaned_path)[1].lower()
    media_types = {
        ".csv": "text/csv",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".xls": "application/vnd.ms-excel",
        ".db": "application/octet-stream",
    }
    media_type = media_types.get(ext, "application/octet-stream")
    return FileResponse(cleaned_path, media_type=media_type, filename=download_name)


@app.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    api_key = _get_api_key()
    result = _run_question_answering(request.file_id, request.question, request.use_cleaned, api_key, request.chart_theme, request.chart_type, request.custom_colors)
    return AskResponse(**result)


@app.get("/charts/{filename}")
async def get_chart(filename: str):
    chart_path = CHART_DIR / filename
    if not chart_path.exists():
        raise HTTPException(status_code=404, detail="Chart not found")
    return FileResponse(chart_path, media_type="image/png")


@app.get("/logs")
async def get_logs():
    try:
        logs = agent.load_log()
        return logs
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load logs: {e}")


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.get("/")
async def serve_ui():
    ui_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "index.html")
    if not os.path.exists(ui_path):
        raise HTTPException(status_code=404, detail="UI not found. Expected static/index.html next to app.py.")
    return FileResponse(ui_path, media_type="text/html")


def main():
    import uvicorn
    print("=" * 60)
    print("SQL Agent API Server")
    print("=" * 60)
    print(f"Starting server at http://localhost:8000")
    print(f"Using NIM model: {agent.MODEL}")
    print(f"API docs at http://localhost:8000/docs")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()