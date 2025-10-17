"""
app.py - Simple Flask dashboard to show provider validation results and allow revalidate
Run:
    flask run --host=0.0.0.0 --port=5000
"""
try:
    from flask import Flask, render_template, redirect, url_for, flash, send_from_directory, get_flashed_messages  # type: ignore
except Exception:
    # Flask is not installed or could not be imported; provide clear runtime errors.
    class _MissingFlask:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("Flask is required to run this app. Install it with: pip install flask")

    def _missing(*args, **kwargs):
        raise RuntimeError("Flask is required to run this app. Install it with: pip install flask")

    Flask = _MissingFlask
    render_template = _missing
    redirect = _missing
    url_for = _missing
    flash = _missing
    send_from_directory = _missing
    get_flashed_messages = _missing
import csv
from html import escape
try:
    import pandas as pd  # type: ignore
except Exception:
    pd = None
from pathlib import Path
import subprocess
import sys

# Base directory for project files and default paths used by the app
BASE_DIR = Path(__file__).resolve().parent
REPORT_CSV = BASE_DIR / "validation_report.csv"   # match validator.py output
VALIDATOR_PY = BASE_DIR / "validator.py"

app = Flask(__name__, template_folder=str(BASE_DIR / "templates"))
app.secret_key = "change-me-in-prod"

@app.route("/")
def index():
    if not REPORT_CSV.exists():
        return render_template("index.html", table_html=None, missing=True, report_name=REPORT_CSV.name)

    try:
        if pd:
            df = pd.read_csv(REPORT_CSV)
            table_html = df.head(200).to_html(classes="table table-striped", index=False, escape=False)
            return render_template("index.html", table_html=table_html, missing=False, report_name=REPORT_CSV.name)
        else:
            # Lightweight CSV -> HTML table fallback when pandas is not available
            headers = []
            rows = []
            with REPORT_CSV.open(newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                headers = next(reader, [])
                for _, row in zip(range(200), reader):
                    rows.append(row)
            header_html = "".join(f"<th>{escape(h)}</th>" for h in headers)
            body_html = ""
            for row in rows:
                body_html += "<tr>" + "".join(f"<td>{escape(str(cell))}</td>" for cell in row) + "</tr>"
            table_html = f'<table class="table table-striped"><thead><tr>{header_html}</tr></thead><tbody>{body_html}</tbody></table>'
            return render_template("index.html", table_html=table_html, missing=False, report_name=REPORT_CSV.name)
    except Exception as e:
        flash(f"Failed to read report: {e}", "danger")
        return render_template("index.html", table_html=None, missing=True, report_name=REPORT_CSV.name)

@app.route("/revalidate")
def revalidate():
    if not VALIDATOR_PY.exists():
        flash(f"validator.py not found at {VALIDATOR_PY}", "danger")
        return redirect(url_for("index"))
    try:
        proc = subprocess.run(
            [sys.executable, str(VALIDATOR_PY)],
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            timeout=120,
        )
        if proc.returncode != 0:
            err = proc.stderr or proc.stdout or "Unknown error"
            flash(f"Validation failed: {err[:400]}", "danger")
        else:
            flash("Validation completed successfully.", "success")
    except Exception as e:
        flash(f"Validation run error: {e}", "danger")
    return redirect(url_for("index"))

@app.route("/download")
def download():
    if not REPORT_CSV.exists():
        flash("Report not found to download.", "warning")
        return redirect(url_for("index"))
    return send_from_directory(directory=str(BASE_DIR), path=REPORT_CSV.name, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
