
NkCodeX - Provider Data Validation and Directory Management Agent
================================================================

Contents:
- providers_input.csv : Synthetic provider dataset (200 records)
- reference_registry.csv : Simulated public registry / NPI-like data
- validator.py : Main validation pipeline (fuzzy matching + ML confidence scoring)
- validation_report.csv : Output report (generated after running validator.py)
- app.py : Simple Flask dashboard to view the report and re-run validation
- requirements.txt : Python dependencies
- README.md : This file

How to run (locally):
1. Create a virtual environment and install dependencies:
   python -m venv venv
   source venv/bin/activate   # or venv\Scripts\activate on Windows
   pip install -r requirements.txt

2. Run the validator to generate report:
   python validator.py

3. Optional: Run Flask dashboard:
   flask run
   Open http://127.0.0.1:5000

Notes:
- This is a simulation. To make it production-grade:
  * Replace the lookup_reference with real API calls (NPI, State license boards, Google Maps)
  * Add rate-limiting, PII redaction, and audit logging
  * Implement UEBA monitoring for agent behavior
  * Integrate with a database (PostgreSQL) and a task queue (Celery) for scale
