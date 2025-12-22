# PFA AI Text Detector Backend (FastAPI)

Minimal FastAPI + SQLite backend that serves a locally fine-tuned DeBERTa-v3-base classifier for detecting AI-generated text. Includes JWT auth, per-segment explainability, exports, and simple stats.

## Setup
```powershell
# from repo root
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

```

## Environment
- `MODEL_DIR` (default `./model`): path to the fine-tuned Hugging Face folder. Prefer an absolute path (PowerShell: `(Resolve-Path ".\\model").Path`).
- `JWT_SECRET` (default `dev-secret-change-me`): HS256 signing secret.
- `DATABASE_URL` (default `sqlite:///./pfa.db`): SQLite connection string. For Supabase/Postgres use `postgresql+psycopg://USER:PASSWORD@HOST:PORT/DBNAME`.

## Using .env (recommended)
1) Copy `.env.example` to `.env`:
```powershell
Copy-Item .env.example .env
```
2) Edit `.env` as needed, including `DATABASE_URL` for Supabase:
   - Example (SQLAlchemy + psycopg v3):
     `DATABASE_URL=postgresql+psycopg://postgres:<PASSWORD>@db.<project-ref>.supabase.co:5432/postgres`
   - If Supabase gives `postgresql://...`, use the same credentials but replace the scheme with `postgresql+psycopg://`.
   - If a direct connection fails on IPv4-only networks, use the Supabase Session Pooler host/port.
3) Run Uvicorn without setting `$env:` variables:
```powershell
uvicorn app.main:app --reload --port 8000
```

## Model files
Copy your fine-tuned DeBERTa-v3-base folder (containing `config.json`, `model.safetensors` or `pytorch_model.bin`, tokenizer files, etc.) into `./model/` so that `AutoTokenizer.from_pretrained` and `AutoModelForSequenceClassification.from_pretrained` can load it locally.

## Model behavior and threshold
- This model is trained on SemEval 2024 Task 8, Subtask A (monolingual). It may not generalize to all writing styles (e.g., Wikipedia, NASA press releases, or public domain novels).
- `AI_THRESHOLD` controls the trade-off between false positives (labeling human as AI) and false negatives (missing AI text).
- Label meaning is fixed: `0 = Human`, `1 = AI-generated`.

## Run
```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

## Example curl calls
```bash
# 1) register
curl -X POST http://localhost:8000/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email":"demo@example.com","password":"pass1234"}'

# 2) login (store token; bash example)
TOKEN=$(curl -s -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"demo@example.com","password":"pass1234"}' | jq -r '.access_token')

# 3) analyze with bearer token
curl -X POST http://localhost:8000/analyze \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"text":"This is a sample paragraph to detect whether it looks AI generated."}'
```
PowerShell token capture (alternative):
```powershell
$resp = curl -Method POST http://localhost:8000/auth/login -ContentType "application/json" -Body '{"email":"demo@example.com","password":"pass1234"}'
$token = ($resp.Content | ConvertFrom-Json).access_token
```

## Troubleshooting
- `ImportError: email-validator is not installed` -> `pip install email-validator` (already included in `requirements.txt`).
- Torch install fails on Python 3.14 -> ensure you are on `torch>=2.9.0,<2.10` which ships matching wheels for this Python version.
- If pip fails building `psycopg2-binary` or complains about `pg_config`, use psycopg v3 (`psycopg[binary]`) as in `requirements.txt`.
- Bcrypt is not used; `pbkdf2_sha256` is chosen for Windows portability and no 72-byte password limit.
