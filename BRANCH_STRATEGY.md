# Branch Strategy

## `main` — Real AWS (Production-Ready)

Start with **no** `MOCK_AWS` flag (or `MOCK_AWS=false`):

```bash
# .env  (gitignored — add your real values)
AWS_ROLE_ARN=arn:aws:iam::<account-id>:role/TerraForgeRole
AWS_EXTERNAL_ID=bc14569d-9676-4417-90c3-469e0a9310dc

# Start backend
PYTHONPATH=. venv/bin/uvicorn api.server:app --reload --port 8000

# Start frontend
cd frontend && npm run dev
```

**Behaviour on `main`:**
- `plan_node` runs a real `terraform plan` against AWS using STS AssumeRole credentials
- Blast Radius Guard runs on the actual plan JSON — real ownership + IAM wildcard checks
- `apply_node` does a real `terraform apply` — infra is actually created
- Requires a valid Role ARN set in Settings → AWS Credentials

---

## `feature/apply-pipeline-v2` — Mock AWS (Dev / Demo)

`.env` on this branch already has `MOCK_AWS=true` baked in:

```bash
# Start backend (no extra flags needed — .env sets MOCK_AWS=true)
PYTHONPATH=. venv/bin/uvicorn api.server:app --reload --port 8000

# Start frontend
cd frontend && npm run dev
```

**Behaviour on feature branch:**
- `plan_node` simulates `terraform plan` by parsing resource blocks from generated HCL — no AWS call
- `apply_node` returns a mock `applied` response immediately
- No credentials required — safe for demos, CI, and development
- Trust scores, Blast Radius Guard, and cost estimates all use the simulated plan

---

## How to Switch Modes

| Want to... | Do this |
|---|---|
| Test real AWS infra creation | Use `main` branch, set Role ARN in Settings |
| Demo / develop safely | Use `feature/apply-pipeline-v2`, `MOCK_AWS=true` in `.env` |
| Toggle quickly on `main` | Set `MOCK_AWS=true` in `.env` to flip to mock mode temporarily |

> **Note:** `.env` is gitignored — never commit real credentials.
