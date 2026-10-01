# Deploying TerraGuard — Cloud Run + Vercel + Supabase

Stack: FastAPI backend containerized on **Google Cloud Run**, Next.js frontend on **Vercel**,
Postgres (jobs + LangGraph checkpoints) on **Supabase**, all on free tiers.

Everything below that touches code is already done in this branch (new `Dockerfile`,
`.dockerignore`, `db/job_store.py` and `workflows/agent_workflow_hitl.py` now support
`DATABASE_URL`, CORS reads `FRONTEND_ORIGINS`). The steps below are the account-bound
actions only you can do — I can't create cloud accounts or run authenticated CLI commands
on your behalf.

Order matters: Supabase first (backend needs `DATABASE_URL` at deploy time), then Cloud Run
(frontend needs the Cloud Run URL), then Vercel, then one final loop back to Cloud Run to
register the Vercel URL for CORS.

---

## 1. Supabase (database)

1. Create a free project at supabase.com.
2. In the dashboard: **Project Settings → Database → Connection string**. Copy the
   **Transaction pooler** string (port `6543`), not the direct connection (port `5432`) —
   the app opens a short-lived connection per request, and the pooler avoids exhausting
   Supabase's free-tier connection limit under that pattern.
3. Save it somewhere — you'll paste it as `DATABASE_URL` in step 2 below. It looks like:
   ```
   postgresql://postgres.<ref>:<password>@aws-0-<region>.pooler.supabase.com:6543/postgres
   ```

No manual schema setup needed — `db/job_store.py` and the LangGraph `PostgresSaver` both
create their own tables on first run (`init_db()` / `memory.setup()`).

---

## 2. Google Cloud Run (backend)

**Prerequisites**: a GCP project with billing enabled (Cloud Run's free tier still requires
a billing account attached, you just won't be charged within the free quota), and the
`gcloud` CLI installed and authenticated (`gcloud auth login`).

Reuse the project already hardcoded as the Vertex AI project (`sre-agent-project-505914`,
or your own project — if you use a different one, update `GOOGLE_CLOUD_PROJECT` below and
in `workflows/agent_workflow_hitl.py`'s default). Deploying in the *same* project as your
Vertex AI project means the Cloud Run service's own identity can call Gemini with zero
extra credentials — see step 2d.

```bash
# 2a. Set your project and enable the APIs you need
gcloud config set project sre-agent-project-505914
gcloud services enable run.googleapis.com artifactregistry.googleapis.com \
  secretmanager.googleapis.com aiplatform.googleapis.com

# 2b. Put secrets in Secret Manager (never pass these as plain --set-env-vars)
echo -n "postgresql://postgres.<ref>:<password>@aws-0-<region>.pooler.supabase.com:6543/postgres" | \
  gcloud secrets create database-url --data-file=-

echo -n "<your IAM user access key>" | gcloud secrets create aws-access-key-id --data-file=-
echo -n "<your IAM user secret key>" | gcloud secrets create aws-secret-access-key --data-file=-
echo -n "<your TerraForgeRole ARN>" | gcloud secrets create aws-role-arn --data-file=-
echo -n "<your external id>" | gcloud secrets create aws-external-id --data-file=-

# 2c. Deploy directly from this local source tree (NOT via a GitHub-connected
# trigger). chroma_db_terraform/ is in .gitignore — building from local source
# is what guarantees the baked-in vector index actually makes it into the image.
gcloud run deploy terraguard-backend \
  --source . \
  --region us-central1 \
  --memory 2Gi \
  --cpu 2 \
  --timeout 3600 \
  --allow-unauthenticated \
  --set-env-vars GOOGLE_CLOUD_PROJECT=sre-agent-project-505914,GOOGLE_CLOUD_LOCATION=us-central1 \
  --set-secrets DATABASE_URL=database-url:latest,AWS_ACCESS_KEY_ID=aws-access-key-id:latest,AWS_SECRET_ACCESS_KEY=aws-secret-access-key:latest,AWS_ROLE_ARN=aws-role-arn:latest,AWS_EXTERNAL_ID=aws-external-id:latest
```

This builds the Dockerfile via Cloud Build automatically (no separate `cloudbuild.yaml`
needed) and deploys it. Note the **Service URL** it prints at the end — you'll need it for
Vercel next.

```bash
# 2d. Let Cloud Run's own identity call Vertex AI, no credentials file needed
PROJECT_NUMBER=$(gcloud projects describe sre-agent-project-505914 --format='value(projectNumber)')
gcloud projects add-iam-policy-binding sre-agent-project-505914 \
  --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
  --role="roles/aiplatform.user"
```

**Known cold-start cost**: Cloud Run's free tier scales to zero. A cold start reloads the
embedding model + CrossEncoder (the singleton-load cost this project's own optimization
work eliminated *within* a run comes back on the *first* request after an idle period).
Measured locally in this exact image: **~46 seconds** before the server accepts its first
request — slower than the ~20s figure from the original optimization report, likely a
slower CPU/IO environment than that was measured on; verify it on the real Cloud Run
instance once deployed rather than assuming either number. The Dockerfile already bakes in
the Chroma index and the Terraform provider plugin to remove the other two cold-start
costs. If demo-day latency matters more than free-tier cost, add `--min-instances 1` to
the deploy command above — keeps one instance warm, eliminates cold starts, costs a small
amount beyond the free tier (not free, but
cheap — a few dollars/month).

---

## 3. Vercel (frontend)

1. Push this repo to GitHub if it isn't already, then import it at vercel.com → **Add New
   Project**.
2. Set **Root Directory** to `frontend`.
3. Add an environment variable: `NEXT_PUBLIC_API_URL` = the Cloud Run Service URL from step
   2c (e.g. `https://terraguard-backend-xxxxx-uc.a.run.app`).
4. Deploy. Vercel auto-detects Next.js — no build config changes needed.
5. Note the resulting `*.vercel.app` URL.

---

## 4. Close the loop: CORS

Now that you have the Vercel URL, let the backend accept requests from it:

```bash
gcloud run services update terraguard-backend \
  --region us-central1 \
  --update-env-vars FRONTEND_ORIGINS=https://your-app.vercel.app
```

---

## Known limitations worth knowing about before your demo

- **Settings-page "Save Config" doesn't persist across redeploys.** `aws/credentials_manager.py`
  writes the Role ARN / External ID to a local `.env` file, which works for local dev but
  is a non-durable write on Cloud Run (each instance's filesystem is ephemeral, and
  multiple instances don't share it). Since `AWS_ROLE_ARN` / `AWS_EXTERNAL_ID` are set as
  deploy-time secrets in step 2b, the Settings page will *show* them correctly and the
  "Test Connection" flow works fine — just don't rely on editing and saving a *different*
  Role ARN there and expecting it to survive a cold start or redeploy. If you need that to
  actually work, it would need to move from a `.env` write into the Postgres `control`
  table the same way `is_apply_paused()`/`set_apply_paused()` already do — that's a
  follow-up, not done here.
- **`advanced` and `secure` workflow tiers still use local SQLite state.** Only the `hitl`
  workflow (the one the actual UI drives) was migrated to Postgres — `advanced` and
  `secure` are only reachable via direct API calls used by the benchmarking scripts, never
  by the frontend, so they were left as-is to keep this change scoped. They would error if
  invoked against the deployed container; harmless unless something starts calling them.
- **AWS static keys, not Workload Identity Federation.** The backend still authenticates to
  AWS STS using a long-lived IAM user access key (stored in Secret Manager, not in code —
  reasonably safe, but still a static key). Full Workload Identity Federation (letting
  Cloud Run's own identity assume the AWS role with zero stored AWS keys at all) is the
  logical next step and lines up with the "zero long-lived secrets" story in your pitch,
  but it's a bigger lift (AWS OIDC identity provider setup) that wasn't done here — say the
  word if you want it.
