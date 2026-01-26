# Gemini Grounding Proxy

This service packages the code in `aieng-agents-utils/aieng/agents/web_search` into a FastAPI
application. It plays a dual role in the Agent Bootcamp project:

- **Agent tooling showcase.** The proxy demonstrates how you can wrap a third-party
  capability, like Google’s Gemini Grounding with Google Search behind your own
  endpoints so internal AI agents can call it like any other tool.
- **Cost and quota guardrail.** Some APIs can be expensive. By forcing every
  request through a proxy like this and authenticating with Firestore-backed API
  keys, you can cap usage, suspend abusers, and revoke keys without touching the
  third-party API directly.

The instructions below cover local development with the Firestore emulator and
production deployment on Google Cloud Run.

---

## 1. Prerequisites

- Python >=3.12
- `gcloud` CLI with an authenticated account
- JDK 21+ (Firestore emulator requirement)
- Access to a Google Cloud project with billing enabled

Recommended:

- `uv` or `pip` for dependency management
- Ability to set environment variables from `.env` files

Authenticate once before continuing:

```bash
export REGION=us-central1
gcloud init
gcloud auth application-default login
gcloud auth configure-docker "$REGION-docker.pkg.dev"
```

---

## 2. Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `FIRESTORE_PROJECT_ID` | Project hosting the Firestore database | _(required)_ |
| `FIRESTORE_COLLECTION` | Collection that stores API key records | `apiKeys` |
| `FIRESTORE_DATABASE_NAME` | Optional named database (non-default) | `grounding` |
| `FIRESTORE_EMULATOR_HOST` | Host:port for the emulator (dev only) | _(unset)_ |
| `GEMINI_API_KEY` | Gemini API key used by the proxy | _(required)_ |
| `GEMINI_MAX_ATTEMPTS`, `GEMINI_MAX_BACKOFF_SECONDS` | Retry tuning | `5`, `10` |
| `API_KEY_CACHE_TTL`, `API_KEY_CACHE_MAX_ITEMS` | Auth cache tuning | `30`, `1024` |
| `DAILY_USAGE_COLLECTION` | Collection that stores per-day usage counters | `dailyUsageCounters` |
| `DAILY_USAGE_MAX_RETRIES`, `DAILY_USAGE_BASE_DELAY`, `DAILY_USAGE_MAX_DELAY` | Daily usage retry tuning | `8`, `0.05`, `1.0` |
| `GEMINI_GROUNDING_FREE_LIMIT_PRO` | Daily free allowance for `gemini-2.5-pro` | `1500` |
| `GEMINI_GROUNDING_FREE_LIMIT_FLASH` | Shared daily free allowance for Flash/Flash-Lite | `1500` |

Keep `.env.example` up to date so teammates can copy it into their own `.env`.

---

## 3. Local Development (Firestore Emulator)

1. **Install gcloud components**

   ```bash
   gcloud components install beta
   gcloud components install cloud-firestore-emulator
   ```

2. **Start the emulator**

   ```bash
   gcloud beta emulators firestore start \
     --project=local-grounding \
     --host-port=0.0.0.0:8922
   ```

   Keep this process running in its own terminal.

3. **Set environment variables**

   ```bash
   export FIRESTORE_PROJECT_ID=local-grounding
   export FIRESTORE_COLLECTION=apiKeys
   export FIRESTORE_DATABASE_NAME=grounding
   export FIRESTORE_EMULATOR_HOST=0.0.0.0:8922
   export GEMINI_API_KEY="dev-placeholder"
   export GEMINI_GROUNDING_FREE_LIMIT_PRO=1500
   export GEMINI_GROUNDING_FREE_LIMIT_FLASH=1500
   ```

4. **Install Python dependencies**

   From the repository root:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r aieng-agents-utils/aieng/agents/web_search/requirements-app.txt
   ```

   (Or use `uv pip install -r aieng-agents-utils/aieng/agents/web_search/requirements-app.txt`.)

5. **Run unit tests**

   ```bash
   pytest tests/test_web_search_auth.py
   ```

6. **Launch the API**

   ```bash
   uvicorn utils.web_search.app:app \
     --reload \
     --reload-dir aieng-agents-utils/aieng/agents/web_search \
     --port 8080
   ```

   - API docs: <http://localhost:8080/docs> (Gemini calls are live; make sure
     you supply a valid key if you need real responses). Note that your IDE might
     forward port 8080 to a different port automatically.

7. **Seed an admin API key (emulator)**

   ```python
   import asyncio

   from google.auth.credentials import AnonymousCredentials
   from google.cloud import firestore

   from src.utils.web_search.auth import APIKeyAuthenticator
   from src.utils.web_search.db import APIKeyRepository

   async def main():
       client = firestore.AsyncClient(
           project="local-grounding",
           credentials=AnonymousCredentials(),
       )
       repo = APIKeyRepository(client, collection_name="apiKeys")
       auth = APIKeyAuthenticator(repo)
       api_key, record = await auth.create_api_key(
           role="admin",
           owner="local dev",
           usage_limit=0,
           created_by="bootstrap",
           metadata={"note": "local admin"},
       )
       print("Admin key:", api_key)
       await client.close()

   asyncio.run(main())
   ```

Make sure to set FIRESTORE_EMULATOR_HOST before calling the above function.

---

## 4. Production Deployment

### 4.1 Project setup (one time)

1. Select a Google Cloud project and ensure billing is enabled.

2. Enable required services:

   ```bash
   gcloud services enable \
     run.googleapis.com \
     firestore.googleapis.com \
     artifactregistry.googleapis.com \
     secretmanager.googleapis.com \
     generativelanguage.googleapis.com
   ```

3. Create a named Firestore database:

   ```bash
   gcloud firestore databases create \
     --database=grounding \
     --region=REGION \
     --type=firestore-native
   ```

4. Create a service account:

   ```bash
   gcloud iam service-accounts create web-search-sa \
     --display-name="Gemini Grounding Proxy"
   ```

5. Grant roles to the service account:

   ```bash
   for ROLE in roles/datastore.user roles/secretmanager.secretAccessor; do
     gcloud projects add-iam-policy-binding "$PROJECT" \
       --member="serviceAccount:web-search-sa@${PROJECT}.iam.gserviceaccount.com" \
       --role="$ROLE"
   done
   ```

6. Create a Gemini API key in Google AI Studio and store it in Secret Manager:

   ```bash
   echo -n "$GEMINI_API_KEY" | gcloud secrets create GEMINI_API_KEY \
     --replication-policy="automatic" \
     --data-file=-
   ```

   Add new versions later with `gcloud secrets versions add GEMINI_API_KEY --data-file=-`.

7. Create (or reuse) an Artifact Registry repository:

   ```bash
   gcloud artifacts repositories create web-search \
     --repository-format=docker \
     --location=REGION
   ```

### 4.2 Build and push the container

```bash
export PROJECT=your-project-id
export REGION=us-central1
export IMAGE_NAME=grounding-proxy
export TAG=$(date +%Y%m%d%H%M)

gcloud builds submit aieng-agents-utils/aieng/agents/web_search \
  --tag "$REGION-docker.pkg.dev/$PROJECT/web-search/$IMAGE_NAME:$TAG"
```

### 4.3 Deploy to Cloud Run

```bash
gcloud run deploy web-search-proxy \
  --image="$REGION-docker.pkg.dev/$PROJECT/web-search/$IMAGE_NAME:$TAG" \
  --region="$REGION" \
  --allow-unauthenticated \
  --service-account="web-search-sa@$PROJECT.iam.gserviceaccount.com" \
  --set-env-vars="FIRESTORE_PROJECT_ID=$PROJECT,FIRESTORE_COLLECTION=apiKeys,FIRESTORE_DATABASE_NAME=grounding" \
  --set-secrets="GEMINI_API_KEY=GEMINI_API_KEY:latest" \
  --memory="512Mi" \
  --cpu="1" \
  --timeout="300" \
  --max-instances="10"
```

Adjust `--min-instances`, `--ingress`, or `--cpu-throttling` as needed.

### 4.4 Bootstrap the first admin API key (production)

Run this script locally with Application Default Credentials pointed at the
production project:

```python
import asyncio
from google.cloud import firestore
from utils.web_search.auth import APIKeyAuthenticator
from utils.web_search.db import APIKeyRepository

PROJECT = "your-project-id"
DATABASE = "grounding"
COLLECTION = "apiKeys"

async def main():
    client = firestore.AsyncClient(project=PROJECT, database=DATABASE)
    repo = APIKeyRepository(client, collection_name=COLLECTION)
    auth = APIKeyAuthenticator(repo)
    api_key, record = await auth.create_api_key(
        role="admin",
        owner="platform-team",
        usage_limit=0,
        created_by="bootstrap-script",
        metadata={"note": "Initial administrator"},
    )
    print("Store this admin key securely:", api_key)
    await client.close()

asyncio.run(main())
```

Add the plaintext key to Secret Manager or a vault immediately; it cannot be
retrieved later.

---

## 5. Verification Checklist

- `gcloud run services describe web-search-proxy --region=$REGION --format='value(status.url)'`
  to copy the service URL.
- Use the admin key to issue a test command:

  ```bash
  curl -sS https://SERVICE_URL/api/admin/api-keys \
    -H "X-API-Key: ADMIN_KEY" \
    -H "Content-Type: application/json" \
    -d '{"owner":"smoke-test","usage_limit":10}'
  ```

- Call the main endpoint with a newly minted key:

  ```bash
  curl -sS https://SERVICE_URL/api/v1/grounding_with_search \
    -H "Content-Type: application/json" \
    -H "X-API-Key: USER_KEY" \
    -d '{"query":"status check"}'
  ```

- Review logs to confirm Firestore and Gemini calls succeed:

  ```bash
  gcloud run services logs read web-search-proxy --region=$REGION
  ```

---

## 6. Troubleshooting

- **Docker push fails with 404.** Ensure the Artifact Registry hostname uses
  the correct region, e.g. `us-central1-docker.pkg.dev`.
- **Firestore permission errors.** Verify the service account has
  `roles/datastore.user` and that `FIRESTORE_PROJECT_ID` / `FIRESTORE_DATABASE_NAME`
  match the deployed database.
- **Gemini authentication failures.** Regenerate the Gemini API key, upload a
  new secret version, and redeploy or restart the Cloud Run service to load it.
