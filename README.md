# triage-lens
AI friday project on gen ai based ticket resolution.

## Backend setup

Use Python 3.10 on Windows for the backend because `chromadb` depends on a
Windows package that does not reliably support Python 3.12+.

To recreate the backend environment used in development:

```powershell
.\backend\setup_backend_env.ps1
```

This creates `.venv310`, installs the pinned packages from
`backend/requirements-lock.txt`, and prepares the local cache folders used by
the backend scripts.

If you only want the smaller hand-maintained dependency set instead of the full
locked environment:

```powershell
.\backend\setup_backend_env.ps1 -UseMinimalRequirements
```

Then run:

```powershell
.\backend\run_ingest.ps1
.\backend\run_api.ps1
```
