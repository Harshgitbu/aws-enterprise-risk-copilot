# Security Rotation Checklist

Use this checklist if secrets were exposed in local files, screenshots, logs, or prior commits.

## 1) Rotate Immediately
- `GOOGLE_API_KEY`
- `HUGGINGFACE_TOKEN`
- `NEWSAPI_KEY`
- `FINNHUB_API_KEY`
- Any AWS IAM access keys used for this project

## 2) Update Runtime Secrets
- Update EC2 environment values.
- Update Render environment variables.
- Update GitHub Actions secrets (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`).

## 3) Verify
- Restart backend/frontend services.
- Confirm `/health` is up.
- Call `/ai/copilot/advanced` and verify Gemini path works.

## 4) Prevent Recurrence
- Keep `.env` untracked.
- Commit only `.env.example` placeholders.
- Avoid posting full terminal screenshots containing secrets.
- Enable GitHub secret scanning and push protection.

## 5) Optional History Cleanup
If secrets were committed previously, rewrite git history before sharing the repo publicly, then force-push with team coordination.
