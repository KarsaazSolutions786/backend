# Railway Deployment Configuration

This directory contains Railway-specific configuration files and documentation.

## Quick Deploy

1. **Push to GitHub**: Ensure all code is committed and pushed
2. **Create Railway Project**: Connect GitHub repo to Railway
3. **Add PostgreSQL**: Add PostgreSQL service to your project
4. **Set Environment Variables**: Add required env vars (see main README)
5. **Initialize Database**: Run `railway run python scripts/init_db_railway.py`

## Files Overview

- `../Procfile` - Process definition for Railway
- `../Dockerfile` - Container configuration
- `../alembic.ini` - Database migration configuration
- `../railway.json` - Railway build configuration
- `../railway.toml` - Railway service configuration

## Environment Variables Required

```bash
SECRET_KEY=8LrIcmpF1_QFIfGlLY6KtpvftqC4Co4mK4KyPOwrtOE
DEBUG=false
DEV_MODE=false
```

Auto-provided by Railway:
- `PORT` (automatic)
- `DATABASE_URL` (when PostgreSQL added)

## Post-Deployment

1. Verify health check: `/health`
2. Check API docs: `/docs`
3. Test main endpoint: `/api/v1/stt/transcribe-and-respond`

## Support

For detailed deployment instructions, see the main README.md file. 