# 🚀 Railway Deployment Checklist - Customer Service

## ✅ **Pre-Deployment Checklist**

### **1. Code Changes**
- [ ] Dockerfile uses direct uvicorn command: `CMD uvicorn src.main:app --host 0.0.0.0 --port ${PORT:-8000}`
- [ ] No hardcoded PORT in Dockerfile
- [ ] EXPOSE directive uses `${PORT:-8000}`
- [ ] All shared dependencies are copied correctly
- [ ] PYTHONPATH is set to `/app/src`

### **2. Railway Dashboard Configuration**
- [ ] **DO NOT set PORT manually** (Railway sets this automatically)
- [ ] Set required environment variables:
  - [ ] `JWT_SECRET` (your secure secret)
  - [ ] `SECRET_KEY` (your secure secret)
  - [ ] `ENVIRONMENT=production`
  - [ ] `DEBUG=false`
  - [ ] `LOG_LEVEL=INFO`
  - [ ] `ALLOWED_ORIGINS` (your domain URLs)
  - [ ] `SERVICE_NAME=customer-service`

### **3. Database Configuration**
- [ ] Use Railway's built-in PostgreSQL
- [ ] Railway automatically provides `DATABASE_URL`
- [ ] **DO NOT set DATABASE_URL manually** if using Railway PostgreSQL

### **4. Service URLs (if using multiple services)**
- [ ] Update inter-service URLs to use Railway domains:
  - [ ] `AUTH_SERVICE_URL=https://your-auth-service.railway.app`
  - [ ] `CUSTOMER_SERVICE_URL=https://your-customer-service.railway.app`
  - [ ] etc.

## 🚀 **Deployment Steps**

### **Step 1: Commit and Push**
```bash
git add .
git commit -m "Fix Railway PORT handling - use direct uvicorn command"
git push
```

### **Step 2: Monitor Railway Deployment**
1. Go to Railway dashboard
2. Check deployment logs
3. Look for successful startup message

### **Step 3: Verify Deployment**
1. Check service health endpoint
2. Verify logs show correct port usage
3. Test API endpoints

## 🔍 **Expected Log Output**

**Successful deployment should show:**
```
INFO:     Started server process [1]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:12345 (Press CTRL+C to quit)
```

**NOT this error:**
```
Error: Invalid value for '--port': '$PORT' is not a valid integer.
```

## 🛠️ **Troubleshooting**

### **If PORT error persists:**

1. **Check Railway Variables:**
   - Go to service → Variables tab
   - Ensure PORT is NOT manually set
   - Railway should set it automatically

2. **Verify Dockerfile:**
   ```dockerfile
   # Should be:
   CMD uvicorn src.main:app --host 0.0.0.0 --port ${PORT:-8000}
   
   # NOT:
   CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "$PORT"]
   ```

3. **Check Environment Variables:**
   - Railway logs should show the actual PORT value
   - Look for environment variable expansion

### **Alternative Solutions:**

**Option 1: Shell Command (if direct uvicorn doesn't work)**
```dockerfile
CMD uvicorn src.main:app --host 0.0.0.0 --port $PORT
```

**Option 2: Python Script (if shell expansion doesn't work)**
```dockerfile
COPY start.py /app/start.py
CMD ["python", "/app/start.py"]
```

## 📋 **Railway-Specific Notes**

1. **PORT Handling:**
   - Railway sets PORT automatically
   - Use `${PORT:-8000}` for fallback
   - Never hardcode PORT in Dockerfile

2. **Database:**
   - Use Railway PostgreSQL for automatic DATABASE_URL
   - Don't set DATABASE_URL manually if using Railway DB

3. **Environment Variables:**
   - Set production variables in Railway dashboard
   - Use HTTPS URLs for inter-service communication

4. **Logs:**
   - Check Railway deployment logs for debugging
   - Look for PORT-related messages

## 🎯 **Success Criteria**

After deployment, you should see:
- ✅ No PORT-related errors
- ✅ Service starts successfully
- ✅ Health endpoint responds
- ✅ Logs show correct port usage
- ✅ API endpoints work correctly

## 📞 **Support**

If issues persist:
1. Check Railway deployment logs
2. Verify environment variables
3. Test with alternative Dockerfile approaches
4. Contact Railway support if needed

Your Railway deployment should now work correctly! 🚀 