# 🚀 Railway Deployment Guide - Fixing PORT Issues

## 🚨 **Problem: PORT Environment Variable Not Working**

You're experiencing this error:
```
Error: Invalid value for '--port': '$PORT' is not a valid integer.
```

This happens because Railway sets the `PORT` environment variable dynamically, but your application isn't handling it correctly.

## ✅ **Solution Overview**

### **What We Fixed:**
1. **Enhanced start.sh script** with Railway-specific PORT handling
2. **Updated Dockerfile** to work with Railway's dynamic PORT assignment
3. **Created Railway environment template** with proper configuration
4. **Added comprehensive debugging** to identify issues

## 🔧 **Step-by-Step Fix**

### **Step 1: Verify Your Files Are Updated**

Your `services/customer-service/start.sh` should now contain:
```bash
#!/bin/bash
set -e

# Railway-specific PORT handling
echo "=== Railway Deployment Debug ==="
echo "Raw PORT variable: '$PORT'"
echo "Environment check:"
env | grep -i port || echo "No PORT found in environment"

# Handle Railway's dynamic PORT assignment
if [ -z "$PORT" ] || [ "$PORT" = "$PORT" ]; then
    PORT=8000
    echo "⚠️  PORT not properly set, using default: $PORT"
else
    echo "✅ Using Railway PORT: $PORT"
fi

# Ensure PORT is a valid integer
if ! [[ "$PORT" =~ ^[0-9]+$ ]]; then
    echo "❌ Invalid PORT value: '$PORT'"
    echo "🔄 Falling back to default port: 8000"
    PORT=8000
fi

echo "🚀 Starting Customer Service on port $PORT"
exec uvicorn src.main:app --host 0.0.0.0 --port "$PORT"
```

### **Step 2: Railway Environment Variables**

In your Railway dashboard, set these environment variables:

#### **Required Variables:**
```bash
# Railway automatically sets these - DO NOT SET MANUALLY
# PORT=random_port (Railway sets this)
# DATABASE_URL=postgresql://... (Railway PostgreSQL sets this)

# You must set these:
JWT_SECRET=your-super-secure-jwt-secret-key-for-production-2024-v1
SECRET_KEY=your-super-secure-jwt-secret-key-for-production-2024-v1
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO
ALLOWED_ORIGINS=https://yourdomain.com,https://app.yourdomain.com
SERVICE_NAME=customer-service
```

#### **Optional Variables (set as needed):**
```bash
# Redis (if using Railway Redis)
REDIS_URL=redis://user:pass@host:port

# RabbitMQ (if using Railway RabbitMQ)
RABBITMQ_URL=amqp://user:pass@host:port

# Service URLs (update with your actual Railway URLs)
AUTH_SERVICE_URL=https://your-auth-service.railway.app
CUSTOMER_SERVICE_URL=https://your-customer-service.railway.app
# ... other service URLs
```

### **Step 3: Railway Dashboard Configuration**

1. **Go to your Railway project dashboard**
2. **Navigate to your service**
3. **Go to Variables tab**
4. **Add the environment variables listed above**
5. **Make sure you DON'T set PORT manually** (Railway sets this)

### **Step 4: Deploy and Test**

1. **Commit and push your changes:**
   ```bash
   git add .
   git commit -m "Fix Railway PORT handling"
   git push
   ```

2. **Railway will automatically redeploy**

3. **Check the logs in Railway dashboard:**
   - You should see the debug output showing PORT handling
   - Look for: "✅ Using Railway PORT: [number]"

## 🔍 **Debugging Steps**

### **If the issue persists:**

1. **Check Railway logs:**
   - Go to your service in Railway dashboard
   - Click on "Deployments" tab
   - Click on the latest deployment
   - Check the logs for debug output

2. **Expected log output:**
   ```
   === Railway Deployment Debug ===
   Raw PORT variable: '12345'
   Environment check:
   PORT=12345
   ✅ Using Railway PORT: 12345
   🚀 Starting Customer Service on port 12345
   ```

3. **If you see "PORT not properly set":**
   - Check Railway Variables tab
   - Ensure you haven't manually set PORT
   - Railway should set it automatically

## 🚀 **Alternative Solutions**

### **Option 1: Direct Uvicorn Command (Simpler)**

If the shell script approach doesn't work, you can modify your Dockerfile to use a direct command:

```dockerfile
# Replace the CMD line with:
CMD uvicorn src.main:app --host 0.0.0.0 --port ${PORT:-8000}
```

### **Option 2: Python Script Approach**

Create a `start.py` file:
```python
import os
import uvicorn

port = int(os.getenv("PORT", 8000))
print(f"Starting on port: {port}")

uvicorn.run("src.main:app", host="0.0.0.0", port=port)
```

Then update Dockerfile:
```dockerfile
CMD ["python", "start.py"]
```

## 📋 **Railway-Specific Best Practices**

1. **Never set PORT manually** - Railway handles this
2. **Use Railway's built-in PostgreSQL** - they provide DATABASE_URL
3. **Use Railway's built-in Redis** - they provide REDIS_URL
4. **Set production environment variables** in Railway dashboard
5. **Use HTTPS URLs** for inter-service communication

## 🎯 **Expected Result**

After implementing these fixes:
- ✅ No more "Invalid value for '--port': '$PORT'" errors
- ✅ Service starts on Railway's assigned port
- ✅ Clear debug output in logs
- ✅ Proper fallback to default port if needed

## 📞 **Support**

If you still encounter issues:
1. Check Railway logs for debug output
2. Verify environment variables in Railway dashboard
3. Ensure you're not manually setting PORT
4. Try the alternative solutions above

Your Railway deployment should now work correctly! 🚀 