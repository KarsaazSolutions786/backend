FROM python:3.11-slim

# Add build argument to force rebuild
ARG BUILD_DATE=unknown
ARG VCS_REF=unknown

# Ensure the working directory is set to /app
WORKDIR /app

# Copy the application code
COPY . .

# Install dependencies and set up the environment
RUN pip install -r requirements.txt

# Expose the port
EXPOSE $PORT

# Copy and make executable
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Run with environment variables
CMD ["/app/start.sh"]