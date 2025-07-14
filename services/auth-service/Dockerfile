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

# Make start.sh executable
RUN chmod +x /app/start.sh

# Use the corrected start.sh script
CMD ["/app/start.sh"] 