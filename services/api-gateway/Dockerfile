FROM kong:3.4

# Copy Kong configuration
COPY kong.yml /kong/declarative/kong.yml
COPY kong-setup.sh /kong/kong-setup.sh

# Make setup script executable
RUN chmod +x /kong/kong-setup.sh

# Set environment variables
ENV KONG_DATABASE=off
ENV KONG_DECLARATIVE_CONFIG=/kong/declarative/kong.yml
ENV KONG_PROXY_ACCESS_LOG=/dev/stdout
ENV KONG_ADMIN_ACCESS_LOG=/dev/stdout
ENV KONG_PROXY_ERROR_LOG=/dev/stderr
ENV KONG_ADMIN_ERROR_LOG=/dev/stderr
ENV KONG_ADMIN_LISTEN=0.0.0.0:8001
ENV KONG_PROXY_LISTEN=0.0.0.0:8000
ENV KONG_PLUGINS=bundled,jwt,cors,rate-limiting,request-transformer,response-transformer,correlation-id,file-log

# Expose ports
EXPOSE 8000 8443 8001 8444

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD kong health

# Start Kong
CMD ["kong", "start", "--vv"] 