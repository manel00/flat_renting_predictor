# Build stage
FROM node:20-alpine AS builder

WORKDIR /app

# Copy package files first for better layer caching
COPY package.json package-lock.json* ./

# Install dependencies with cache mount for faster rebuilds
RUN --mount=type=cache,target=/root/.npm \
    npm install --prefer-offline --no-audit

# Copy source code
COPY . .

# Build the application
RUN npm run build -- --configuration production

# Production stage
FROM nginx:alpine

# Copy built application from builder
COPY --from=builder /app/dist/frontend/browser /usr/share/nginx/html

# For Render: use nginx.render.conf (no backend proxy)
# For local Docker: override this in docker-compose.yml
COPY nginx.render.conf /etc/nginx/conf.d/default.conf

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD wget --quiet --tries=1 --spider http://localhost/health || exit 1

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
