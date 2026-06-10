# ─────────────────────────────────────────────────────────
# Frontend Dockerfile — two-stage build
#
# Stage 1 (build): npm install + vite build
# Stage 2 (serve): nginx serves static assets
# ─────────────────────────────────────────────────────────

# ── Stage 1: build ────────────────────────────────────────
FROM node:20-alpine AS build

WORKDIR /app

COPY package.json package-lock.json* ./
RUN npm ci --silent

COPY . .

# Build with production API URL pointing to backend service
ARG VITE_API_URL=/api
ENV VITE_API_URL=$VITE_API_URL

RUN npm run build

# ── Stage 2: serve ────────────────────────────────────────
FROM nginx:alpine AS serve

# Copy built assets
COPY --from=build /app/dist /usr/share/nginx/html

# Nginx config: serve SPA, proxy /api to backend
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=5s \
    CMD wget -qO- http://localhost/ || exit 1

CMD ["nginx", "-g", "daemon off;"]
