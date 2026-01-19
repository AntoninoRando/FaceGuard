# FaceGuard Docker Setup

## Quick Start

### Build and Run (Production)
```bash
# Build and start the container
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the container
docker-compose down
```

### Development Mode (Hot Reload)
```bash
# Run with source code mounted for development
docker-compose -f docker-compose.dev.yml up
```

## Docker Commands

### Build
```bash
# Build the image
docker-compose build

# Build without cache
docker-compose build --no-cache
```

### Run
```bash
# Start in background (detached)
docker-compose up -d

# Start in foreground
docker-compose up

# Start and rebuild
docker-compose up --build
```

### Manage
```bash
# Stop containers
docker-compose stop

# Remove containers
docker-compose down

# Remove containers and volumes
docker-compose down -v

# View logs
docker-compose logs -f faceguard-api

# Execute command in running container
docker-compose exec faceguard-api bash

# Check container status
docker-compose ps
```

## API Access

Once running, the API will be available at:
- **Base URL**: http://localhost:8000
- **Swagger Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Data Persistence

The following directories are mounted as volumes for data persistence:
- `src/data/` - Gallery images and training data
- `src/embeddings/` - Face embeddings
- `src/models/` - Model files
- `src/antispoofing_config.json` - Anti-spoofing configuration
- `src/antispoofing_feedback.json` - Feedback data

## Environment Variables

You can customize the deployment by adding environment variables in `docker-compose.yml`:

```yaml
environment:
  - PORT=8000
  - HOST=0.0.0.0
  - LOG_LEVEL=info
```

## Troubleshooting

### Check container health
```bash
docker-compose ps
```

### View detailed logs
```bash
docker-compose logs -f
```

### Access container shell
```bash
docker-compose exec faceguard-api bash
```

### Rebuild after code changes (production)
```bash
docker-compose down
docker-compose up --build
```

## Production Deployment

For production, consider:
1. Remove the `reload=True` flag in `api.py` or set via environment
2. Use a proper WSGI server configuration
3. Set up proper logging and monitoring
4. Configure reverse proxy (nginx/traefik)
5. Use secrets management for sensitive data
6. Enable HTTPS/TLS

### Example with Nginx Reverse Proxy
```yaml
# Add to docker-compose.yml
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - faceguard-api
```

## System Requirements

- Docker Engine 20.10+
- Docker Compose 1.29+
- At least 4GB RAM
- 10GB disk space for images and data
