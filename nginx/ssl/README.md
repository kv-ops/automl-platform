# SSL certificates

Place your TLS certificate and private key in this directory before starting the Nginx container.

Expected file names:

- `tls.crt` – PEM encoded certificate (full chain recommended)
- `tls.key` – PEM encoded private key

For local testing you can generate a self-signed pair with:

```bash
openssl req -x509 -nodes -days 365 -newkey rsa:4096 \
  -keyout nginx/ssl/tls.key \
  -out nginx/ssl/tls.crt \
  -subj "/CN=localhost"
```

These files are mounted read-only at `/etc/nginx/ssl/` inside the container.
