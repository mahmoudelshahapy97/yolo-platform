#!/bin/sh
# One-shot: create a self-signed certificate if none is mounted. Put a real one at
# deploy/nginx/certs/tls.crt + tls.key (e.g. from your CA or certbot) to replace it.
set -eu
cd /certs
if [ -s tls.crt ] && [ -s tls.key ]; then
  echo "certificate present: $(openssl x509 -in tls.crt -noout -subject -enddate | tr '\n' ' ')"
  exit 0
fi
SAN="DNS:localhost,IP:127.0.0.1"
for h in $(echo "${TLS_HOSTNAMES:-}" | tr ',' ' '); do
  case "$h" in *[!0-9.]*) SAN="$SAN,DNS:$h" ;; *) SAN="$SAN,IP:$h" ;; esac
done
openssl req -x509 -newkey rsa:2048 -nodes -days 825 -subj "/CN=yolo-platform" \
  -addext "subjectAltName=$SAN" -keyout tls.key -out tls.crt 2>/dev/null
chmod 644 tls.crt; chmod 600 tls.key; chown 101:101 tls.key 2>/dev/null || true
echo "self-signed certificate created for $SAN"
