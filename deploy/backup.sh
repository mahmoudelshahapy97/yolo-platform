#!/bin/sh
# Nightly pg_dump at 02:30 (container time, UTC), keeping BACKUP_KEEP_DAYS days.
# Restore: gunzip -c backups/inference-<stamp>.sql.gz | docker compose exec -T postgres psql -U inference -d inference
set -eu
KEEP="${BACKUP_KEEP_DAYS:-14}"
while true; do
  now=$(date +%s)
  next=$(date -d "$(date +%Y-%m-%d) 02:30" +%s 2>/dev/null || echo $((now + 86400)))
  [ "$next" -le "$now" ] && next=$((next + 86400))
  sleep $((next - now))
  stamp=$(date +%Y%m%d-%H%M%S)
  if pg_dump --no-owner --clean --if-exists | gzip > "/backups/inference-$stamp.sql.gz.tmp"; then
    mv "/backups/inference-$stamp.sql.gz.tmp" "/backups/inference-$stamp.sql.gz"
    echo "backup inference-$stamp.sql.gz written"
  else
    rm -f "/backups/inference-$stamp.sql.gz.tmp"
    echo "backup FAILED at $stamp" >&2
  fi
  find /backups -name 'inference-*.sql.gz' -mtime +"$KEEP" -delete
done
