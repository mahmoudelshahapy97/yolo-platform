#!/usr/bin/env bash
# Transcode the test clips into camera-like streams for the RTSP test rig:
#   videos/sub/clipN.mp4   ~640x360 H.264, 25 fps, 2 s GOP   (an IP camera's substream)
#   videos/main/clipN.mp4  1920x1080 H.264, 25 fps, 2 s GOP  (the main stream)
# Runs ffmpeg inside the worker image (the host has none). Run from the repo root.
set -euo pipefail
IMAGE="${WORKER_IMAGE:-yolo-platform-worker:dev}"
CLIPS=(cctv_video1.mp4 cctv_video3.mp4 cctv_video20.mp4 normal-1.mp4 5.mp4 fire1.mp4)
mkdir -p videos/sub videos/main
i=0
for clip in "${CLIPS[@]}"; do
  for kind in sub main; do
    out="videos/$kind/clip$i.mp4"
    [ -s "$out" ] && continue
    if [ "$kind" = sub ]; then vf="scale=640:360:force_original_aspect_ratio=decrease,pad=ceil(iw/2)*2:ceil(ih/2)*2"; br=600k
    else vf="scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2"; br=4000k; fi
    docker run --rm -u "$(id -u):$(id -g)" -v "$PWD/videos:/videos" --entrypoint ffmpeg "$IMAGE" \
      -loglevel error -y -i "/videos/$clip" -vf "$vf,fps=25" -c:v libx264 -preset veryfast -profile:v main \
      -b:v "$br" -maxrate "$br" -bufsize "$br" -g 50 -keyint_min 50 -sc_threshold 0 -an "/$out"
    echo "wrote $out"
  done
  i=$((i + 1))
done
