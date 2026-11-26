#!/usr/bin/env bash
for d in /home/ctown-admin/models/roundtrip /home/ctown-admin/w7-fixtures /home/ctown-admin/models; do
  echo "== $d"
  ls -la "$d"/*.gguf 2>/dev/null | head -4
  find "$d" -maxdepth 2 -name '*.gguf' 2>/dev/null | head -4
done
