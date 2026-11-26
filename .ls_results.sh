#!/usr/bin/env bash
ls -1 /home/ctown-admin/results/paired/ 2>/dev/null | head
for d in /home/ctown-admin/results/paired/*/; do echo "-- $d"; ls -1 "$d" 2>/dev/null | head -3; done
