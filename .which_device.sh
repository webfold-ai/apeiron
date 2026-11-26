#!/usr/bin/env bash
echo "== the receipt's identity block"
grep -iE 'device|vulkan|gpu|pool' /home/ctown-admin/results/paired/new-engine-gpu/*.md 2>/dev/null | head -8
echo "== vulkan devices the loader reports"
(vulkaninfo --summary 2>/dev/null || true) | grep -iE 'deviceName|driverName|apiVersion' | head -8
echo "== nvidia ICD present?"
ls -la /usr/share/vulkan/icd.d/ 2>/dev/null | grep -i nvidia || echo 'no nvidia ICD json'
