#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

SBATCH="${1:-slurm_smoke.sbatch}"
[[ -f "$SBATCH" ]] || { echo "missing: $SBATCH"; exit 1; }

jobname="$(awk -F= '/^#SBATCH --job-name=/{gsub(/^[[:space:]]+|[[:space:]]+$/,"",$2); print $2; exit}' "$SBATCH")"
opat="$(awk -F= '/^#SBATCH --output=/{gsub(/^[[:space:]]+|[[:space:]]+$/,"",$2); print $2; exit}' "$SBATCH")"
epat="$(awk -F= '/^#SBATCH --error=/{gsub(/^[[:space:]]+|[[:space:]]+$/,"",$2); print $2; exit}' "$SBATCH")"

[[ -n "$jobname" ]] || jobname="job"
[[ -n "$opat" && -n "$epat" ]] || { echo "$SBATCH needs #SBATCH --output= and --error="; exit 1; }

mkdir -p "$(dirname "$opat")"

jid="$(sbatch --parsable "$SBATCH")"

out="${opat//\%x/$jobname}"
out="${out//\%j/$jid}"
err="${epat//\%x/$jobname}"
err="${err//\%j/$jid}"

echo "job=$jid  name=$jobname"
echo "out=$out"
echo "err=$err"
echo "Ctrl+C stops tail only; job keeps running (scancel $jid to kill)."

for _ in $(seq 1 300); do
  [[ -f "$out" || -f "$err" ]] && break
  sleep 1
done

exec tail -f "$out" "$err"

