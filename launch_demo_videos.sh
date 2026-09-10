#!/usr/bin/env bash
# Prima del primo invio (SLURM apre i log prima di avviare questo script):
#   mkdir -p results/logs
#   sbatch launch_demo_videos.sh
#   squeue -u "$USER"
# Modificare la lista VIDEOS direttamente in launch_demo_videos.sh.
#SBATCH --job-name=smirk-demo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=results/logs/smirk_demo_%j.out
#SBATCH --error=results/logs/smirk_demo_%j.err
# Le impostazioni seguenti dipendono dall'assegnazione sul cluster Leonardo.
# Rimuovere un solo '#' e sostituire i segnaposto prima dell'uso, se necessarie:
##SBATCH --account=<account_leonardo>
##SBATCH --partition=<partizione_gpu_leonardo>

set -Eeuo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# Impostare CONDA_SH prima di sbatch se conda non e' nel PATH del job, ad esempio:
#   export CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"
CONDA_SH="${CONDA_SH:-}"
OUTPUT_DIR="$REPO_ROOT/results/smirk_generator"
CHECKPOINT="$REPO_ROOT/pretrained_models/SMIRK_em1.pt"
FACE_DETECTION_MODE="pose_roi"
VISUALIZATION_LAYOUT="overlay"
OVERLAY_ALPHA="0.55"

VIDEOS=(
  "/leonardo_work/IscrC_SLPSCALE/RGB2SMPLX/rgb2smplx_test/input/CSY/front/100.MP4"
  # "/percorso/secondo_video.MP4"
)

fatal_missing=0
for required in "$REPO_ROOT/demo_video.py" "$CHECKPOINT" "$REPO_ROOT/assets/face_landmarker.task"; do
  if [[ ! -f "$required" ]]; then
    printf 'ERRORE: risorsa obbligatoria mancante: %s\n' "$required" >&2
    fatal_missing=1
  fi
done
if (( fatal_missing )); then
  exit 2
fi

if [[ -z "$CONDA_SH" ]]; then
  if command -v conda >/dev/null 2>&1; then
    conda_base="$(conda info --base)" || {
      printf 'ERRORE: impossibile determinare la base Conda con "conda info --base".\n' >&2
      exit 2
    }
    CONDA_SH="$conda_base/etc/profile.d/conda.sh"
  else
    printf 'ERRORE: conda non disponibile. Impostare CONDA_SH al percorso esatto di conda.sh.\n' >&2
    exit 2
  fi
fi
if [[ ! -f "$CONDA_SH" ]]; then
  printf 'ERRORE: inizializzazione Conda mancante: %s\n' "$CONDA_SH" >&2
  exit 2
fi
# shellcheck source=/dev/null
source "$CONDA_SH"
if ! conda activate smirk; then
  printf 'ERRORE: impossibile attivare l’ambiente Conda "smirk" tramite %s.\n' "$CONDA_SH" >&2
  exit 2
fi

hostname
nvidia-smi
which python
python --version

mkdir -p "$OUTPUT_DIR" "$OUTPUT_DIR/logs"

success=0
failed=0
skipped=0
total=${#VIDEOS[@]}
job_id="${SLURM_JOB_ID:-manual}"

for video in "${VIDEOS[@]}"; do
  if [[ ! -f "$video" ]]; then
    printf 'ERRORE: video mancante, elemento saltato: %s\n' "$video" >&2
    ((skipped += 1))
    continue
  fi

  filename="$(basename -- "$video")"
  stem="${filename%.*}"
  # Il digest del percorso rende univoci output e log anche per basename uguali.
  path_hash="$(printf '%s' "$video" | sha256sum | cut -c1-10)"
  output_name="${stem}_${path_hash}"
  video_log="$OUTPUT_DIR/logs/${output_name}_${job_id}.log"

  printf '\nElaborazione: %s\nOutput: %s/%s.mp4\nLog: %s\n' \
    "$video" "$OUTPUT_DIR" "$output_name" "$video_log"

  set +e
  python -u demo_video.py \
    --input_path "$video" \
    --out_path "$OUTPUT_DIR" \
    --output-name "$output_name" \
    --checkpoint "$CHECKPOINT" \
    --device cuda \
    --crop \
    --render_orig \
    --face-detection-mode "$FACE_DETECTION_MODE" \
    --visualization-layout "$VISUALIZATION_LAYOUT" \
    --overlay-alpha "$OVERLAY_ALPHA" \
    2>&1 | tee "$video_log"
  python_status=${PIPESTATUS[0]}
  set -e

  if (( python_status == 0 )); then
    ((success += 1))
  else
    printf 'ERRORE: elaborazione fallita (exit code %d): %s\n' "$python_status" "$video" >&2
    ((failed += 1))
  fi
done

printf '\nVideo totali: %d\nCompletati: %d\nFalliti: %d\nSaltati: %d\nOutput: %s\n' \
  "$total" "$success" "$failed" "$skipped" "$OUTPUT_DIR"

if (( failed > 0 )); then
  exit 1
fi
