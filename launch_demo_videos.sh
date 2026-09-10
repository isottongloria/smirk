#!/usr/bin/env bash
#
# Prima del primo invio (SLURM apre i log prima di avviare questo script):
#   mkdir -p results/logs
#   sbatch launch_demo_videos.sh
#   squeue -u "$USER"
#
# Modificare la lista VIDEOS qui sotto per scegliere i filmati da elaborare.
# Le direttive account/partition dipendono dal progetto Leonardo e sono quindi
# disabilitate: rimuovere un '#' e sostituire i valori prima di sbatch, se richiesto.
#
#SBATCH --job-name=smirk-demo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=results/logs/smirk_demo_%j.out
#SBATCH --error=results/logs/smirk_demo_%j.err
#SBATCH --account=IscrC_SLPSCALE
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod  

set -Eeuo pipefail

# sbatch copia lo script nello spool di Slurm prima di eseguirlo: in quel caso
# BASH_SOURCE punta a /var/spool/slurmd/job..., non al checkout di SMIRK.
# SMIRK_ROOT resta sovrascrivibile per usare lo stesso launcher da altri checkout.
SMIRK_ROOT="${SMIRK_ROOT:-/leonardo_work/IscrC_SLPSCALE/smirk}"
if [[ ! -d "$SMIRK_ROOT" ]]; then
  printf 'ERRORE: directory SMIRK mancante: %s\n' "$SMIRK_ROOT" >&2
  printf '%s\n' 'Impostare SMIRK_ROOT al percorso assoluto del repository.' >&2
  exit 1
fi
REPO_ROOT="$(cd -- "$SMIRK_ROOT" && pwd -P)"
cd "$REPO_ROOT"

# Facoltativo: percorso esplicito a conda.sh per i nodi non interattivi.
# Esempio: CONDA_SH="/leonardo/home/user/miniconda3/etc/profile.d/conda.sh"
CONDA_SH="${CONDA_SH:-}"

OUTPUT_DIR="$REPO_ROOT/results/smirk_generator"
CHECKPOINT="$REPO_ROOT/pretrained_models/SMIRK_em1.pt"
FACE_DETECTION_MODE="pose_roi"
VISUALIZATION_LAYOUT="overlay"
OVERLAY_ALPHA="0.55"

VIDEOS=(
"/leonardo_work/IscrC_SLPSCALE/smirk/samples/05May_2011_Thursday_tagesschau-26.mp4"
"/leonardo_work/IscrC_SLPSCALE/smirk/samples/05May_2011_Thursday_tagesschau-27.mp4"
"/leonardo_work/IscrC_SLPSCALE/RGB2SMPLX/rgb2smplx_test/input/CSY/front/0.MP4"
"/leonardo_work/IscrC_SLPSCALE/RGB2SMPLX/rgb2smplx_test/input/CSY/front/100.MP4"
"/leonardo_work/IscrC_SLPSCALE/RGB2SMPLX/rgb2smplx_test/input/CSY/front/3D.MP4"
"/leonardo_work/IscrC_SLPSCALE/RGB2SMPLX/rgb2smplx_test/input/CSY/front/3.MP4"
"/leonardo_work/IscrC_SLPSCALE/RGB2SMPLX/rgb2smplx_test/input/CSY/front/40.MP4"
)

fatal_missing=0
for required_file in \
  "$REPO_ROOT/demo_video.py" \
  "$CHECKPOINT" \
  "$REPO_ROOT/assets/face_landmarker.task"; do
  if [[ ! -f "$required_file" ]]; then
    printf 'ERRORE: risorsa richiesta mancante: %s\n' "$required_file" >&2
    fatal_missing=1
  fi
done
(( fatal_missing == 0 )) || exit 1

# Nei job batch la shell non carica normalmente l'inizializzazione di Conda.
if [[ -z "$CONDA_SH" ]]; then
  if command -v conda >/dev/null 2>&1; then
    conda_base="$(conda info --base)" || {
      printf 'ERRORE: impossibile determinare la base Conda con: conda info --base\n' >&2
      exit 1
    }
    CONDA_SH="$conda_base/etc/profile.d/conda.sh"
  else
    printf '%s\n' 'ERRORE: conda non disponibile; impostare CONDA_SH al percorso esatto di conda.sh.' >&2
    exit 1
  fi
fi

if [[ ! -f "$CONDA_SH" ]]; then
  printf 'ERRORE: inizializzazione Conda mancante: %s\n' "$CONDA_SH" >&2
  exit 1
fi
# shellcheck source=/dev/null
source "$CONDA_SH"
if ! conda activate smirk; then
  printf '%s\n' "ERRORE: ambiente Conda richiesto non disponibile: smirk (conda.sh: $CONDA_SH)" >&2
  exit 1
fi

hostname
nvidia-smi
which python
python --version

mkdir -p "$OUTPUT_DIR/logs"

success=0
failed=0
skipped=0
job_id="${SLURM_JOB_ID:-manual}"

for index in "${!VIDEOS[@]}"; do
  video="${VIDEOS[$index]}"
  if [[ ! -f "$video" ]]; then
    printf 'ERRORE: video mancante, elemento saltato: %s\n' "$video" >&2
    ((skipped += 1))
    continue
  fi

  filename="${video##*/}"
  stem="${filename%.*}"
  safe_stem="${stem//[^[:alnum:]_.-]/_}"
  # Indice e checksum del percorso rendono univoci anche basename identici.
  path_checksum="$(printf '%s' "$video" | cksum | awk '{print $1}')"
  video_key="$(printf '%03d_%s_%s' "$((index + 1))" "$safe_stem" "$path_checksum")"
  video_output_dir="$OUTPUT_DIR/$video_key"
  video_log="$OUTPUT_DIR/logs/${safe_stem}_${job_id}_${video_key}.log"
  mkdir -p "$video_output_dir"

  printf 'Elaborazione: %s\nOutput video: %s\nLog video: %s\n' \
    "$video" "$video_output_dir" "$video_log"

  set +e
  python -u "$REPO_ROOT/demo_video.py" \
    --input_path "$video" \
    --out_path "$video_output_dir" \
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
  "${#VIDEOS[@]}" "$success" "$failed" "$skipped" "$OUTPUT_DIR"

(( failed == 0 )) || exit 1
