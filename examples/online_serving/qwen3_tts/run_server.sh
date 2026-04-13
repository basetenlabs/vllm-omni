#!/bin/bash
# Launch vLLM-Omni server for Qwen3-TTS models
#
# Usage:
#   ./run_server.sh                                                        # Default: 1.7B CustomVoice
#   ./run_server.sh CustomVoice                                            # 1.7B CustomVoice
#   ./run_server.sh VoiceDesign 1.7B                                       # 1.7B VoiceDesign
#   ./run_server.sh Base 0.6B                                              # 0.6B Base (voice clone)
#   ./run_server.sh /path/to/local/model                                   # Local model directory
#   SERVED_MODEL_NAME=my-tts ./run_server.sh                               # Custom served model name
#   STAGE_CONFIGS_PATH=./my_qwen_stage_config.yaml ./run_server.sh         # Custom stage config
#
# Timestamps mode (word-level alignment via Qwen3-ForcedAligner-0.6B):
#   ./run_server.sh --timestamps CustomVoice 1.7B
#   FORCED_ALIGNER_MODEL=/local/path ./run_server.sh --timestamps
#   FORCED_ALIGNER_DEVICE=cuda:1 ./run_server.sh --timestamps

set -e

ENABLE_TIMESTAMPS=false
if [ "$1" = "--timestamps" ]; then
    ENABLE_TIMESTAMPS=true
    shift
fi

ARG="${1:-CustomVoice}"
SIZE="${2:-1.7B}"

if [ -d "$ARG" ]; then
    MODEL="$ARG"
else
    TASK_TYPE="$ARG"
    case "$SIZE" in
        0.6B|1.7B) ;;
        *)
            echo "Unknown size: $SIZE"
            echo "Supported: 0.6B, 1.7B"
            exit 1
            ;;
    esac
    case "$TASK_TYPE" in
        CustomVoice|VoiceDesign|Base)
            MODEL="Qwen/Qwen3-TTS-12Hz-${SIZE}-${TASK_TYPE}"
            ;;
        *)
            echo "Unknown task type: $TASK_TYPE"
            echo "Supported: CustomVoice, VoiceDesign, Base, or a local directory path"
            exit 1
            ;;
    esac
fi

SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-}"
STAGE_CONFIGS_PATH="${STAGE_CONFIGS_PATH:-vllm_omni/model_executor/stage_configs/qwen3_tts_latency.yaml}"

if [ "$ENABLE_TIMESTAMPS" = true ]; then
    export FORCED_ALIGNER_MODEL="${FORCED_ALIGNER_MODEL:-Qwen/Qwen3-ForcedAligner-0.6B}"
    export FORCED_ALIGNER_DEVICE="${FORCED_ALIGNER_DEVICE:-cuda:0}"
    export FORCED_ALIGNER_DTYPE="${FORCED_ALIGNER_DTYPE:-bfloat16}"

    # Reserve GPU memory for the aligner model (~1GB in bf16)
    GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.8}"

    if [ -d "$FORCED_ALIGNER_MODEL" ]; then
        echo "Timestamps enabled — using local ForcedAligner model: $FORCED_ALIGNER_MODEL"
    else
        echo "Timestamps enabled — downloading ForcedAligner model: $FORCED_ALIGNER_MODEL"
        python -c "
from huggingface_hub import snapshot_download
snapshot_download('${FORCED_ALIGNER_MODEL}')
print('ForcedAligner model cached successfully')
"
    fi
    echo "ForcedAligner will load on device=$FORCED_ALIGNER_DEVICE dtype=$FORCED_ALIGNER_DTYPE"
else
    GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.9}"
fi

echo "Starting Qwen3-TTS server with model: $MODEL"

SERVE_ARGS=(
    --stage-configs-path "$STAGE_CONFIGS_PATH"
    --host 0.0.0.0
    --port 8091
    --gpu-memory-utilization "$GPU_MEM_UTIL"
    --trust-remote-code
    --omni
)

if [ -n "$SERVED_MODEL_NAME" ]; then
    echo "Served model name: $SERVED_MODEL_NAME"
    SERVE_ARGS+=(--served-model-name "$SERVED_MODEL_NAME")
fi

vllm-omni serve "$MODEL" "${SERVE_ARGS[@]}" &

# Wait for the vllm server to be ready before initializing voice clones
echo "Waiting for vllm server on port 8091..."
until curl -sf http://localhost:8091/health >/dev/null 2>&1; do
    sleep 2
done
echo "vllm server is ready on port 8091."

if [[ -n "${REQUIRED_VOICES:-}" ]]; then
    echo "Initializing required voice clones in parallel..."
    PIDS=()
    for VOICE in $(echo "$REQUIRED_VOICES" | tr ',' ' '); do
        REF_AUDIO="/app/data/clone_data/${VOICE}.wav"
        REF_TEXT="/app/data/clone_data/${VOICE}.txt"

        if [[ ! -f "$REF_AUDIO" ]]; then
            echo "Missing ref-audio for $VOICE — skipping"
            continue
        fi

        CURL_ARGS=(
            -s -f -X POST "http://localhost:8091/v1/audio/voices"
            -F "audio_sample=@${REF_AUDIO}"
            -F "name=${VOICE}"
            -F "consent=user_consent"
        )

        if [[ -f "$REF_TEXT" ]]; then
            CURL_ARGS+=(-F "ref_text=$(cat "$REF_TEXT")")
        else
            echo "No ref-text for $VOICE — uploading with audio only"
        fi

        curl "${CURL_ARGS[@]}" &
        PIDS+=($!)
    done

    for PID in "${PIDS[@]}"; do
        wait "$PID" || echo "Voice init PID $PID failed"
    done
    echo "All required voice clones submitted. Waiting for readiness..."

    until curl -sf http://localhost:8091/ready >/dev/null 2>&1; do
        sleep 1
    done
    echo "Server is ready — all required voices initialized."
else
    echo "REQUIRED_VOICES not set — skipping voice clone initialization."
fi

# Keep the server running
wait