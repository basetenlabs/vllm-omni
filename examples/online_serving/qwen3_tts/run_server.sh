#!/bin/bash
# Launch vLLM-Omni server for Qwen3-TTS models
#
# Usage:
#   ./run_server.sh                                                        # Default: 1.7B CustomVoice, ttfa profile
#   ./run_server.sh CustomVoice                                            # 1.7B CustomVoice
#   ./run_server.sh VoiceDesign 1.7B                                       # 1.7B VoiceDesign
#   ./run_server.sh Base 0.6B                                              # 0.6B Base (voice clone)
#   ./run_server.sh /path/to/local/model                                   # Local model directory
#   SERVED_MODEL_NAME=my-tts ./run_server.sh                               # Custom served model name
#
# Stage-config profiles (select with --profile, or set STAGE_CONFIGS_PATH directly):
#   --profile ttfa                 # Low TTFA at ~25 concurrent streams (default)
#   --profile ttfa_32              # Low TTFA at ~32 concurrent streams
#   --profile high_concurrency     # Aggregate throughput at 32–64 concurrent streams
#   STAGE_CONFIGS_PATH=./custom.yaml ./run_server.sh   # Arbitrary custom config
#
# Timestamps mode (word-level alignment via Qwen3-ForcedAligner-0.6B):
#   ./run_server.sh --timestamps CustomVoice 1.7B
#   FORCED_ALIGNER_MODEL=/local/path ./run_server.sh --timestamps
#   FORCED_ALIGNER_DEVICE=cuda:1 ./run_server.sh --timestamps

set -e

ENABLE_TIMESTAMPS=false
PROFILE="${PROFILE:-ttfa}"
while [ $# -gt 0 ]; do
    case "$1" in
        --timestamps)
            ENABLE_TIMESTAMPS=true
            shift
            ;;
        --profile)
            PROFILE="$2"
            shift 2
            ;;
        --profile=*)
            PROFILE="${1#*=}"
            shift
            ;;
        *)
            break
            ;;
    esac
done

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

if [ -z "${STAGE_CONFIGS_PATH:-}" ]; then
    # When --timestamps is set, the ForcedAligner model lives in the API
    # server process and shares GPU 0 with the talker/code2wav. Swap to
    # the timestamps-aware variant which leaves ~10+ GiB of headroom for
    # the aligner's forward pass (prevents OOM at real audio lengths).
    case "$PROFILE" in
        ttfa|ttfa_25)
            if [ "$ENABLE_TIMESTAMPS" = true ]; then
                STAGE_CONFIGS_PATH="vllm_omni/model_executor/stage_configs/qwen3_tts_ttfa_timestamps.yaml"
            else
                STAGE_CONFIGS_PATH="vllm_omni/model_executor/stage_configs/qwen3_tts_ttfa.yaml"
            fi
            ;;
        ttfa_32)
            if [ "$ENABLE_TIMESTAMPS" = true ]; then
                STAGE_CONFIGS_PATH="vllm_omni/model_executor/stage_configs/qwen3_tts_ttfa_32_timestamps.yaml"
            else
                STAGE_CONFIGS_PATH="vllm_omni/model_executor/stage_configs/qwen3_tts_ttfa_32.yaml"
            fi
            ;;
        high_concurrency|hc)
            STAGE_CONFIGS_PATH="vllm_omni/model_executor/stage_configs/qwen3_tts_high_concurrency.yaml"
            ;;
        *)
            echo "Unknown profile: $PROFILE"
            echo "Supported: ttfa (alias: ttfa_25), ttfa_32, high_concurrency (alias: hc)"
            exit 1
            ;;
    esac

    if [ "$ENABLE_TIMESTAMPS" = true ] && [ ! -f "$STAGE_CONFIGS_PATH" ]; then
        echo "WARNING: timestamps-aware config $STAGE_CONFIGS_PATH not found; falling back to non-timestamps variant."
        echo "         The aligner may OOM on first real-length forward pass. See TTFA config comments."
        case "$PROFILE" in
            ttfa|ttfa_25)  STAGE_CONFIGS_PATH="vllm_omni/model_executor/stage_configs/qwen3_tts_ttfa.yaml" ;;
            ttfa_32)       STAGE_CONFIGS_PATH="vllm_omni/model_executor/stage_configs/qwen3_tts_ttfa_32.yaml" ;;
        esac
    fi
fi
echo "Using stage config: $STAGE_CONFIGS_PATH (profile=$PROFILE, timestamps=$ENABLE_TIMESTAMPS)"

if [ "$ENABLE_TIMESTAMPS" = true ]; then
    export FORCED_ALIGNER_MODEL="${FORCED_ALIGNER_MODEL:-Qwen/Qwen3-ForcedAligner-0.6B}"
    export FORCED_ALIGNER_DEVICE="${FORCED_ALIGNER_DEVICE:-cuda:0}"
    export FORCED_ALIGNER_DTYPE="${FORCED_ALIGNER_DTYPE:-bfloat16}"

    # When the aligner co-resides on GPU 0 with the talker's CUDA graph
    # pools and the code2wav KV pool, fragmentation between PyTorch's
    # general allocator and the private graph pools can starve the
    # aligner's forward allocation even when nominal free memory looks
    # healthy. expandable_segments lets PyTorch grow segments dynamically
    # instead of requiring contiguous blocks upfront.
    export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

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