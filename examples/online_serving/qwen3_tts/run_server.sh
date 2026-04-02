#!/bin/bash
# Launch vLLM-Omni server for Qwen3-TTS models
#
# Usage:
#   ./run_server.sh                                    # Default: 0.6B CustomVoice
#   ./run_server.sh CustomVoice                        # 0.6B CustomVoice
#   ./run_server.sh VoiceDesign 1.7B                   # 1.7B VoiceDesign
#   ./run_server.sh Base 0.6B                          # 0.6B Base (voice clone)
#   ./run_server.sh /path/to/local/model               # Local model directory

set -e

ARG="${1:-CustomVoice}"
SIZE="${2:-0.6B}"

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

echo "Starting Qwen3-TTS server with model: $MODEL"

vllm-omni serve "$MODEL" \
    --stage-configs-path vllm_omni/model_executor/stage_configs/qwen3_tts.yaml \
    --host 0.0.0.0 \
    --port 8091 \
    --gpu-memory-utilization 0.9 \
    --trust-remote-code \
    --omni
