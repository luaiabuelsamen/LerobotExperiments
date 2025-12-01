#!/bin/bash
# Convenience script to run various components from the project root

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Set PYTHONPATH to include the project root so src/ modules can be imported
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

case "$1" in
    "calibrate")
        python scripts/calibrate_arm.py "${@:2}"
        ;;
    "control")
        DISPLAY=:1 python scripts/simple_control.py "${@:2}"
        ;;
    "train")
        python src/train_act.py "${@:2}"
        ;;
    "run")
        DISPLAY=:1 python scripts/run_act_policy.py "${@:2}"
        ;;
esac