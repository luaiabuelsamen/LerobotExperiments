#!/bin/bash
# Convenience script to run various components from the project root

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

case "$1" in
    "calibrate")
        python scripts/calibrate_arm.py "${@:2}"
        ;;
    "record")
        python scripts/record_demos.py "${@:2}"
        ;;
    "control")
        python scripts/simple_control.py "${@:2}"
        ;;
    "train")
        python src/train_act.py "${@:2}"
        ;;
    "run")
        python scripts/run_act_policy.py "${@:2}"
        ;;
    "help"|*)
        echo "SO-ARM100 ACT Training Project Runner"
        echo ""
        echo "Usage: $0 <command> [args...]"
        echo ""
        echo "Commands:"
        echo "  calibrate    Run arm calibration"
        echo "  record       Record demonstrations"
        echo "  control      Manual control interface"
        echo "  train        Train ACT policy"
        echo "  run          Run trained policy"
        echo "  help         Show this help"
        echo ""
        echo "Examples:"
        echo "  $0 record --output data/my_dataset"
        echo "  $0 train --dataset data/recordings/dataset_20251130_142003"
        echo "  $0 run --model models/act_test/final_model.pt"
        ;;
esac