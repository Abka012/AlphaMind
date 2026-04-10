#!/bin/bash
# cleanup.sh - Delete old data and models for HFT bot

echo "🧹 Cleaning up old data and models..."
echo ""

# Delete old models
if [ -d "saved_models" ]; then
    rm -f saved_models/*.pkl
    echo "✓ Deleted saved_models/*.pkl"
else
    echo "⚠ saved_models/ not found"
fi

# Delete CSV files in data/raw_ticks
if [ -d "data/raw_ticks" ]; then
    rm -f data/raw_ticks/*.csv
    echo "✓ Deleted data/raw_ticks/*.csv"
else
    echo "⚠ data/raw_ticks not found"
fi

echo ""
echo "✅ Cleanup complete!"
echo ""
echo "To download fresh data: python data/download_raw_ticks.py"
echo "To train new models: python train.py"
