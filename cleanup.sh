#!/bin/bash
# cleanup.sh - Delete old data and models for MFT bot

echo "🧹 Cleaning up old data and models..."
echo ""

# Delete old models
if [ -d "saved_models" ]; then
    rm -f saved_models/*.pkl
    echo "✓ Deleted saved_models/*.pkl"
else
    echo "⚠ saved_models/*.pkl not found"
fi

# Delete CSV files in data/raw_ticks
if [ -d "data/raw_ticks" ]; then
    rm -f data/raw_ticks/*.csv
    echo "✓ Deleted data/raw_ticks/*.csv"
else
    echo "⚠ data/raw_ticks not found"
fi

# Delete Json log in saved_models
if [ -d "saved_models" ]; then
    rm -f saved_models/*.json
    echo "✓ Deleted saved_models/*.json"
else
    echo "⚠ saved_models/*.json not found"
fi

echo ""
echo "✅ Cleanup complete!"
echo ""
echo "To download fresh data: python data/download_raw_ticks.py"
echo "To train new models: python train.py"
