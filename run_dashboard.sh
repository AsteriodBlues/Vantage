#!/bin/bash
# Launch VANTAGE F1 Dashboard

echo "🏎️  Starting VANTAGE F1 Dashboard..."
echo ""

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run Streamlit
streamlit run app/dashboard.py

