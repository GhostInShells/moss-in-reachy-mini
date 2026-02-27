source .venv/bin/activate

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)


uv sync --all-extras
streamlit run "$SCRIPT_DIR/memory_ui.py"