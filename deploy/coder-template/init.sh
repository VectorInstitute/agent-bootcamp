# Run project init steps
uv venv .venv
source .venv/bin/activate
uv sync --dev

# Create a Jupyter kernel for the project
uv run ipython kernel install --user --name="agent-bootcamp"

echo "Starting Jupyter lab"

# Start Jupyter JupyterLab
nohup bash -c "uv run jupyter lab --ip='*' --port=8888 --no-browser --ServerApp.token=''" &> /tmp/jupyter.log &