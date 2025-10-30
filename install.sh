uv sync
uv run maturin develop --release --uv
uv pip install torch-cluster==1.6.3 -f https://data.pyg.org/whl/torch-2.7.0+cu126.html --no-build-isolation
uv pip install torch-spline-conv==1.2.2 -f https://data.pyg.org/whl/torch-2.7.0+cu126.html --no-build-isolation