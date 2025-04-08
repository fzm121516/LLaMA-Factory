apt install tmux

conda create --name loki --clone torch
conda activate loki
pip install -e .
pip install transformers==4.50.3


conda activate torch
pip install -e .
llamafactory-cli version
