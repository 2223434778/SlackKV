#!/usr/bin/env bash
# Archived original invocations, not an automatic new budget authorization.
# Existing names must never be overwritten. Replays require new names and budget.
research_root=/media/a6000/D/ZhaoBinHe/SlackKV
code_checkout=/media/a6000/D/ZhaoBinHe/slackkv-large-model-research
runtime_source=/media/a6000/D/ZhaoBinHe/slackkv-autonomous-round4-source
model_path=/media/a6000/D/ZhaoBinHe/slackkv-models/qwen2p5-32b-5ede1c97
python_path=/media/a6000/D/ZhaoBinHe/.conda-envs/slackkv-phase2c-fix/bin/python
acquisition_path="$research_root/results/large_model_20261012/acquire3/acquisition.json"

env PYTHONPATH="$code_checkout" "$python_path" -m experiments.large_model.run --root "$research_root" --source "$runtime_source" --model "$model_path" --acquisition "$acquisition_path" --name feasibility1
env PYTHONPATH="$code_checkout" "$python_path" -m experiments.large_model.run --root "$research_root" --source "$runtime_source" --model "$model_path" --acquisition "$acquisition_path" --name oracle1 --oracle
env PYTHONPATH="$code_checkout" "$python_path" -m experiments.large_model.run --root "$research_root" --source "$runtime_source" --model "$model_path" --acquisition "$acquisition_path" --name screen_nohost1 --screen-policy nohost
env PYTHONPATH="$code_checkout" "$python_path" -m experiments.large_model.run --root "$research_root" --source "$runtime_source" --model "$model_path" --acquisition "$acquisition_path" --name screen_full1 --screen-policy full
env PYTHONPATH="$code_checkout" "$python_path" -m experiments.large_model.run --root "$research_root" --source "$runtime_source" --model "$model_path" --acquisition "$acquisition_path" --name screen_fair1 --screen-policy fair --authorized-gpu-hours 4.5

# CPU-only recomputation from lossless Git artifacts:
python -m experiments.large_model.verify_archive --directory results/large_model_20261012
python -m experiments.large_model.analyze_oracle --directory results/large_model_20261012/oracle1
python -m experiments.large_model.analyze_screen --compare results/large_model_20261012/screen_nohost1 results/large_model_20261012/screen_full1 results/large_model_20261012/screen_fair1
