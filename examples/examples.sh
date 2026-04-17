cardiomas build-corpus --config examples/ollama/runtime_ptbxl.yaml --force
cardiomas check-ollama --config examples/ollama/runtime_ptbxl.yaml
cardiomas query "What labels are present in the dataset?" --config examples/ollama/runtime_ptbxl.yaml --live