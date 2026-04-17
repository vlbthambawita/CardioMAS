cardiomas build-corpus --config examples/ollama/ptbxl_code_only.yaml --force
cardiomas check-ollama --config examples/ollama/ptbxl_code_only.yaml
cardiomas query "What labels are present in the dataset?" --config examples/ollama/ptbxl_code_only.yaml --live
cardiomas query "how many unique patients in this dataset" --config examples/ollama/ptbxl_code_only.yaml --live

#react mode
cardiomas build-corpus --config examples/ollama/ptbxl_react.yaml --force
cardiomas check-ollama --config examples/ollama/ptbxl_react.yaml
cardiomas query "What labels are present in the dataset?" --config examples/ollama/ptbxl_react.yaml --live
cardiomas query "how many unique patients in this dataset" --config examples/ollama/ptbxl_react.yaml --live