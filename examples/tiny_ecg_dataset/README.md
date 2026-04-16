# Tiny ECG Dataset Example

This miniature fixture is included for the organization-style workflow introduced from `todo_16042025.md`.

Use it to exercise the new path without running expensive jobs:

```bash
cardiomas organize --config examples/tiny_ecg_dataset/organization_config.yaml

# or pass values directly
cardiomas organize examples/tiny_ecg_dataset \
  --dataset-name tiny-ecg-demo \
  --approve
```

The workflow writes reusable knowledge, coding, cardiology, and testing artifacts under `organization_output/`.
