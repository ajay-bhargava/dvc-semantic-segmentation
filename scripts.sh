dvc import-url s3://dvc-datasets/semantic-segmentation/tracked/pets-dataset/labelstudio-pets-dataset-export.json \
    data/labelstudio-pets-dataset.json \
    --to-remote \
    --remote pets-remote \
    --version-aware \
    --desc "LabelStudio export of Pets Dataset for Semantic Segmentation" \
    --type "json" \
    --meta created="2023-02-12"
