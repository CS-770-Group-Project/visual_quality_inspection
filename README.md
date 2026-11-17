# visual_quality_inspection
## Project overview

Classify casting-product photographs as acceptable (0) or defective (1).
Compare logistic regression with two small CNN configurations.

## Local setup

1. Clone the repository and open it in VS Code.
2. Create a Python environment:

   python3 -m venv .venv

3. Activate it on macOS:

   source .venv/bin/activate

4. Install dependencies:

   python -m pip install -r requirements.txt

5. Open the notebook and select the project's .venv as its kernel.

The initial data preparation and baseline were run using Python 3.14.6.
Package versions are recorded in requirements.txt.

## Dataset placement

Download the original casting_512x512 collection from:
https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product

Place the image folders at:

- data/casting_512x512/def_front/
- data/casting_512x512/ok_front/

Do not use the larger, already-augmented casting_data collection.
Images are downloaded separately and are not stored in this repository.
Follow the dataset source's posted terms.

## Data audit and shared splits

All 1,300 images were readable and measured 512 × 512 pixels.
No exact duplicate image content or conflicting-label duplicates
were found. No images were excluded.

Use splits/split_manifest.csv for all models.
Do not create separate splits for CNN experiments.

- Training: 910 images
- Validation: 195 images
- Test: 195 images
- Split seed: 770
- Acceptable = 0; defective = 1

## Shared preprocessing

Convert images to grayscale, resize to 64 × 64 using bilinear
resampling, and divide pixel values by 255.

Logistic regression uses flattened pixels.
CNNs use the same prepared pixels with a channel dimension.

## Running the current notebook

Open setup_check.ipynb from the repository root.
Restart the kernel and run the cells in order.
The current notebook covers the data audit, splits, preprocessing,
and logistic regression baseline.

## Evaluation rules

Compare all three configurations using defective-class validation F1.
Break ties in favor of the simpler model.
Use a fixed defect-probability threshold of 0.5.

Select CNN checkpoints using validation results only.
Evaluate only the selected fitted model on the test set.
Do not retrain on validation data or adjust settings after test evaluation.

## Current status

Completed: data audit, fixed splits, preprocessing, and baseline.
Pending: two CNN runs, validation comparison, final test evaluation,
error analysis, and final report.

## Final submission

Submit one ZIP containing only:

- project.ipynb: combined, executed notebook with visible outputs
- report.pdf: maximum two pages, including references and contributions
- requirements.txt: package versions

Do not include images, environments, caches, credentials, or model weights.