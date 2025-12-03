# Plant Species Image Classification

**Project Overview:**

- **What:** A small image-classification project that trains a convolutional neural network to recognize many species of plants. The workspace contains a prepared dataset (train / test folders) and a baseline training notebook: `baseline_model.ipynb`.
- **Where:** This folder is `plants/` and the dataset lives under `plants/datasets/`.

**Dataset Layout:**

- `datasets/dataset/` : training images organized by class (one folder per species).
- `datasets/dataset-test/` : validation/test images organized by class.
- `datasets/dataset-user_images/` : optional user images for quick tests.

Each class folder should contain typical image files (jpg, png). If training fails due to corrupted images, run the `check_tf_read` cell in `baseline_model.ipynb` to locate problematic files.

**Notebook / Baseline Model:**

- File: `baseline_model.ipynb` (in `plants/`).
- Summary: builds a simple Sequential CNN using `tf.keras`, trains with `image_dataset_from_directory`, and plots accuracy/loss. Key settings in the notebook:
  - `IMG_SIZE = (224, 224)`
  - `BATCH_SIZE = 32`
  - `epochs = 20` (adjustable)

**Quick Start (local):**

1. Create and activate a virtual environment (recommended):
   - `python3 -m venv .venv`
   - `source .venv/bin/activate`
2. Install essential packages:
   - `pip install --upgrade pip`
   - `pip install tensorflow matplotlib jupyter`
3. Launch the notebook interface and open `baseline_model.ipynb`:
   - `jupyter lab`  (or `jupyter notebook`)
4. In the notebook, run cells top-to-bottom. The dataset paths used by the notebook are:
   - `train_dir = "datasets/dataset"`
   - `test_dir  = "datasets/dataset-test"`
