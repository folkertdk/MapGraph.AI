# MapGraph.AI

MoE-based road segmentation and graph extraction from aerial imagery.

This completed starter version focuses on the first coding milestone: **make the environment run, load image/mask data, visualize one example pair, and train a simple segmentation baseline**. After that, it includes a simple graph-extraction script and a future-ready MoE model branch so the project can be extended toward the proposal.

## Project goal

Convert aerial images into structured road graphs:

1. aerial image → road segmentation mask
2. road mask → skeletonized road centerline
3. skeleton → graph with nodes and edges
4. later: compare normal baseline segmentation vs Mixture-of-Experts segmentation

The proposal describes the wider direction as MapGraph AI: using road segmentation, graph conversion, and eventually a Mixture-of-Experts router for different city types such as urban, suburban, and rural.

## What was changed from the starter GitHub folder

The original zip had empty source files. I filled in the project as follows:

```text
MapGraph.AI-main/
  configs/
    baseline.yaml          # paths, model, training, and output settings
  data/
    sample/
      images/              # two example satellite images copied from uploaded files
      masks/               # two example road masks copied from uploaded files
  outputs/                 # generated visualizations/checkpoints/predictions/graphs
  src/
    __init__.py            # allows running scripts with python -m src.script_name
    dataset.py             # loads and pairs images + masks
    graph_extract.py       # converts a road mask into a skeleton graph
    model.py               # Simple U-Net baseline + simple MoE-U-Net
    predict.py             # runs trained model and saves predicted masks
    train.py               # trains the baseline/MoE segmentation model
    utils.py               # config, seed, device, folder helpers
    visualize_pair.py      # loads and visualizes one image/mask pair
  requirements.txt
  README.md
```

No files need to be removed. The important additions are `model.py`, `utils.py`, `visualize_pair.py`, and `src/__init__.py`. The previously empty `dataset.py`, `train.py`, `predict.py`, and `graph_extract.py` are now functional.

## Setup

From inside the project folder:

```bash
python -m venv .venv
source .venv/bin/activate      # Mac/Linux
# .venv\Scripts\activate      # Windows PowerShell

pip install -r requirements.txt
```

## Step 1: Verify that image/mask loading works

```bash
python -m src.visualize_pair --config configs/baseline.yaml
```

Expected result:

```text
Saved visualization to outputs/example_pair.png
Image path: data/sample/images/...
Mask path:  data/sample/masks/...
```

Open:

```text
outputs/example_pair.png
```

You should see one aerial image and one road segmentation mask.

## Step 2: Train the simple segmentation baseline

```bash
python -m src.train --config configs/baseline.yaml
```

This saves:

```text
outputs/baseline_run/last.pt
outputs/baseline_run/best.pt
```

Important note: the included sample data is only for checking that the pipeline works. It is too small to train a strong model. For real training, update these fields in `configs/baseline.yaml`:

```yaml
data:
  image_dir: path/to/full/images
  mask_dir: path/to/full/masks
  img_size: 512
  # optional: increase train.num_threads if you have a strong CPU
```

## Step 3: Predict road masks

```bash
python -m src.predict --config configs/baseline.yaml --checkpoint outputs/baseline_run/best.pt
```

Predicted masks will be saved in:

```text
outputs/predictions/
```

## Step 4: Extract a graph from a mask

You can test graph extraction using a ground-truth sample mask:

```bash
python -m src.graph_extract --mask data/sample/masks/khartoum_mask1.png
```

Or with a predicted mask after running inference:

```bash
python -m src.graph_extract --mask outputs/predictions/khartoum_img21_pred_mask.png
```

Outputs:

```text
outputs/graphs/*_graph.json
outputs/graphs/*_graph.png
```

## How the code works

### `src/dataset.py`

Loads RGB aerial images and binary road masks. It supports `.png`, `.jpg`, `.jpeg`, `.tif`, and `.tiff`. Because the uploaded examples may not be exact image/mask pairs, the config uses:

```yaml
strict_pairing: false
```

That means the loader pairs files by sorted order. For the full real dataset, if your filenames match, you can set:

```yaml
strict_pairing: true
```

### `src/model.py`

Includes:

- `SimpleUNet`: current baseline model.
- `MoEUNet`: future Mixture-of-Experts version.
- `GatingNetwork`: produces expert probabilities.

The baseline output is one logit per pixel. During training, `BCEWithLogitsLoss` learns road vs background.

### `src/train.py`

Builds the dataset, splits it into train/validation, trains the model, prints loss and Dice score, and saves checkpoints.

### `src/predict.py`

Loads a checkpoint, applies sigmoid to model logits, thresholds the result, and saves binary predicted road masks.

### `src/graph_extract.py`

Converts a binary road mask into a graph:

1. threshold mask,
2. skeletonize road pixels,
3. identify endpoints and intersections as nodes,
4. trace skeleton paths as edges,
5. save graph JSON and a visual PNG.

## Switching to the MoE branch later

In `configs/baseline.yaml`, change:

```yaml
model:
  architecture: moe
  num_experts: 3
```

Then train again:

```bash
python -m src.train --config configs/baseline.yaml
```

This gives you a simple MoE baseline that can later be improved with proper urban/suburban/rural labels or a better router.

## Recommended next development steps

1. Replace sample data with the full SpaceNet Roads dataset.
2. Train the baseline on more tiles.
3. Save baseline metrics: Dice, IoU, precision, recall.
4. Improve graph extraction with pruning to remove tiny noisy branches.
5. Add environment labels or heuristics for urban/suburban/rural routing.
6. Compare baseline U-Net vs MoE-U-Net.
