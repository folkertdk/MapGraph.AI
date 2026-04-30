# MapGraph.AI
MoE-based road segmentation and graph extraction from aerial imagery.


# README

MapGraph.AI is a computer vision project for extracting road networks from aerial images and converting them into graph-like map structures.

The project starts with a simple road segmentation baseline and then explores a Mixture-of-Experts (MoE) approach, where specialized models handle different environment types such as urban, suburban, and rural areas.

## Goal

Convert aerial images into structured road graphs:

- pixels → road segmentation mask
- mask → skeletonized road centerlines
- centerlines → graph with nodes and edges

## Planned Pipeline

1. Load aerial image dataset
2. Train baseline road segmentation model
3. Generate predicted road masks
4. Convert masks into skeletons
5. Extract graph structure
6. Compare baseline model with MoE-based model

## Project Structure

```text
MapGraph.AI/
  configs/
  data/
  notebooks/
  outputs/
  src/
  README.md
  requirements.txt
