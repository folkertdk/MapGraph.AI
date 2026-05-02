"""Convert a binary road mask into a cleaner road graph.

This version is more robust than the first simple graph extractor.

Why the old version produced too many nodes:
- Road masks are thick white lines.
- Skeletonization creates small stair-step artifacts.
- Every tiny bend or broken pixel can accidentally become a node.
- Border-touching roads can create many fake endpoints.
- Small prediction noise creates many tiny disconnected graph pieces.

This improved version does:
1. Binary mask cleanup.
2. Skeletonization.
3. Spur pruning to remove tiny road stubs.
4. Junction/end-point detection.
5. Node clustering so one intersection becomes one graph node.
6. Edge tracing between node clusters.
7. Short-edge filtering.
8. Cleaner graph visualization.

Expected result:
Instead of hundreds of meaningless nodes, you should get fewer, cleaner nodes
representing intersections and road endpoints.
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path
from typing import Dict, List, Set, Tuple

import cv2
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from skimage.morphology import skeletonize

from .utils import ensure_dir

Pixel = Tuple[int, int]


def load_binary_mask(
    path: str | Path,
    threshold: int = 127,
    max_size: int | None = 512,
) -> np.ndarray:
    """Load a mask as a boolean road/background image.

    White pixels are treated as roads.
    Black pixels are treated as background.
    """
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {path}")

    if max_size and max(mask.shape) > max_size:
        scale = max_size / max(mask.shape)
        new_w = max(1, int(mask.shape[1] * scale))
        new_h = max(1, int(mask.shape[0] * scale))

        mask = cv2.resize(
            mask,
            (new_w, new_h),
            interpolation=cv2.INTER_NEAREST,
        )

    binary = mask > threshold

    road_fraction = float(binary.mean())

    print(f"Road pixel fraction before cleanup: {road_fraction:.2%}")

    if road_fraction == 0.0:
        print("Warning: mask is fully black. Graph will be empty.")

    if road_fraction > 0.80:
        print("Warning: mask is mostly white. Check threshold or inverted mask.")

    return binary


def clean_binary_mask(
    binary: np.ndarray,
    min_component_area: int = 30,
    close_kernel_size: int = 3,
) -> np.ndarray:
    """Remove tiny blobs and close small gaps before skeletonization."""
    mask = binary.astype(np.uint8)

    if close_kernel_size > 0:
        kernel = np.ones((close_kernel_size, close_kernel_size), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8,
    )

    cleaned = np.zeros_like(mask, dtype=np.uint8)

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]

        if area >= min_component_area:
            cleaned[labels == label] = 1

    return cleaned.astype(bool)


def neighbors(pixel: Pixel, shape: Tuple[int, int]) -> List[Pixel]:
    """Return 8-connected neighbors inside image bounds."""
    y, x = pixel
    h, w = shape

    result: List[Pixel] = []

    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dy == 0 and dx == 0:
                continue

            ny = y + dy
            nx_ = x + dx

            if 0 <= ny < h and 0 <= nx_ < w:
                result.append((ny, nx_))

    return result


def skeleton_degree(pixel: Pixel, skeleton: np.ndarray) -> int:
    """Count skeleton neighbors around a pixel."""
    return sum(bool(skeleton[n]) for n in neighbors(pixel, skeleton.shape))


def prune_skeleton_spurs(
    skeleton: np.ndarray,
    iterations: int = 12,
) -> np.ndarray:
    """Remove short dangling branches from a skeleton.

    This repeatedly removes endpoint pixels. It helps remove small prediction
    noise and small side branches that create too many graph nodes.

    Do not set this too high, or real dead-end roads may disappear.
    """
    pruned = skeleton.copy().astype(bool)

    for _ in range(iterations):
        endpoints = []

        ys, xs = np.where(pruned)

        for y, x in zip(ys, xs):
            if skeleton_degree((int(y), int(x)), pruned) == 1:
                endpoints.append((int(y), int(x)))

        if not endpoints:
            break

        for p in endpoints:
            pruned[p] = False

    return pruned


def find_node_pixels(
    skeleton: np.ndarray,
    ignore_border: int = 2,
) -> Set[Pixel]:
    """Find pixels that should become graph node candidates.

    A graph node candidate is:
    - endpoint: degree 1
    - junction/intersection: degree 3 or more

    Degree 2 pixels are normal road-centerline pixels and should not become nodes.
    """
    h, w = skeleton.shape

    node_pixels: Set[Pixel] = set()

    ys, xs = np.where(skeleton)

    for y, x in zip(ys, xs):
        y = int(y)
        x = int(x)

        if (
            y < ignore_border
            or x < ignore_border
            or y >= h - ignore_border
            or x >= w - ignore_border
        ):
            continue

        degree = skeleton_degree((y, x), skeleton)

        if degree == 1 or degree >= 3:
            node_pixels.add((y, x))

    return node_pixels


def connected_components_from_pixels(
    pixels: Set[Pixel],
    shape: Tuple[int, int],
) -> List[Set[Pixel]]:
    """Group nearby node pixels into connected clusters.

    One real intersection often appears as a small blob of several skeleton
    pixels. This function groups that blob into one node.
    """
    remaining = set(pixels)
    components: List[Set[Pixel]] = []

    while remaining:
        start = remaining.pop()
        component = {start}
        queue = deque([start])

        while queue:
            current = queue.popleft()

            for nb in neighbors(current, shape):
                if nb in remaining:
                    remaining.remove(nb)
                    component.add(nb)
                    queue.append(nb)

        components.append(component)

    return components


def cluster_nodes(
    node_pixels: Set[Pixel],
    skeleton_shape: Tuple[int, int],
    merge_radius: int = 6,
) -> Tuple[nx.Graph, Dict[Pixel, int]]:
    """Turn node pixels into graph nodes and assign nearby pixels to clusters.

    merge_radius makes the graph cleaner by merging node candidates that are
    close to each other around the same intersection.
    """
    components = connected_components_from_pixels(node_pixels, skeleton_shape)

    centers = []

    for component in components:
        arr = np.array(list(component), dtype=np.float32)
        cy, cx = arr.mean(axis=0)
        centers.append((float(cy), float(cx), component))

    # Merge components whose centroids are close.
    parent = list(range(len(centers)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]

        return i

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)

        if ra != rb:
            parent[rb] = ra

    for i in range(len(centers)):
        yi, xi, _ = centers[i]

        for j in range(i + 1, len(centers)):
            yj, xj, _ = centers[j]
            dist = ((yi - yj) ** 2 + (xi - xj) ** 2) ** 0.5

            if dist <= merge_radius:
                union(i, j)

    merged: Dict[int, Set[Pixel]] = {}

    for i, (_, _, component) in enumerate(centers):
        root = find(i)

        if root not in merged:
            merged[root] = set()

        merged[root].update(component)

    graph = nx.Graph()
    pixel_to_node: Dict[Pixel, int] = {}

    for node_id, component in enumerate(merged.values()):
        arr = np.array(list(component), dtype=np.float32)
        cy, cx = arr.mean(axis=0)

        graph.add_node(
            node_id,
            y=float(cy),
            x=float(cx),
            pixel_count=len(component),
        )

        for p in component:
            pixel_to_node[p] = node_id

    return graph, pixel_to_node


def assign_nearby_skeleton_pixels_to_nodes(
    skeleton: np.ndarray,
    pixel_to_node: Dict[Pixel, int],
    radius: int = 3,
) -> Dict[Pixel, int]:
    """Assign skeleton pixels near a node cluster to that node.

    This prevents edge tracing from getting stuck inside small junction blobs.
    """
    expanded = dict(pixel_to_node)

    for node_pixel, node_id in list(pixel_to_node.items()):
        y, x = node_pixel

        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                ny = y + dy
                nx_ = x + dx

                if (
                    0 <= ny < skeleton.shape[0]
                    and 0 <= nx_ < skeleton.shape[1]
                    and skeleton[ny, nx_]
                ):
                    expanded[(ny, nx_)] = node_id

    return expanded


def edge_key(a: Pixel, b: Pixel) -> Tuple[Pixel, Pixel]:
    """Undirected edge key between two adjacent pixels."""
    return tuple(sorted([a, b]))  # type: ignore[return-value]


def trace_edges(
    skeleton: np.ndarray,
    graph: nx.Graph,
    pixel_to_node: Dict[Pixel, int],
    min_edge_length: int = 8,
) -> nx.Graph:
    """Trace skeleton paths between graph nodes."""
    road_pixels: Set[Pixel] = set(map(tuple, np.argwhere(skeleton)))
    visited_pixel_edges: Set[Tuple[Pixel, Pixel]] = set()

    # Start tracing from every pixel assigned to a graph node.
    for start_pixel, start_node in list(pixel_to_node.items()):
        if start_pixel not in road_pixels:
            continue

        for nb in neighbors(start_pixel, skeleton.shape):
            if nb not in road_pixels:
                continue

            if edge_key(start_pixel, nb) in visited_pixel_edges:
                continue

            path = [start_pixel, nb]
            previous = start_pixel
            current = nb

            visited_pixel_edges.add(edge_key(previous, current))

            end_node = None

            while True:
                if current in pixel_to_node:
                    end_node = pixel_to_node[current]
                    break

                next_candidates = [
                    p
                    for p in neighbors(current, skeleton.shape)
                    if p in road_pixels and p != previous
                ]

                # Dead end not assigned as node.
                if not next_candidates:
                    break

                # Prefer unvisited continuation if available.
                unvisited = [
                    p
                    for p in next_candidates
                    if edge_key(current, p) not in visited_pixel_edges
                ]

                if unvisited:
                    nxt = unvisited[0]
                else:
                    break

                visited_pixel_edges.add(edge_key(current, nxt))

                previous = current
                current = nxt
                path.append(current)

                # Safety guard for malformed skeletons.
                if len(path) > skeleton.size:
                    break

            if end_node is None:
                continue

            if end_node == start_node:
                continue

            if len(path) < min_edge_length:
                continue

            if graph.has_edge(start_node, end_node):
                old_length = graph[start_node][end_node].get("length", 0)

                # Keep the longer path if duplicate edge is discovered.
                if len(path) > old_length:
                    graph[start_node][end_node]["length"] = len(path)
                    graph[start_node][end_node]["pixels"] = [
                        (int(y), int(x)) for y, x in path
                    ]
            else:
                graph.add_edge(
                    start_node,
                    end_node,
                    length=len(path),
                    pixels=[(int(y), int(x)) for y, x in path],
                )

    return graph


def remove_isolated_and_tiny_components(
    graph: nx.Graph,
    min_component_nodes: int = 2,
) -> nx.Graph:
    """Remove graph components that are too small to be useful."""
    cleaned = graph.copy()

    for component in list(nx.connected_components(cleaned)):
        if len(component) < min_component_nodes:
            cleaned.remove_nodes_from(component)

    # Relabel nodes to keep IDs clean.
    cleaned = nx.convert_node_labels_to_integers(cleaned)

    return cleaned


def mask_to_graph(
    binary: np.ndarray,
    min_component_area: int = 30,
    close_kernel_size: int = 3,
    spur_prune_iterations: int = 12,
    node_merge_radius: int = 6,
    node_expand_radius: int = 3,
    min_edge_length: int = 8,
    ignore_border: int = 2,
) -> Tuple[np.ndarray, nx.Graph]:
    """Full mask-to-graph pipeline."""
    cleaned = clean_binary_mask(
        binary,
        min_component_area=min_component_area,
        close_kernel_size=close_kernel_size,
    )

    skeleton = skeletonize(cleaned)
    skeleton = prune_skeleton_spurs(
        skeleton,
        iterations=spur_prune_iterations,
    )

    node_pixels = find_node_pixels(
        skeleton,
        ignore_border=ignore_border,
    )

    if not node_pixels:
        return skeleton, nx.Graph()

    graph, pixel_to_node = cluster_nodes(
        node_pixels,
        skeleton_shape=skeleton.shape,
        merge_radius=node_merge_radius,
    )

    expanded_pixel_to_node = assign_nearby_skeleton_pixels_to_nodes(
        skeleton,
        pixel_to_node,
        radius=node_expand_radius,
    )

    graph = trace_edges(
        skeleton,
        graph,
        expanded_pixel_to_node,
        min_edge_length=min_edge_length,
    )

    graph = remove_isolated_and_tiny_components(
        graph,
        min_component_nodes=2,
    )

    return skeleton, graph


def save_graph_json(graph: nx.Graph, out_path: str | Path) -> None:
    """Save graph nodes and edges as JSON."""
    data = {
        "nodes": [
            {
                "id": int(node),
                "y": float(attrs["y"]),
                "x": float(attrs["x"]),
            }
            for node, attrs in graph.nodes(data=True)
        ],
        "edges": [
            {
                "source": int(u),
                "target": int(v),
                "length": int(attrs.get("length", 0)),
            }
            for u, v, attrs in graph.edges(data=True)
        ],
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def save_graph_visualization(
    skeleton: np.ndarray,
    graph: nx.Graph,
    out_path: str | Path,
) -> None:
    """Save graph visualization."""
    plt.figure(figsize=(8, 8))
    plt.imshow(skeleton, cmap="gray")

    for u, v, attrs in graph.edges(data=True):
        pixels = attrs.get("pixels", [])

        if pixels:
            ys = [p[0] for p in pixels]
            xs = [p[1] for p in pixels]
            plt.plot(xs, ys, linewidth=1.5)

    if graph.number_of_nodes() > 0:
        xs = [attrs["x"] for _, attrs in graph.nodes(data=True)]
        ys = [attrs["y"] for _, attrs in graph.nodes(data=True)]
        plt.scatter(xs, ys, s=18)

    plt.title(
        f"Extracted graph: "
        f"{graph.number_of_nodes()} nodes, "
        f"{graph.number_of_edges()} edges"
    )

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--mask", required=True, help="Path to binary road mask")
    parser.add_argument("--out_dir", default="outputs/graphs")
    parser.add_argument("--max_size", type=int, default=512)

    parser.add_argument("--min_component_area", type=int, default=30)
    parser.add_argument("--close_kernel_size", type=int, default=3)
    parser.add_argument("--spur_prune_iterations", type=int, default=12)
    parser.add_argument("--node_merge_radius", type=int, default=6)
    parser.add_argument("--node_expand_radius", type=int, default=3)
    parser.add_argument("--min_edge_length", type=int, default=8)
    parser.add_argument("--ignore_border", type=int, default=2)

    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)

    binary = load_binary_mask(
        args.mask,
        max_size=args.max_size if args.max_size > 0 else None,
    )

    skeleton, graph = mask_to_graph(
        binary=binary,
        min_component_area=args.min_component_area,
        close_kernel_size=args.close_kernel_size,
        spur_prune_iterations=args.spur_prune_iterations,
        node_merge_radius=args.node_merge_radius,
        node_expand_radius=args.node_expand_radius,
        min_edge_length=args.min_edge_length,
        ignore_border=args.ignore_border,
    )

    stem = Path(args.mask).stem

    json_path = out_dir / f"{stem}_graph.json"
    png_path = out_dir / f"{stem}_graph.png"

    save_graph_json(graph, json_path)
    save_graph_visualization(skeleton, graph, png_path)

    print(f"Saved graph JSON: {json_path}")
    print(f"Saved graph visualization: {png_path}")
    print(f"Nodes: {graph.number_of_nodes()}, edges: {graph.number_of_edges()}")


if __name__ == "__main__":
    main()