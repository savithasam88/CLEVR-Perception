import json
from collections import defaultdict
from pathlib import Path


def load_clevr_scenes(scene_json_path):
    with open(scene_json_path, "r") as f:
        data = json.load(f)
    return data["scenes"]


def object_to_node_key(obj):
    """
    Canonical key for a catalog node.
    Collapse objects with identical attributes into one node.
    """
    return (
        obj["size"],
        obj["color"],
        obj["material"],
        obj["shape"],
    )


def node_key_to_dict(node_key):
    size, color, material, shape = node_key
    return {
        "size": size,
        "color": color,
        "material": material,
        "shape": shape,
    }


def build_catalog_graph(scenes, use_image_filename=True):
    """
    Build a consolidated catalog graph from CLEVR scenes.

    Nodes:
      keyed by object attributes only, collapsed across all images.

    Edges:
      keyed by (source_node_key, relation, target_node_key), collapsed
      across all images.

    Each node and edge stores an inverted index of images in which it appears.
    """
    catalog_nodes = {}
    catalog_edges = {}

    for scene in scenes:
        image_id = scene["image_filename"] if use_image_filename else scene["image_index"]
        objects = scene["objects"]
        relationships = scene["relationships"]

        # Map per-image object index -> canonical catalog node key
        obj_idx_to_node_key = {}
        image_node_keys = set()
        image_edge_keys = set()

        for obj_idx, obj in enumerate(objects):
            node_key = object_to_node_key(obj)
            obj_idx_to_node_key[obj_idx] = node_key
            image_node_keys.add(node_key)

            if node_key not in catalog_nodes:
                catalog_nodes[node_key] = {
                    "attributes": node_key_to_dict(node_key),
                    "images": set(),
                }

        # Add node inverted index entries once per image
        for node_key in image_node_keys:
            catalog_nodes[node_key]["images"].add(image_id)

        # Build collapsed edge keys for this image
        for rel_name, rel_lists in relationships.items():
            for src_idx, tgt_indices in enumerate(rel_lists):
                src_node_key = obj_idx_to_node_key[src_idx]
                for tgt_idx in tgt_indices:
                    tgt_node_key = obj_idx_to_node_key[tgt_idx]

                    edge_key = (src_node_key, rel_name, tgt_node_key)
                    image_edge_keys.add(edge_key)

                    if edge_key not in catalog_edges:
                        catalog_edges[edge_key] = {
                            "source": src_node_key,
                            "relation": rel_name,
                            "target": tgt_node_key,
                            "images": set(),
                        }

        # Add edge inverted index entries once per image
        for edge_key in image_edge_keys:
            catalog_edges[edge_key]["images"].add(image_id)

    return {
        "nodes": catalog_nodes,
        "edges": catalog_edges,
    }


def serialize_catalog_graph(catalog_graph):
    """
    Convert tuple keys + sets into JSON-serializable structures.
    """
    serialized_nodes = []
    for node_key, node_data in catalog_graph["nodes"].items():
        serialized_nodes.append({
            "id": "|".join(node_key),
            "attributes": node_data["attributes"],
            "images": sorted(node_data["images"]),
            "image_count": len(node_data["images"]),
        })

    serialized_edges = []
    for edge_key, edge_data in catalog_graph["edges"].items():
        src_key, rel_name, tgt_key = edge_key
        serialized_edges.append({
            "source": "|".join(src_key),
            "relation": rel_name,
            "target": "|".join(tgt_key),
            "images": sorted(edge_data["images"]),
            "image_count": len(edge_data["images"]),
        })

    return {
        "nodes": serialized_nodes,
        "edges": serialized_edges,
    }


def save_catalog_graph(catalog_graph, output_path):
    serialized = serialize_catalog_graph(catalog_graph)
    with open(output_path, "w") as f:
        json.dump(serialized, f, indent=2)


if __name__ == "__main__":
    scene_json_path = "/Volumes/papaya/CLEVR_v1.0/scenes/CLEVR_train_scenes.json" # This is specific for lgilpin and should be changed  
    scenes = load_clevr_scenes(scene_json_path)

    catalog_graph = build_catalog_graph(scenes)

    print(f"Catalog nodes: {len(catalog_graph['nodes'])}")
    print(f"Catalog edges: {len(catalog_graph['edges'])}")

    # Show a few examples
    first_nodes = list(catalog_graph["nodes"].items())[:5]
    print("\nExample nodes:")
    for node_key, node_data in first_nodes:
        print(node_key, "appears in", len(node_data["images"]), "images")

    first_edges = list(catalog_graph["edges"].items())[:5]
    print("\nExample edges:")
    for edge_key, edge_data in first_edges:
        print(edge_key, "appears in", len(edge_data["images"]), "images")

    save_catalog_graph(catalog_graph, "clevr_catalog_graph.json")
    print("\nSaved catalog graph to clevr_catalog_graph.json")
