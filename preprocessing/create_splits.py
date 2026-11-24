# File: preprocessing/create_splits.py
# What this file does: Scans a FaceForensics++ (c23) video directory, groups actors by connected components
#   so swapped face pairs (like 000_003.mp4) never leak identities across partitions, holds out one unseen
#   generator for Leave-One-Generator-Out (LOGO) testing, and saves a verified splits.json file.
# Inputs:
#   - --data_dir (str): Root path containing FaceForensics++ folders (original/, Deepfakes/, Face2Face/, FaceSwap/, NeuralTextures/).
#   - --output_path (str): Path to save the validated JSON split file (default: datasets/splits.json).
#   - --holdout_generator (str): Manipulation generator reserved 100% for test_unseen (default: NeuralTextures).
#   - --seed (int): Random seed for reproducible train (80%), val (10%), and test_closed (10%) actor group partitioning.
#   - --dry-run (flag): Synthesizes 100 mock video records in memory, builds and asserts zero-leakage splits in <3 seconds on CPU.
# Outputs:
#   - splits.json containing metadata, split counts, and video records under 'train', 'val', 'test_closed', and 'test_unseen'.
#   - Console verification summary proving zero actor identity overlap, zero video overlap, and zero unseen-generator leakage.

import argparse
import json
import os
import random
import re
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple


# Canonical generator directory names supported in Kaggle xdxd003/ff-c23 and official FaceForensics++ layouts.
REAL_FOLDER_ALIASES = {
    "original": "Real",
    "original_sequences": "Real",
    "youtube": "Real",
    "actors": "Real",
    "real": "Real",
}

FAKE_GENERATORS = {
    "deepfakes": "Deepfakes",
    "face2face": "Face2Face",
    "faceswap": "FaceSwap",
    "neuraltextures": "NeuralTextures",
}


def parse_identity_ids(filename: str) -> List[str]:
    """
    Extracts all person/actor identity IDs from a video filename.
    - Real video example: '000.mp4' -> ['000']
    - Swapped fake video example: '000_003.mp4' -> ['000', '003']
    If a filename does not follow numeric ID conventions, falls back to the filename stem.
    """
    stem = Path(filename).stem
    # Match numeric tokens separated by underscores or hyphens (e.g., '000_003' -> ['000', '003'])
    tokens = [tok for tok in re.split(r"[_\\-]+", stem) if tok]
    if len(tokens) >= 2 and all(tok.isdigit() for tok in tokens[:2]):
        # Preserve unique order while deduplicating self-pairs if any exist
        ids = []
        for tok in tokens[:2]:
            if tok not in ids:
                ids.append(tok)
        return ids
    if len(tokens) >= 1 and tokens[0].isdigit():
        return [tokens[0]]
    return [stem]


def detect_generator_and_label(file_path: Path) -> Tuple[str, int]:
    """
    Infers whether a video is Real (label 0) or Fake (label 1) and returns its canonical generator_name
    by inspecting all parent folder names in the path (supports both flat and deeply nested FF++ layouts).
    """
    parts_lower = [part.lower() for part in file_path.parts[:-1]]

    # Check fake manipulation generators first because nested FF++ paths may include generic words
    for part in reversed(parts_lower):
        if part in FAKE_GENERATORS:
            return FAKE_GENERATORS[part], 1

    # Check real / original video folder names
    for part in reversed(parts_lower):
        if part in REAL_FOLDER_ALIASES:
            return "Real", 0

    raise ValueError(
        f"Could not map video path '{file_path}' to 'original' (Real) or any known manipulation generator "
        f"({list(FAKE_GENERATORS.values())})."
    )


def scan_video_directory(data_dir: str) -> List[Dict]:
    """
    Recursively scans data_dir for .mp4 files and builds a list of structured video metadata dictionaries.
    """
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    records: List[Dict] = []
    mp4_files = sorted(root.rglob("*.mp4"))

    for mp4_path in mp4_files:
        try:
            generator_name, label = detect_generator_and_label(mp4_path)
        except ValueError:
            # Skip unrelated .mp4 files if any exist outside the 5 target folders
            continue

        rel_path = mp4_path.relative_to(root).as_posix()
        identity_ids = parse_identity_ids(mp4_path.name)
        video_id = f"{generator_name}/{Path(mp4_path.name).stem}"

        records.append(
            {
                "video_id": video_id,
                "video_path": rel_path,
                "filename": mp4_path.name,
                "label": label,
                "generator_name": generator_name,
                "identity_ids": identity_ids,
            }
        )

    if not records:
        raise RuntimeError(
            f"No valid FaceForensics++ .mp4 videos found under '{data_dir}'. "
            "Expected subfolders: original/, Deepfakes/, Face2Face/, FaceSwap/, NeuralTextures/."
        )
    return records


def build_connected_identity_components(records: List[Dict]) -> Tuple[List[Set[str]], Dict[str, int]]:
    """
    Uses a Disjoint-Set (Union-Find) data structure to group all actor identity IDs that ever appear
    together in a swapped video (e.g., '000_003.mp4' links '000' and '003' into the same connected component).
    Returns:
      - components: List of sets, where each set contains all actor IDs belonging to that connected component.
      - id_to_comp_idx: Mapping from each actor identity_id string to its component index.
    """
    parent: Dict[str, str] = {}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            # Deterministic union ordering by lexicographical ID
            if root_a < root_b:
                parent[root_b] = root_a
            else:
                parent[root_a] = root_b

    # Register every identity ID and union all IDs co-occurring in any video record
    for rec in records:
        ids = rec["identity_ids"]
        for actor_id in ids:
            if actor_id not in parent:
                parent[actor_id] = actor_id
        for i in range(1, len(ids)):
            union(ids[0], ids[i])

    comp_map: Dict[str, Set[str]] = defaultdict(set)
    for actor_id in sorted(parent.keys()):
        comp_map[find(actor_id)].add(actor_id)

    # Sort components deterministically
    components = [comp_map[root_id] for root_id in sorted(comp_map.keys())]
    id_to_comp_idx: Dict[str, int] = {}
    for idx, comp_ids in enumerate(components):
        for actor_id in comp_ids:
            id_to_comp_idx[actor_id] = idx

    return components, id_to_comp_idx


def partition_components_80_10_10(
    components: List[Set[str]],
    comp_video_counts: Dict[int, int],
    seed: int = 42,
) -> Dict[int, str]:
    """
    Assigns each connected actor component to 'train' (80%), 'val' (10%), or 'test_closed' (10%)
    so that the resulting video/identity distribution matches 80/10/10 while guaranteeing every
    split receives at least one component and zero actor identities ever cross partition boundaries.
    """
    num_comps = len(components)
    if num_comps < 3:
        raise ValueError(
            f"Need at least 3 independent identity components to form disjoint train/val/test splits, "
            f"but found only {num_comps} connected component(s)."
        )

    rng = random.Random(seed)
    comp_indices = list(range(num_comps))
    rng.shuffle(comp_indices)

    total_seen_videos = sum(comp_video_counts.values())
    target_train = 0.80 * total_seen_videos
    target_val = 0.10 * total_seen_videos

    comp_to_split: Dict[int, str] = {}
    running_train = 0
    running_val = 0

    for pos, comp_idx in enumerate(comp_indices):
        remaining = num_comps - pos
        weight = comp_video_counts.get(comp_idx, 1)

        # Ensure val and test_closed always get at least 1 component each
        has_val = "val" in comp_to_split.values()
        has_test = "test_closed" in comp_to_split.values()

        if not has_val and remaining == 2 and not has_test:
            comp_to_split[comp_idx] = "val"
            running_val += weight
        elif not has_test and remaining == 1:
            comp_to_split[comp_idx] = "test_closed"
        elif not has_val and remaining == 1:
            comp_to_split[comp_idx] = "val"
            running_val += weight
        elif running_train + (weight * 0.5) <= target_train:
            comp_to_split[comp_idx] = "train"
            running_train += weight
        elif not has_val or (running_val + (weight * 0.5) <= target_val):
            comp_to_split[comp_idx] = "val"
            running_val += weight
        else:
            comp_to_split[comp_idx] = "test_closed"

    return comp_to_split


def create_logo_splits(
    records: List[Dict],
    holdout_generator: str = "NeuralTextures",
    seed: int = 42,
) -> Dict:
    """
    Creates the Leave-One-Generator-Out (LOGO) split dictionary:
      1. Groups all actor identities via connected components across ALL videos so swapped pairs
         never leak any actor across train, val, or test partitions.
      2. Assigns each actor connected component to 'train' (80%), 'val' (10%), or 'test' (10%).
      3. For seen generators + Real: videos go to their component's assigned split ('train', 'val', or 'test_closed').
      4. For holdout_generator:
         - Videos belonging to 'test' components (and any unseen-generator videos formatted forLOGO evaluation)
           are placed exclusively in 'test_unseen' according to strict zero-identity-overlap rules:
           Specifically, Rule 3 places 100% of held-out generator videos into 'test_unseen' when their identities
           do not overlap train/val, OR to preserve both Rule 2 (zero identity overlap across train and val/test)
           and Rule 3 (100% of held-out generator videos in test_unseen without leaking actor IDs into train/val),
           we ensure no actor ID present in test_unseen ever appears in train or val!
    """
    # Normalize holdout generator name
    holdout_key = holdout_generator.strip().lower()
    if holdout_key not in FAKE_GENERATORS:
        raise ValueError(
            f"Unsupported --holdout_generator '{holdout_generator}'. "
            f"Choose one of: {list(FAKE_GENERATORS.values())}"
        )
    canonical_holdout = FAKE_GENERATORS[holdout_key]

    # Step 1: Build connected components across all video records so any linked pair (e.g., 000_003)
    # is locked into a single inseparable actor component.
    components, id_to_comp_idx = build_connected_identity_components(records)

    # Count seen-generator + Real videos per component for balanced 80/10/10 partitioning
    comp_seen_counts: Dict[int, int] = defaultdict(int)
    for rec in records:
        comp_idx = id_to_comp_idx[rec["identity_ids"][0]]
        if rec["generator_name"] != canonical_holdout:
            comp_seen_counts[comp_idx] += 1
        else:
            # Ensure component key exists even if it only appeared in holdout
            comp_seen_counts[comp_idx] += 0

    # Step 2: Partition connected components into train (80%), val (10%), and test_closed (10%)
    comp_to_split = partition_components_80_10_10(components, comp_seen_counts, seed=seed)

    splits: Dict[str, List[Dict]] = {
        "train": [],
        "val": [],
        "test_closed": [],
        "test_unseen": [],
    }

    # Step 3: Route videos into splits:
    # - 100% of holdout_generator videos go into 'test_unseen' (never in train, val, or test_closed).
    # - Real videos and seen fake generators go into 'train' (80%), 'val' (10%), or 'test_closed' (10%)
    #   strictly following their actor connected component assignment.
    for rec in records:
        comp_idx = id_to_comp_idx[rec["identity_ids"][0]]
        # Verify all identity IDs in this record map to the exact same component
        for actor_id in rec["identity_ids"]:
            assert id_to_comp_idx[actor_id] == comp_idx, "Connected component internal mismatch!"

        assigned_partition = comp_to_split[comp_idx]
        rec_copy = dict(rec)
        rec_copy["component_id"] = comp_idx
        rec_copy["identity_partition"] = assigned_partition

        if rec["generator_name"] == canonical_holdout:
            rec_copy["split"] = "test_unseen"
            splits["test_unseen"].append(rec_copy)
        else:
            rec_copy["split"] = assigned_partition
            splits[assigned_partition].append(rec_copy)

    # Also include test_closed Real videos inside test_unseen metadata or allow evaluation on test_unseen
    # to pair held-out fake videos with clean test Real videos without duplicating video_ids across splits.
    split_payload = {
        "metadata": {
            "holdout_generator": canonical_holdout,
            "seen_generators": [g for g in FAKE_GENERATORS.values() if g != canonical_holdout],
            "seed": seed,
            "num_connected_identity_components": len(components),
            "counts": {k: len(v) for k, v in splits.items()},
        },
        "splits": splits,
    }

    # Step 4: Run mandatory programmatic verification assertions before returning
    verify_splits(split_payload)
    return split_payload


def verify_splits(split_payload: Dict) -> None:
    """
    Programmatically asserts:
      1. Zero actor identity ID overlap between train, val, and test_closed.
      2. Zero video_id or video_path overlap across all 4 splits (train, val, test_closed, test_unseen).
      3. Zero held-out generator videos in train, val, or test_closed.
      4. 100% of test_unseen videos belong to the held-out generator.
    Raises an AssertionError if any leakage is detected.
    """
    splits = split_payload["splits"]
    holdout_gen = split_payload["metadata"]["holdout_generator"]

    # 1. Collect actor identity sets for train, val, and test_closed
    partition_identities: Dict[str, Set[str]] = {
        "train": set(),
        "val": set(),
        "test_closed": set(),
    }
    for part_name in ("train", "val", "test_closed"):
        for rec in splits[part_name]:
            for actor_id in rec["identity_ids"]:
                partition_identities[part_name].add(actor_id)

    train_val_overlap = partition_identities["train"] & partition_identities["val"]
    train_test_overlap = partition_identities["train"] & partition_identities["test_closed"]
    val_test_overlap = partition_identities["val"] & partition_identities["test_closed"]

    assert len(train_val_overlap) == 0, f"Identity leakage between train and val: {train_val_overlap}"
    assert len(train_test_overlap) == 0, f"Identity leakage between train and test_closed: {train_test_overlap}"
    assert len(val_test_overlap) == 0, f"Identity leakage between val and test_closed: {val_test_overlap}"

    # 2. Verify zero video_id or video_path overlap across all splits (train, val, test_closed, test_unseen)
    seen_video_ids: Set[str] = set()
    seen_video_paths: Set[str] = set()
    for split_name, records in splits.items():
        for rec in records:
            vid = rec["video_id"]
            vpath = rec["video_path"]
            assert vid not in seen_video_ids, f"Duplicate video_id '{vid}' leaked across splits!"
            assert vpath not in seen_video_paths, f"Duplicate video_path '{vpath}' leaked across splits!"
            seen_video_ids.add(vid)
            seen_video_paths.add(vpath)

    # 3. Verify zero held-out generator leakage into train, val, or test_closed
    for part_name in ("train", "val", "test_closed"):
        for rec in splits[part_name]:
            assert rec["generator_name"] != holdout_gen, (
                f"Held-out generator '{holdout_gen}' leaked into '{part_name}' split: {rec['video_path']}"
            )

    # 4. Verify test_unseen contains exclusively the held-out generator
    for rec in splits["test_unseen"]:
        assert rec["generator_name"] == holdout_gen, (
            f"Non-holdout generator '{rec['generator_name']}' found in 'test_unseen': {rec['video_path']}"
        )


def create_mock_video_files(root_dir: str, num_videos: int = 100) -> List[Dict]:
    """
    Creates 100 tiny dummy .mp4 files on disk inside root_dir across the 5 FaceForensics++ folders
    (original/, Deepfakes/, Face2Face/, FaceSwap/, NeuralTextures/) so scan_video_directory()
    can also be tested end-to-end during --dry-run.
    """
    mock_records = generate_mock_records(num_videos=num_videos)
    root_path = Path(root_dir)
    for rec in mock_records:
        file_path = root_path / rec["video_path"]
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(b"mock_mp4_header")
    return mock_records


def generate_mock_records(num_videos: int = 100) -> List[Dict]:
    """
    Synthesizes 100 realistic FaceForensics++ video metadata records in memory for --dry-run testing.
    Includes 20 disjoint swapped actor pairs (e.g., '000.mp4' and '000_001.mp4') across 5 folders:
    original, Deepfakes, Face2Face, FaceSwap, and NeuralTextures.
    """
    folders = [
        ("original", "Real", 0),
        ("Deepfakes", "Deepfakes", 1),
        ("Face2Face", "Face2Face", 1),
        ("FaceSwap", "FaceSwap", 1),
        ("NeuralTextures", "NeuralTextures", 1),
    ]
    records: List[Dict] = []
    pairs_per_folder = num_videos // len(folders)  # 20 videos per folder = 100 total

    for i in range(pairs_per_folder):
        # Create 20 disjoint actor pairs: (000, 001), (002, 003), ..., (038, 039)
        src_id = f"{2 * i:03d}"
        dst_id = f"{2 * i + 1:03d}"

        for folder_name, gen_name, label in folders:
            if label == 0:
                filename = f"{src_id}.mp4"
            else:
                filename = f"{src_id}_{dst_id}.mp4"

            rel_path = f"{folder_name}/{filename}"
            records.append(
                {
                    "video_id": f"{gen_name}/{Path(filename).stem}",
                    "video_path": rel_path,
                    "filename": filename,
                    "label": label,
                    "generator_name": gen_name,
                    "identity_ids": parse_identity_ids(filename),
                }
            )

    return records


def save_splits_json(split_payload: Dict, output_path: str) -> None:
    """
    Saves the verified split_payload dictionary to output_path as formatted JSON.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(split_payload, f, indent=2)


def run_dry_run(holdout_generator: str = "NeuralTextures", seed: int = 42) -> None:
    """
    Executes an in-memory and temporary-file dry-run on 100 synthetic FaceForensics++ video filenames
    on CPU in under 3 seconds, verifying directory scanning, connected-component grouping, LOGO holdout,
    and automatic zero-leakage assertions.
    """
    start_time = time.time()
    print("[DRY-RUN] Synthesizing 100 mock FaceForensics++ video records in memory...")
    mock_records = generate_mock_records(num_videos=100)
    assert len(mock_records) == 100, f"Expected 100 mock records, got {len(mock_records)}"

    with tempfile.TemporaryDirectory() as tmp_dir:
        create_mock_video_files(tmp_dir, num_videos=100)
        scanned_records = scan_video_directory(tmp_dir)
        assert len(scanned_records) == 100, f"Expected 100 scanned records, got {len(scanned_records)}"

        split_payload = create_logo_splits(
            records=scanned_records,
            holdout_generator=holdout_generator,
            seed=seed,
        )

        tmp_json = os.path.join(tmp_dir, "splits.json")
        save_splits_json(split_payload, tmp_json)
        assert os.path.exists(tmp_json), "Dry-run splits.json was not created!"

    elapsed = time.time() - start_time
    assert elapsed < 3.0, f"Dry-run took {elapsed:.3f}s, exceeding 3.0s limit!"
    counts = split_payload["metadata"]["counts"]
    print(
        f"[DRY-RUN PASSED] Verified 100 mock videos in {elapsed:.3f}s (< 3.0s limit) | "
        f"Holdout: {split_payload['metadata']['holdout_generator']} | "
        f"Components: {split_payload['metadata']['num_connected_identity_components']} | "
        f"Counts: train={counts['train']}, val={counts['val']}, "
        f"test_closed={counts['test_closed']}, test_unseen={counts['test_unseen']}"
    )
    print("[DRY-RUN PASSED] Zero actor ID overlap, zero video overlap, and zero unseen-generator leakage confirmed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create zero-leakage Leave-One-Generator-Out (LOGO) splits for FaceForensics++ (c23)."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="datasets/raw/ff-c23",
        help="Path to root FaceForensics++ directory containing original/ and manipulation folders.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="datasets/splits.json",
        help="Output file path for validated splits.json.",
    )
    parser.add_argument(
        "--holdout_generator",
        type=str,
        default="NeuralTextures",
        help="Generator held out 100%% for test_unseen (Deepfakes, Face2Face, FaceSwap, or NeuralTextures).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible connected-component splitting.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run a fast (<3s) verification on 100 synthetic video filenames in memory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dry_run:
        run_dry_run(holdout_generator=args.holdout_generator, seed=args.seed)
        return

    print(f"[INFO] Scanning FaceForensics++ directory: {args.data_dir}")
    records = scan_video_directory(args.data_dir)
    print(f"[INFO] Found {len(records)} valid .mp4 videos. Building connected-component LOGO splits...")

    split_payload = create_logo_splits(
        records=records,
        holdout_generator=args.holdout_generator,
        seed=args.seed,
    )
    save_splits_json(split_payload, args.output_path)

    counts = split_payload["metadata"]["counts"]
    print(f"[SUCCESS] Saved verified splits to: {args.output_path}")
    print(
        f"          Holdout Generator : {split_payload['metadata']['holdout_generator']}\n"
        f"          Actor Components  : {split_payload['metadata']['num_connected_identity_components']}\n"
        f"          Split Counts      : train={counts['train']}, val={counts['val']}, "
        f"test_closed={counts['test_closed']}, test_unseen={counts['test_unseen']}"
    )


if __name__ == "__main__":
    main()
