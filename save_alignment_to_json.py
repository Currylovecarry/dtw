import json

def save_alignment_path(alignment_path, output_path="alignment.json"):
    """
    alignment_path: List of (i, j) tuples
    output_path:    JSON file to save
    """
    # 转换为字典列表
    alignment_data = [
        {"video1_frame": int(i), "video2_frame": int(j)}
        for i, j in alignment_path
    ]

    with open(output_path, "w") as f:
        json.dump(alignment_data, f, indent=2)

    print(f"Alignment saved to {output_path}")
