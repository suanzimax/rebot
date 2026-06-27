from pathlib import Path


def read_metrics(path: Path):
    metrics = {}
    if not path.exists():
        return metrics

    with open(path, "r") as f:
        for line in f:
            if ":" in line:
                key, value = line.strip().split(":", 1)
                try:
                    metrics[key.strip()] = float(value.strip())
                except ValueError:
                    metrics[key.strip()] = value.strip()

    return metrics


day02 = read_metrics(Path("results/day02_sac_fetchreach/final_metrics.txt"))
day03 = read_metrics(Path("results/day03_sac_her_fetchreach/final_metrics.txt"))

output_path = Path("results/day03_sac_her_fetchreach/dense_vs_sparse_her_comparison.md")

with open(output_path, "w") as f:
    f.write("# Dense Reward SAC vs Sparse Reward SAC + HER\n\n")

    f.write("## Day 2: SAC on FetchReachDense-v3\n\n")
    if day02:
        for k, v in day02.items():
            f.write(f"- {k}: {v}\n")
    else:
        f.write("- Day 2 metrics not found.\n")

    f.write("\n## Day 3: SAC + HER on FetchReach-v3\n\n")
    if day03:
        for k, v in day03.items():
            f.write(f"- {k}: {v}\n")
    else:
        f.write("- Day 3 metrics not found.\n")

    f.write("\n## Interpretation\n\n")
    f.write(
        "Dense reward provides continuous distance feedback, which usually makes "
        "early learning easier. Sparse reward provides only task completion feedback, "
        "so HER is introduced to relabel failed trajectories and improve sample efficiency.\n"
    )

print(f"Saved comparison to {output_path}")