import csv

from scipy import stats

REPO_ROOT_RELATIVE = "."  # run from repo root


def load(path: str) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def build_effective_score_map() -> dict[tuple[str, str], float]:
    source_files = {
        "high": "data/processed/CLEAN_high.csv",
        "medium": "data/processed/CLEAN_medium.csv",
        "low": "data/processed/low.csv",
    }
    eff_map = {}
    for bucket, path in source_files.items():
        for r in load(path):
            eff_map[(bucket, r["text"])] = float(r["effective_score"])
    return eff_map


def analyze(faith_path: str, name: str, eff_map: dict[tuple[str, str], float]) -> None:
    faith = load(faith_path)
    eff, lime, shap = [], [], []
    missing = 0
    for r in faith:
        key = (r["bucket"], r["text"])
        e = eff_map.get(key)
        if e is None:
            missing += 1
            continue
        eff.append(e)
        lime.append(float(r["lime_faithfulness_score"]))
        shap.append(float(r["shap_faithfulness_score"]))

    print(f"=== {name} (n={len(eff)}, missing={missing}) ===")
    for method_name, scores in [("LIME", lime), ("SHAP", shap)]:
        pearson_r, pearson_p = stats.pearsonr(eff, scores)
        spearman_r, spearman_p = stats.spearmanr(eff, scores)
        print(
            f"  {method_name}: Pearson r={pearson_r:+.4f} (p={pearson_p:.4f})   "
            f"Spearman rho={spearman_r:+.4f} (p={spearman_p:.4f})"
        )
    print()


def main() -> None:
    eff_map = build_effective_score_map()
    analyze("data/processed/faithfulness_results.csv", "Original (fixed top-5)", eff_map)
    analyze(
        "data/processed/faithfulness_results_adaptive.csv",
        "Length-corrected (adaptive top-N)",
        eff_map,
    )


if __name__ == "__main__":
    main()
