import argparse
from pathlib import Path
import shutil
import subprocess

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def init_private_data_repo(target_dir: str | Path | None = None) -> Path:
    if target_dir is None:
        target_path = PROJECT_ROOT.parent / "quant-data-private"
    else:
        target_path = Path(target_dir).resolve()

    target_path.mkdir(parents=True, exist_ok=True)
    raw_dir = target_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    src_raw = PROJECT_ROOT / "data" / "raw"
    files_to_copy = [
        "s_and_p_500_daily_composition.parquet",
        "s_and_p_500_prices.parquet",
    ]

    for fname in files_to_copy:
        src_file = src_raw / fname
        if not src_file.exists():
            raise FileNotFoundError(f"Source file not found: {src_file}")
        dst_file = raw_dir / fname
        shutil.copy2(src_file, dst_file)

    readme_content = (
        "# Quant Data Private\n\n"
        "Private market data store for S&P 500 quantitative research platform.\n"
        "Contains raw daily OHLCV prices and historical index composition.\n"
        "Used exclusively by automated CI/CD runners for daily inference.\n"
    )
    with open(target_path / "README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)

    git_dir = target_path / ".git"
    if not git_dir.exists():
        subprocess.run(["git", "init"], cwd=target_path, check=True, capture_output=True)

    subprocess.run(["git", "add", "README.md", "raw/"], cwd=target_path, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "chore: initial commit of private raw market data"], cwd=target_path, check=False, capture_output=True)

    return target_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-dir", type=str, default=None)
    args = parser.parse_args()

    out_path = init_private_data_repo(args.target_dir)
    print(f"Private data repository initialized at: {out_path}")
    print("\nNext steps to link with GitHub:")
    print(f"  cd \"{out_path}\"")
    print("  git remote add origin https://github.com/marwan602/quant-data-private.git")
    print("  git branch -M main")
    print("  git push -u origin main")
