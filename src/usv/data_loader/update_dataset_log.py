"""
Updates the dataset Excel log with new clips from metadata CSVs.
- Clips already present in the log by exact name retain all existing data.
- Duration is always recalculated from the metadata CSV.
- New clips not present in the log are added empty (except Clip name and Duration).
- Old log rows that could not be matched to any new clip are printed as warnings.
- Column order is strictly preserved.
"""
import argparse
import pandas as pd
from pathlib import Path


COLS = [
    "Clip name",
    "Water", "Sky", "Land", "Piers", "Bridge", "Vessel",
    "Buoy", "LandingMark", "BridgeLight", "Other",
    "Difficulty", "Duration", "Commentary",
]


def update_dataset_log(
    excel_path: str,
    metadata_dir: str,
    output_path: str,
    target_fps: int = 5,
):
    # 1. Load existing log
    df_old = pd.read_excel(excel_path)
    for c in COLS:
        if c not in df_old.columns:
            df_old[c] = None
    df_old = df_old.set_index("Clip name")

    # 2. Scan metadata CSVs and build new rows
    metadata_dir = Path(metadata_dir)
    new_rows = []
    matched_old_names = set()

    for csv_file in sorted(metadata_dir.glob("*.csv")):
        clip_name = csv_file.stem
        df_csv = pd.read_csv(csv_file)
        duration_sec = round(len(df_csv) / target_fps, 2)

        row = {col: None for col in COLS}
        row["Clip name"] = clip_name
        row["Duration"] = duration_sec

        if clip_name in df_old.index:
            old_data = df_old.loc[clip_name]
            for col in COLS:
                if col == "Duration":
                    continue
                if col in old_data and pd.notnull(old_data[col]):
                    row[col] = old_data[col]
            matched_old_names.add(clip_name)

        new_rows.append(row)

    # 3. Warn about old rows that were not carried over
    unmatched = [name for name in df_old.index if name not in matched_old_names]
    if unmatched:
        print(f"\n[WARNING] {len(unmatched)} old log row(s) could not be matched "
              f"to any new clip and were NOT carried over. Manual migration required:")
        for name in unmatched:
            print(f"  - {name}")
        print()

    # 4. Build, sort and save
    df_new = pd.DataFrame(new_rows, columns=COLS)
    df_new = df_new.sort_values(by="Clip name").reset_index(drop=True)
    df_new.to_excel(output_path, index=False)
    print(f"Log updated: {output_path} ({len(df_new)} clips, "
          f"{len(matched_old_names)} rows carried over from old log)")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Update dataset_log.xlsx with clips from metadata directory."
    )
    parser.add_argument(
        "--excel-path",
        default="data/metadata/dataset_log_v5.xlsx",
        help="Path to the existing dataset log Excel file.",
    )
    parser.add_argument(
        "--metadata-dir",
        default="data/interim/sequent_frames_v5-1/metadata",
        help="Directory containing per-clip metadata CSVs.",
    )
    parser.add_argument(
        "--output-path",
        default="data/metadata/dataset_log_v5-1.xlsx",
        help="Output path for the updated Excel file (can overwrite input).",
    )
    parser.add_argument(
        "--target-fps", type=int, default=5,
        help="FPS used during clip extraction, for duration calculation (default: 5).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    update_dataset_log(
        excel_path=args.excel_path,
        metadata_dir=args.metadata_dir,
        output_path=args.output_path,
        target_fps=args.target_fps,
    )
