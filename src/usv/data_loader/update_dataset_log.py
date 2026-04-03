"""
Updates the dataset Excel log with new clips.
- part1: Inherits metadata from the original clip.
- part2+: Created empty (except Name and Duration).
- Column order is strictly preserved.
"""
import pandas as pd
import re
from pathlib import Path

def update_dataset_log(excel_path: str, metadata_dir: str, output_path: str):
    # 1. Define strict column order based on your requirement
    cols = [
        "название отрывка", "брать ли в финальную выборку?", "Water", "Sky", "Land",
        "Piers", "Bridge", "Vessel", "Buoy", "LandingMark", "BridgeLight",
        "Other", "Сложность", "Длительность", "комментарий"
    ]

    # 2. Load existing Excel
    df_old = pd.read_excel(excel_path)
    # Ensure all columns exist, fill missing with NaN if necessary
    for c in cols:
        if c not in df_old.columns:
            df_old[c] = None

    # Index by the name column for easy lookup
    df_old = df_old.set_index("название отрывка")

    # 3. Scan metadata CSVs
    metadata_dir = Path(metadata_dir)
    new_rows = []

    csv_files = sorted(list(metadata_dir.glob("*.csv")))

    for csv_file in csv_files:
        clip_name = csv_file.stem
        df_csv = pd.read_csv(csv_file)
        duration_sec = round(len(df_csv) / 5.0, 2)

        # Determine the base name (e.g., 'center_VID_..._0003' from 'center_VID_..._0003_part1')
        base_name = re.sub(r'_part\d+$', '', clip_name)
        is_part_2_plus = "_part" in clip_name and not clip_name.endswith("_part1")

        # Create row
        row = {col: None for col in cols}
        row["название отрывка"] = clip_name
        row["Длительность"] = duration_sec

        # If it's the original or _part1, try to carry over data
        if not is_part_2_plus:
            if base_name in df_old.index:
                old_data = df_old.loc[base_name]
                for col in cols:
                    if col in old_data and pd.notnull(old_data[col]):
                        row[col] = old_data[col]

        new_rows.append(row)

    # 4. Create and sort DataFrame
    df_new = pd.DataFrame(new_rows, columns=cols)

    # Sort: 0001, 0001_part1, 0001_part2, 0002...
    df_new = df_new.sort_values(by="название отрывка")

    # 5. Save
    df_new.to_excel(output_path, index=False)
    print(f"Excel updated successfully: {output_path}")

if __name__ == "__main__":
    update_dataset_log(
        excel_path=r"C:\Users\User\Desktop\dataset_log.xlsx",
        metadata_dir=r"data/interim/sequent_frames_v5/metadata",
        output_path=r"dataset_log_updated.xlsx"
    )
