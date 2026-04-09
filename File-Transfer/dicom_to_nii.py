import argparse
import re
from pathlib import Path
from typing import List, Optional

import SimpleITK as sitk


def sanitize_filename(name: str) -> str:
    """
    Remove invalid characters from a filename.
    """
    if not name:
        return "unnamed_series"
    name = re.sub(r'[\\/*?:"<>|]+', "_", name)
    name = re.sub(r"\s+", "_", name.strip())
    return name[:150] if len(name) > 150 else name


def get_series_description(first_dcm_file: str) -> str:
    """
    Try to read the DICOM SeriesDescription for output naming.
    """
    try:
        reader = sitk.ImageFileReader()
        reader.SetFileName(first_dcm_file)
        reader.LoadPrivateTagsOn()
        reader.ReadImageInformation()

        # DICOM tag: Series Description = 0008|103e
        tag = "0008|103e"
        if reader.HasMetaDataKey(tag):
            return reader.GetMetaData(tag)
    except Exception:
        pass
    return "series"


def convert_one_series(dicom_dir: Path, output_path: Path) -> Optional[Path]:
    """
    Convert one DICOM folder containing a single series into a NIfTI file.
    If multiple series are detected in the folder, only the first one is converted.
    """
    dicom_dir = dicom_dir.resolve()
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(dicom_dir))
    if not series_ids:
        print(f"[Skipped] No DICOM series found in directory: {dicom_dir}")
        return None

    if len(series_ids) > 1:
        print(f"[Warning] Multiple series detected in directory. Only the first one will be converted: {dicom_dir}")
        print(f"          Series IDs: {series_ids}")

    series_id = series_ids[0]
    file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(dicom_dir), series_id)

    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(file_names)
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()

    image = reader.Execute()

    sitk.WriteImage(image, str(output_path))
    print(f"[Done] {dicom_dir} -> {output_path}")
    return output_path


def find_all_series_dirs(root_dir: Path) -> List[Path]:
    """
    Recursively find all directories under root_dir that contain at least one DICOM series.
    """
    root_dir = root_dir.resolve()
    series_dirs = []

    # Check the root directory itself first
    try:
        ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(root_dir))
        if ids:
            series_dirs.append(root_dir)
    except Exception:
        pass

    # Then recursively check all subdirectories
    for subdir in root_dir.rglob("*"):
        if not subdir.is_dir():
            continue
        try:
            ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(subdir))
            if ids:
                series_dirs.append(subdir)
        except Exception:
            continue

    # Remove duplicates
    unique_dirs = []
    seen = set()
    for d in series_dirs:
        s = str(d)
        if s not in seen:
            seen.add(s)
            unique_dirs.append(d)

    return unique_dirs


def convert_all_series(root_dir: Path, out_dir: Path) -> None:
    """
    Recursively search for all DICOM series under root_dir
    and convert each series into a separate .nii.gz file.
    """
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    series_dirs = find_all_series_dirs(root_dir)
    if not series_dirs:
        print(f"[Finished] No DICOM series found under: {root_dir}")
        return

    print(f"[Info] Found {len(series_dirs)} directories containing DICOM series")

    count = 0
    for dicom_dir in series_dirs:
        try:
            series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(dicom_dir))
            if not series_ids:
                continue

            for idx, series_id in enumerate(series_ids, start=1):
                file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(dicom_dir), series_id)
                if not file_names:
                    continue

                desc = get_series_description(file_names[0])
                folder_name = sanitize_filename(dicom_dir.name)
                desc_name = sanitize_filename(desc)
                short_id = sanitize_filename(series_id[-8:])

                out_name = f"{folder_name}__{desc_name}__{short_id}.nii.gz"
                output_path = out_dir / out_name

                reader = sitk.ImageSeriesReader()
                reader.SetFileNames(file_names)
                reader.MetaDataDictionaryArrayUpdateOn()
                reader.LoadPrivateTagsOn()

                image = reader.Execute()
                sitk.WriteImage(image, str(output_path))

                print(f"[Done] {dicom_dir} | series {idx}/{len(series_ids)} -> {output_path.name}")
                count += 1

        except Exception as e:
            print(f"[Error] Failed to convert: {dicom_dir}\n        Reason: {e}")

    print(f"[Finished] Exported {count} NIfTI files to: {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Convert DICOM to NIfTI (.nii.gz)")
    parser.add_argument(
        "--input",
        required=True,
        type=str,
        help="Path to the input DICOM folder"
    )
    parser.add_argument(
        "--output",
        required=True,
        type=str,
        help="Output .nii/.nii.gz file path (single conversion) or output folder (batch conversion)"
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search all subfolders under input and convert all detected series"
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if args.recursive:
        convert_all_series(input_path, output_path)
    else:
        if output_path.suffix not in [".nii", ".gz"] and not str(output_path).endswith(".nii.gz"):
            raise ValueError("In non-recursive mode, --output must be a .nii or .nii.gz file path")
        convert_one_series(input_path, output_path)


if __name__ == "__main__":
    main()
