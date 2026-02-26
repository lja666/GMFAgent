# -*- coding: utf-8 -*-
"""
Copy base data from legacy paths into project base_data folder.
Run from project root:  python scripts/copy_base_data.py

If a legacy path does not exist, that item is skipped (with a warning).
"""
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_DATA = PROJECT_ROOT / "base_data"

# Legacy paths (previous defaults). Edit these if your files are elsewhere.
LEGACY = {
    "rasters/population.tif": Path(r"D:\DeskTop\应急地震人员伤亡数据和计算\landscan-global-2022-assets\landscan-global-2022.tif"),
    "rasters/vs30.tif": Path(r"D:\DeskTop\台湾地震\global_vs30_tif\global_vs30.tif"),
    "rasters/dem.tif": Path(r"D:\DeskTop\栅格文件自动化处理\hyd_glo_dem_30s\hyd_glo_dem_30s.tif"),
    "shapefiles/admin": Path(r"D:\Q Gis文件\ne_10m_admin_1_states_provinces"),  # directory: copy all .shp/.dbf/.shx/.prj etc.
    "fault/china_faults.csv": Path(r"D:\DeskTop\系统平台\usgs_query\断层信息\china_eqfault_boundary_and_angles_drop_duplicates_dgree_from_E.csv"),
    "gmpe_root": Path(PROJECT_ROOT.parent / "gmpe_root1"),  # directory
    "ml_models/ml_xgb.pickle.dat": Path(r"D:\数据文件\基于机器学习的巨灾风险分析方法\浩天\调用模型预测\crustal\回归.pickle.dat"),
}


def copy_file(src: Path, dst: Path) -> bool:
    if not src.exists():
        print(f"  Skip (not found): {src}")
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"  Copied: {src.name} -> {dst}")
    return True


def copy_shapefile_dir(src_dir: Path, dst_dir: Path) -> bool:
    if not src_dir.is_dir():
        print(f"  Skip (not found): {src_dir}")
        return False
    dst_dir.mkdir(parents=True, exist_ok=True)
    # Copy all files in the shapefile directory ( .shp, .dbf, .shx, .prj, .cpg, .qpj, etc. )
    for f in src_dir.iterdir():
        if f.is_file():
            shutil.copy2(f, dst_dir / f.name)
            print(f"  Copied: {f.name} -> {dst_dir / f.name}")
    return True


def copy_tree(src: Path, dst: Path) -> bool:
    if not src.is_dir():
        print(f"  Skip (not found): {src}")
        return False
    shutil.copytree(src, dst, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".git"))
    print(f"  Copied tree: {src} -> {dst}")
    return True


def main():
    print("Copying base data into", BASE_DATA)
    BASE_DATA.mkdir(parents=True, exist_ok=True)

    for rel, src in LEGACY.items():
        dst = BASE_DATA / rel
        print(f"\n{rel}:")
        if rel == "shapefiles/admin":
            copy_shapefile_dir(src, dst)
        elif rel == "gmpe_root":
            copy_tree(src, dst)
        else:
            copy_file(src, dst)

    print("\nDone. Check base_data/ and set config (or env) if you used different filenames.")


if __name__ == "__main__":
    main()
