"""DFT availability audit and remaining-candidate export."""
import builtins
from datetime import datetime
from pathlib import Path
import io
import re
import shutil
import zipfile
import numpy as np
import pandas as pd

def package_remaining_db1_dft(project_root='D:\\TB3', *, dft_root='D:\\DFT_TEST', output_dir=None, show=True, verbose=True):
    """Audit available spectra and package remaining final DB1 Top-50 structures.

Run after notebook Cell 16A. Preserves its manifest/rank matching and original
file test (>10 rows, >=7 columns, finite first seven columns). Scans loose DAT
files and ZIP archives only; RAR archives are not scanned. Availability of a
numeric dielectric file does not establish SCF convergence or scientific QC.
No manual exclusions are applied. The same material ID is counted only once.
Each call creates a fresh output folder; previous packages are never removed.
An explicit output_dir must not already exist. Does not submit calculations.
"""

    def print(*args, **kwargs):
        if verbose:
            builtins.print(*args, **kwargs)

    def display(value):
        if show:
            try:
                from IPython.display import display as ipython_display
            except ImportError:
                builtins.print(value.to_string(index=False))
            else:
                ipython_display(value)
    ROOT = Path(project_root) / 'processed/paired_training'
    DB1_DIR = ROOT / 'paper_outputs' / 'dft_validation_shortlist' / 'discovery_first_zintl_SLME'
    SOURCE_DIR = DB1_DIR / 'SEND_TO_DFT_DB1_FINAL_Top50_SLME'
    SOURCE_MANIFEST = SOURCE_DIR / 'DB1_FINAL_Top50_SLME_manifest.csv'
    SOURCE_CIF_DIR = SOURCE_DIR / 'CIF'
    SOURCE_JSON_DIR = SOURCE_DIR / 'JSON'
    DFT_ROOT = Path(dft_root)
    manifest = pd.read_csv(SOURCE_MANIFEST)
    manifest['material_id'] = manifest['material_id'].astype(str).str.strip().str.lower()
    assert len(manifest) == 50
    assert manifest['material_id'].nunique() == 50
    TOP50_IDS = set(manifest['material_id'])

    def get_material_id(filename):
        """
    Examples:
      epsilon_rank001_mp_btzq_NaP5.dat
          -> mp-btzq

      epsilon_rank..._mp-ekf_....dat
          -> mp-ekf
    """
        text = Path(filename).name.lower().replace('_', '-')
        match = re.search('mp-[a-z0-9]+', text)
        return match.group(0) if match else None

    def valid_epsilon_dat(raw_bytes):
        """
    Require a numerical dielectric-spectrum table
    with at least:
        E,
        eps1_x, eps1_y, eps1_z,
        eps2_x, eps2_y, eps2_z
    """
        try:
            arr = np.loadtxt(io.BytesIO(raw_bytes))
            arr = np.asarray(arr, dtype=float)
            return arr.ndim == 2 and arr.shape[0] > 10 and (arr.shape[1] >= 7) and np.isfinite(arr[:, :7]).all()
        except Exception:
            return False
    completed_sources = {}
    invalid_files = []

    def register_result(material_id, source):
        if material_id not in TOP50_IDS:
            return
        completed_sources.setdefault(material_id, set()).add(str(source))
    for dat_path in DFT_ROOT.rglob('*.dat'):
        material_id = get_material_id(dat_path.name)
        if material_id not in TOP50_IDS:
            continue
        try:
            raw = dat_path.read_bytes()
            if valid_epsilon_dat(raw):
                register_result(material_id, dat_path)
            else:
                invalid_files.append(str(dat_path))
        except Exception:
            invalid_files.append(str(dat_path))
    for zip_path in DFT_ROOT.rglob('*.zip'):
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                for member in zf.namelist():
                    if not member.lower().endswith('.dat'):
                        continue
                    material_id = get_material_id(member)
                    if material_id not in TOP50_IDS:
                        continue
                    raw = zf.read(member)
                    if valid_epsilon_dat(raw):
                        register_result(material_id, f'{zip_path}::{member}')
                    else:
                        invalid_files.append(f'{zip_path}::{member}')
        except zipfile.BadZipFile:
            print('WARNING — bad ZIP:', zip_path)
    COMPLETED_DFT_IDS = set(completed_sources)
    REMAINING_DFT_IDS = TOP50_IDS - COMPLETED_DFT_IDS
    completed = manifest[manifest['material_id'].isin(COMPLETED_DFT_IDS)].copy().sort_values('DFT_shortlist_rank' if 'DFT_shortlist_rank' in manifest.columns else 'SLME_rank').reset_index(drop=True)
    remaining = manifest[manifest['material_id'].isin(REMAINING_DFT_IDS)].copy().sort_values('DFT_shortlist_rank' if 'DFT_shortlist_rank' in manifest.columns else 'SLME_rank').reset_index(drop=True)
    remaining['DFT_run_rank'] = np.arange(len(remaining)) + 1
    assert len(completed) + len(remaining) == 50
    completed['DFT_source_count'] = completed['material_id'].map(lambda mid: len(completed_sources[mid]))
    completed['DFT_sources'] = completed['material_id'].map(lambda mid: ' | '.join(sorted(completed_sources[mid])))
    N_COMPLETED = len(completed)
    N_REMAINING = len(remaining)
    RUN_DIR = Path(output_dir) if output_dir is not None else DB1_DIR / 'DFT_completion_runs' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    RUN_DIR.mkdir(parents=True, exist_ok=False)
    AUDIT_DIR = RUN_DIR / 'DFT_completion_audit'
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    COMPLETED_CSV = AUDIT_DIR / 'DB1_Top50_completed_DFT_detected.csv'
    REMAINING_CSV = AUDIT_DIR / 'DB1_Top50_remaining_DFT_candidates.csv'
    completed.to_csv(COMPLETED_CSV, index=False, encoding='utf-8-sig')
    remaining.to_csv(REMAINING_CSV, index=False, encoding='utf-8-sig')
    if N_REMAINING > 0:
        OUT_DIR = RUN_DIR / f'SEND_TO_DFT_DB1_FINAL_{N_REMAINING}_REMAINING'
        OUT_CIF_DIR = OUT_DIR / 'CIF'
        OUT_JSON_DIR = OUT_DIR / 'JSON'
        OUT_CIF_DIR.mkdir(parents=True)
        OUT_JSON_DIR.mkdir(parents=True)
        for _, row in remaining.iterrows():
            src_cif = SOURCE_CIF_DIR / str(row['CIF_file'])
            src_json = SOURCE_JSON_DIR / str(row['JSON_file'])
            if not src_cif.is_file():
                raise FileNotFoundError(src_cif)
            if not src_json.is_file():
                raise FileNotFoundError(src_json)
            shutil.copy2(src_cif, OUT_CIF_DIR / src_cif.name)
            shutil.copy2(src_json, OUT_JSON_DIR / src_json.name)
        remaining.to_csv(OUT_DIR / 'remaining_DFT_manifest.csv', index=False, encoding='utf-8-sig')
        completed.to_csv(OUT_DIR / 'already_completed_DFT.csv', index=False, encoding='utf-8-sig')
        (OUT_DIR / 'remaining_material_ids.txt').write_text('\n'.join(remaining['material_id']) + '\n', encoding='utf-8')
        (OUT_DIR / 'README.txt').write_text(f'FINAL DB1 REMAINING DFT PACKAGE\n\nOriginal FINAL shortlist : 50\nCompleted DFT detected    : {N_COMPLETED}\nRemaining for DFT         : {N_REMAINING}\n\nCompleted status was determined automatically\nfrom valid dielectric-spectrum .dat files under:\n\n{DFT_ROOT}\n\nNo material ID was manually hard-coded as completed.\n\nOriginal DFT_shortlist_rank / SLME_rank are preserved.\nDFT_run_rank = 1..{N_REMAINING} is only the execution\norder for the remaining calculations.\n', encoding='utf-8')
        cifs = list(OUT_CIF_DIR.glob('*.cif'))
        jsons = list(OUT_JSON_DIR.glob('*.json'))
        assert len(cifs) == N_REMAINING
        assert len(jsons) == N_REMAINING
        ZIP_BASE = RUN_DIR / f'SEND_TO_DFT_DB1_FINAL_{N_REMAINING}_REMAINING'
        ZIP_PATH = Path(shutil.make_archive(str(ZIP_BASE), 'zip', root_dir=str(OUT_DIR)))
    else:
        OUT_DIR = None
        ZIP_PATH = None
    print()
    print('=' * 90)
    print('DB1 TOP-50 DFT COMPLETION AUDIT')
    print('=' * 90)
    print('Original Top-50     :', 50)
    print('Completed DFT found :', N_COMPLETED)
    print('Remaining for DFT   :', N_REMAINING)
    print('Invalid .dat ignored:', len(invalid_files))
    print('\nCompleted DFT ranks:')
    rank_col = 'DFT_shortlist_rank' if 'DFT_shortlist_rank' in completed.columns else 'SLME_rank'
    print(completed[rank_col].tolist())
    print('\nRemaining ranks:')
    print(remaining[rank_col].tolist())
    print('\nRemaining material IDs:')
    for _, row in remaining.iterrows():
        print(f"Rank {row[rank_col]:>2} | {row['material_id']} | {row.get('formula_pretty', '')}")
    print('\nCompleted audit CSV:')
    print(COMPLETED_CSV)
    print('\nRemaining audit CSV:')
    print(REMAINING_CSV)
    if ZIP_PATH is not None:
        print('\nZIP TO SEND:')
        print(ZIP_PATH)
    else:
        print('\nAll 50 candidates already have valid DFT results.')
    preview_cols = ['DFT_run_rank', 'DFT_shortlist_rank', 'SLME_rank', 'material_id', 'formula_pretty', 'predicted_SLME_percent']
    preview_cols = [c for c in preview_cols if c in remaining.columns]
    display(remaining[preview_cols])
    return {'completed': completed, 'remaining': remaining, 'invalid_files': invalid_files, 'sources': completed_sources, 'n_completed': N_COMPLETED, 'n_remaining': N_REMAINING, 'output_dir': RUN_DIR, 'package_dir': OUT_DIR, 'zip_path': ZIP_PATH, 'completed_csv': COMPLETED_CSV, 'remaining_csv': REMAINING_CSV}
