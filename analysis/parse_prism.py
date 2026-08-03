"""
Parser for GraphPad Prism (.pzfx) files used to assemble the paired
ImageJ vs AI dataset reported in the companion paper.

NOTE ON USE CASE
----------------
This script is provided as an ancillary utility for users who hold their
own Prism (.pzfx) files structured similarly to the project's annotation
workflow (one .pzfx per annotator, with TwoWay or XY tables labelled
ImageJ / IA / Roboflow). The .pzfx source files used to assemble the
manuscript's dataset are NOT redistributed in this Zenodo deposit
(they contain intermediate, non-publishable annotation drafts).
The downstream output of this script - the cleaned, tidy CSV - IS
deposited as `paired_imagej_vs_ai_clean.csv`.

Therefore: the typical reproducibility user does NOT need to run this
script. They can go straight to `paired_analysis.py` which consumes the
deposited CSV. This file is kept in the deposit only for transparency
about how the CSV was produced.

USAGE (for users with their own .pzfx files)
--------------------------------------------
    pip install pandas numpy
    python3 parse_prism.py \\
        --input-dir  /path/to/folder/with/pzfx_files \\
        --output-csv ./paired_data_raw.csv

Expected layout of the input folder:
    annotator1.pzfx   annotator2.pzfx   annotator3.pzfx   (fraction-scale)
    annotator1_pct.pzfx  annotator2_pct.pzfx              (percent-scale)

The three annotators are identified only as Annotator 1-3 throughout the public
data. The mapping to individual names is held by the corresponding author and is
deliberately not reproduced here or anywhere in this repository.
"""
import argparse
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

NS = {'p': 'http://graphpad.com/prism/Prism.htm'}


def parse_val(v):
    if v is None or v == '':
        return np.nan
    try:
        return float(v.replace(',', '.'))
    except (ValueError, AttributeError):
        return np.nan


def get_table_data_positional(table):
    """Return list of (title, [[t8, t12, t24, ...], ...]) preserving positional order."""
    result = []
    for ycol in table.findall('p:YColumn', NS):
        title_elem = ycol.find('p:Title', NS)
        title = title_elem.text if title_elem is not None and title_elem.text else None
        reps = []
        for sc in ycol.findall('p:Subcolumn', NS):
            vals = [parse_val(d.text) for d in sc.findall('p:d', NS)]
            reps.append(vals)
        result.append((title, reps))
    return result


def classify_table(title, ttype):
    """Return 'imagej', 'ai', or None."""
    if not title:
        return None
    tl = title.lower()
    # AI / Roboflow markers
    if any(k in tl for k in ['rboflow', 'roboflow', 'analise ia']):
        return 'ai'
    if tl.strip() == 'ia':
        return 'ai'
    if 'ia coluna' in tl or 'ia xy' in tl:
        return 'ai'
    # ImageJ markers
    if any(k in tl for k in ['imagej', 'image j', 'image_j']):
        return 'imagej'
    # Generic column/xy without IA -> ImageJ
    if ('colun' in tl or 'xy' in tl) and 'ia' not in tl:
        return 'imagej'
    return None


def is_twoway_paired(ttype):
    return ttype == 'TwoWay'


def extract_from_file(filepath, annotator, scale='fraction'):
    tree = ET.parse(filepath)
    root = tree.getroot()
    tables = root.findall('p:Table', NS)

    # Prefer TwoWay tables for clean pairing (skip leading 0h zero row).
    imagej_table = None
    ai_table = None
    for table in tables:
        title_node = table.find('p:Title', NS)
        title = title_node.text if title_node is not None else None
        ttype = table.attrib.get('TableType', '')
        cls = classify_table(title, ttype)
        if cls == 'imagej' and is_twoway_paired(ttype) and imagej_table is None:
            imagej_table = table
        elif cls == 'ai' and is_twoway_paired(ttype) and ai_table is None:
            ai_table = table

    # Fallback: if no TwoWay found, use XY but strip 0h row.
    use_xy_fallback = False
    if imagej_table is None or ai_table is None:
        imagej_table = ai_table = None
        for table in tables:
            title_node = table.find('p:Title', NS)
            title = title_node.text if title_node is not None else None
            ttype = table.attrib.get('TableType', '')
            cls = classify_table(title, ttype)
            if cls == 'imagej' and ttype == 'XY' and imagej_table is None:
                imagej_table = table
            elif cls == 'ai' and ttype == 'XY' and ai_table is None:
                ai_table = table
        use_xy_fallback = True

    if imagej_table is None or ai_table is None:
        print(f"WARNING: couldn't find paired tables in {filepath}")
        return pd.DataFrame()

    ij = get_table_data_positional(imagej_table)
    ai = get_table_data_positional(ai_table)

    rows = []
    timepoints = [8, 12, 24]

    for grp_idx in range(min(len(ij), len(ai))):
        ij_title, ij_reps = ij[grp_idx]
        ai_title, ai_reps = ai[grp_idx]
        group_label = (
            ij_title if (ij_title and ij_title.strip())
            else (ai_title if (ai_title and ai_title.strip()) else f"G{grp_idx + 1}")
        )

        for rep_idx in range(max(len(ij_reps), len(ai_reps))):
            ij_vals = ij_reps[rep_idx] if rep_idx < len(ij_reps) else []
            ai_vals = ai_reps[rep_idx] if rep_idx < len(ai_reps) else []

            # XY fallback: skip the 0h row.
            if use_xy_fallback:
                ij_vals = ij_vals[1:] if len(ij_vals) > 0 else ij_vals
                ai_vals = ai_vals[1:] if len(ai_vals) > 0 else ai_vals

            if all(np.isnan(v) for v in ij_vals) and all(np.isnan(v) for v in ai_vals):
                continue

            for t_idx, t in enumerate(timepoints):
                if t_idx < len(ij_vals) and t_idx < len(ai_vals):
                    rows.append({
                        'annotator': annotator,
                        'scale': scale,
                        'group': group_label,
                        'replicate': rep_idx,
                        'timepoint_h': t,
                        'imagej': ij_vals[t_idx],
                        'ai': ai_vals[t_idx],
                    })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Parse GraphPad Prism (.pzfx) annotation files into a tidy CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--input-dir', type=Path, required=True,
        help='Folder containing the .pzfx files (annotator1.pzfx, annotator2.pzfx, '
             'annotator3.pzfx, annotator1_pct.pzfx, annotator2_pct.pzfx).'
    )
    parser.add_argument(
        '--output-csv', type=Path, default=Path('paired_data_raw.csv'),
        help='Destination CSV path (default: ./paired_data_raw.csv).'
    )
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        parser.error(f"--input-dir does not exist or is not a directory: {args.input_dir}")

    files = [
        ('annotator1.pzfx',     'Annotator 1', 'fraction'),
        ('annotator2.pzfx',     'Annotator 2', 'fraction'),
        ('annotator3.pzfx',     'Annotator 3', 'fraction'),
        ('annotator1_pct.pzfx', 'Annotator 1', 'percent'),
        ('annotator2_pct.pzfx', 'Annotator 2', 'percent'),
    ]

    all_dfs = []
    for fname, annot, scale in files:
        fpath = args.input_dir / fname
        if not fpath.is_file():
            print(f"  SKIP {fname:20s} (not found in {args.input_dir})")
            continue
        df = extract_from_file(fpath, annot, scale)
        print(f"  {fname:20s} -> {len(df):4d} rows  (annotator={annot}, scale={scale})")
        all_dfs.append(df)

    if not all_dfs:
        parser.error(f"No .pzfx files found in {args.input_dir}. Nothing to do.")

    df = pd.concat(all_dfs, ignore_index=True)
    df.to_csv(args.output_csv, index=False)

    print(f"\nWrote: {args.output_csv}")
    print(f"Total: {len(df)} paired observations")
    print(f"\nBy annotator x scale:")
    print(df.groupby(['annotator', 'scale']).size().to_string())
    print(f"\nBy annotator x group x timepoint (fraction scale only):")
    print(df[df['scale'] == 'fraction'].groupby(['annotator', 'group', 'timepoint_h']).size().to_string())


if __name__ == '__main__':
    main()
