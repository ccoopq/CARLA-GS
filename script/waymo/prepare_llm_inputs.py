"""
Build the LLM inputs of one scene from the Waymo converter output:
  <datadir>/track/track_info.txt  ->  <outdir>/track_info.csv  (object tracks, comma-separated)
  <datadir>/ego_pose/NNNNNN.txt   ->  <outdir>/ego_pose.txt    (JSON: frame id -> 4x4 ego pose)

Usage:
  python script/waymo/prepare_llm_inputs.py --datadir DATA_DIR/<scene> --outdir LLMCorner/data/<scene>
"""
import argparse
import json
import os

import numpy as np
import pandas as pd


def convert_track_info(txt_path, csv_path):
    with open(txt_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    header = lines[0].split()
    rows = [line.split() for line in lines[1:]]
    for i, row in enumerate(rows):
        if len(row) != len(header):
            print(f'Warning: line {i + 2} has {len(row)} columns, expected {len(header)}')

    df = pd.DataFrame(rows, columns=header)
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except (ValueError, TypeError):
            pass
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f'Wrote {len(df)} track records to {csv_path}')


def convert_ego_pose(pose_dir, out_path):
    # Only the per-frame vehicle poses (NNNNNN.txt); NNNNNN_<cam>.txt are camera poses.
    poses = {}
    for filename in sorted(os.listdir(pose_dir)):
        if filename.endswith('.txt') and '_' not in filename:
            frame_id = os.path.splitext(filename)[0].lstrip('0') or '0'
            poses[frame_id] = np.loadtxt(os.path.join(pose_dir, filename)).tolist()

    with open(out_path, 'w') as f:
        json.dump(poses, f, indent=4)
    print(f'Wrote {len(poses)} ego poses to {out_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare track_info.csv and ego_pose.txt for LLMCorner/main.py.')
    parser.add_argument('--datadir', required=True, help='processed scene folder from waymo_converter.py')
    parser.add_argument('--outdir', required=True, help='e.g. LLMCorner/data/<scene>')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    convert_track_info(os.path.join(args.datadir, 'track', 'track_info.txt'),
                       os.path.join(args.outdir, 'track_info.csv'))
    convert_ego_pose(os.path.join(args.datadir, 'ego_pose'),
                     os.path.join(args.outdir, 'ego_pose.txt'))
