# -*- coding: UTF-8 -*-

import os
import subprocess
import pandas as pd
import argparse
import re
import traceback
import numpy as np
from typing import List


# Repeat experiments and save results to csv
# Example: python exp.py --in_f run.sh --out_f exp.csv --n 5


def parse_args():
    parser = argparse.ArgumentParser(description="Run")
    parser.add_argument('--log_dir', nargs='?', default='./log/',
                        help='Log save dir.')
    parser.add_argument('--cmd_dir', nargs='?', default='./',
                        help='Command dir.')
    parser.add_argument('--in_f', nargs='?', default='run.sh',
                        help='Input commands.')
    parser.add_argument('--out_f', nargs='?', default='exp.csv',
                        help='Output csv.')
    parser.add_argument('--base_seed', type=int, default=0,
                        help='Random seed at the beginning.')
    parser.add_argument('--n', type=int, default=1,
                        help='Repeat times of each command.')
    parser.add_argument('--skip', type=int, default=0,
                        help='skip number.')
    return parser.parse_args()


def find_info(result: List[str]) -> dict:
    info = dict()
    prefix = ''
    for line in result:
        if line.startswith(prefix + 'final_results'):
            line = line.replace('final_results ', '')
            info['Test'] = eval(line)
    return info


def main():
    args = parse_args()
    columns = ['Model', 'Test', 'Seed', 'Run CMD']
    skip = args.skip
    df = pd.DataFrame(columns=columns)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    in_f = open(os.path.join(args.cmd_dir, args.in_f), 'r')
    lines = in_f.readlines()

    # Iterate commands
    for cmd in lines:
        cmd = cmd.strip()
        if cmd == '' or cmd.startswith('#') or cmd.startswith('export'):
            continue
        p = re.compile('--model_name (\w+)')
        model_name = p.search(cmd).group(1)

        # Repeat experiments
        for i in range(args.base_seed, args.base_seed + args.n):
            try:
                command = cmd
                if command.find(' --random_seed') == -1:
                    command += ' --random_seed ' + str(i)
                print(command)
                if skip > 0:
                    skip -= 1
                    continue
                result = subprocess.check_output(command, shell=True)
                result = result.decode('utf-8')
                result = [line.strip() for line in result.split(os.linesep)]
                info = find_info(result)
                info['Seed'] = str(i)
                info['Run CMD'] = command
                if args.n == 1:
                    info['Model'] = model_name
                row = [info[c] if c in info else '' for c in columns]
                df.loc[len(df)] = row
                df.to_csv(os.path.join(args.log_dir, args.out_f), index=False)
                print(df[columns[:5]])
            except Exception as e:
                traceback.print_exc()
                continue

        # Average results
        if args.n > 1:
            info = {'Model': model_name}
            tests = df['Test'].tolist()[-args.n:]
            tests = np.array(tests)
            avgs = tests.mean(axis=0).astype(str)
            print(avgs)
            info['Test'] = ','.join(avgs)
            row = [info[c] if c in info else '' for c in columns]
            df.loc[len(df)] = row
            print(df[columns[:5]])
        for i in range(3):
            row = [''] * len(columns)
            df.loc[len(df)] = row

        # Save results
        df.to_csv(os.path.join(args.log_dir, args.out_f), index=False)


if __name__ == '__main__':
    main()
