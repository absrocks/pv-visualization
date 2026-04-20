#!/usr/bin/env python3
"""
foam_auto_clean.py
------------------
Periodically monitors log.interFoam, extracts the latest simulation time,
floors it to 1 decimal place, subtracts a configurable offset, and runs
foamclean to remove old time directories.

Keeps a history of executed clean times to avoid redundant runs.

Usage:
    python3 foam_auto_clean.py --interval 15 --subtract 0.2
    python3 foam_auto_clean.py --interval 15 --subtract 0.2 --logfile log.interFoam
    python3 foam_auto_clean.py --help
"""

import argparse
import math
import os
import re
import subprocess
import sys
import time


def get_latest_time(logfile):
    """Extract the latest 'Time = ...' value from an OpenFOAM log file.
    Reads from the end of the file for efficiency with large logs."""
    file_size = os.path.getsize(logfile)
    chunk_size = min(file_size, 32768)  # last 32 KB

    with open(logfile, 'r') as f:
        f.seek(max(0, file_size - chunk_size))
        content = f.read()

    matches = re.findall(r'^Time\s*=\s*([\d.]+)', content, re.MULTILINE)
    if matches:
        return float(matches[-1])
    return None


def floor_1dp(value):
    """Floor to 1 decimal place.  18.89 -> 18.8,  5.01 -> 5.0"""
    return math.floor(value * 10) / 10.0


def main():
    parser = argparse.ArgumentParser(
        description='Periodically monitor log.interFoam and run foamclean '
                    'to remove old time directories.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python3 foam_auto_clean.py --interval 15 --subtract 0.2
  python3 foam_auto_clean.py --interval 10 --subtract 0.4 --logfile log.interFoam
  python3 foam_auto_clean.py --interval 15 --subtract 0.2 --command "foamListTimes -remove -time"

workflow:
  1. Read latest 'Time = X.XX' from log file
  2. Floor to 1 decimal place  (e.g. 18.89 -> 18.8)
  3. Subtract offset           (e.g. 18.8 - 0.2 = 18.6)
  4. Run: foamclean 18.6       (only if not already executed)
  5. Sleep and repeat
""")
    parser.add_argument('--interval', type=float, required=True,
                        help='How often to check the log file (in minutes)')
    parser.add_argument('--subtract', type=float, default=0.2,
                        help='Value to subtract from floored time (default: 0.2)')
    parser.add_argument('--logfile', default='log.interFoam',
                        help='Path to OpenFOAM log file (default: log.interFoam)')
    parser.add_argument('--command', default='foamclean',
                        help='Clean command to run (default: foamclean)')
    args = parser.parse_args()

    interval_sec = args.interval * 60
    executed = set()   # history of clean_time values already processed

    print(f'foam_auto_clean started')
    print(f'  Log file  : {args.logfile}')
    print(f'  Interval  : {args.interval} min')
    print(f'  Subtract  : {args.subtract}')
    print(f'  Command   : {args.command}')
    print(f'  {"-"*50}')

    try:
        while True:
            ts = time.strftime('%Y-%m-%d %H:%M:%S')

            if not os.path.exists(args.logfile):
                print(f'  [{ts}] Waiting for {args.logfile} ...')
                time.sleep(interval_sec)
                continue

            latest = get_latest_time(args.logfile)
            if latest is None:
                print(f'  [{ts}] No "Time = ..." entry found yet')
                time.sleep(interval_sec)
                continue

            floored = floor_1dp(latest)
            clean_time = round(floored - args.subtract, 1)

            if clean_time <= 0:
                print(f'  [{ts}] Time={latest}  floored={floored}  '
                      f'clean_time={clean_time} (<=0, skipping)')
                time.sleep(interval_sec)
                continue

            if clean_time in executed:
                print(f'  [{ts}] Time={latest}  floored={floored}  '
                      f'clean_time={clean_time}  -> already executed, skipping')
            else:
                cmd = f'{args.command} {clean_time}'
                print(f'  [{ts}] Time={latest}  floored={floored}  '
                      f'clean_time={clean_time}  -> running: {cmd}')
                subprocess.run(cmd, shell=True)
                executed.add(clean_time)
                print(f'    History: {sorted(executed)}')

            time.sleep(interval_sec)

    except KeyboardInterrupt:
        print(f'\n  Stopped. Executed foamclean for times: {sorted(executed)}')
        sys.exit(0)


if __name__ == '__main__':
    main()
