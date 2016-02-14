"""Download a collection of files from a list of HTTP URLs.

Example:
$ python3 src/utils/download.py data/uiowa_mis_zips.json ~/
"""

import argparse
from joblib import Parallel, delayed
import json
import os
import time
import urllib.request


def fetch_one(url, output_dir, skip_existing=True):
    output_file = os.path.join(output_dir, url.split('http://')[-1])
    if os.path.exists(output_file) and skip_existing:
        return

    outdir = os.path.dirname(output_file)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    print("[{}] Fetching: {}".format(time.asctime(), url))
    surl = urllib.parse.quote(url, safe=':./')
    urllib.request.urlretrieve(surl, output_file)


def main(manifest_file, output_dir='./', skip_existing=True, num_cpus=-1):
    resources = json.load(open(manifest_file))
    pool = Parallel(n_jobs=num_cpus)
    fx = delayed(fetch_one)
    pool(fx(url, output_dir, skip_existing) for url in resources)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "manifest_file",
        metavar="manifest_file", type=str,
        help="A JSON file of files under 'resources'.")
    parser.add_argument(
        "output_dir",
        metavar="output_dir", type=str,
        help="Output path for downloaded data.")
    parser.add_argument(
        "--skip_existing",
        metavar="skip_existing", type=bool, default=False,
        help="If True, re-download files that already exist.")
    parser.add_argument(
        "--num_cpus",
        metavar="num_cpus", type=int, default=-1,
        help="Number of CPUs to use; by default, uses all.")

    args = parser.parse_args()
    main(args.manifest_file, args.output_dir,
         skip_existing=args.skip_existing, num_cpus=args.num_cpus)
