import ssgetpy
import argparse

parser = argparse.ArgumentParser(description='Download Matrix Market files')

parser.add_argument('--format', type=str, default="MM")
parser.add_argument('--destpath', type=str, default="/mnt/data/SuiteSparse")

args = parser.parse_args()

result = ssgetpy.search(rowbounds=(1000, 10000000), nzbounds=(
    1000, 10000000),  isspd=False, limit=10000)
print(len(result))
result.download(destpath=args.destpath, format=args.format, extract=True)
