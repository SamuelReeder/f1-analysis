"""Pack and unpack the uncommitted inputs a refresh needs on a fresh machine.

python -m f1rank.cachestore pack refresh-cache.tar.gz
python -m f1rank.cachestore unpack refresh-cache.tar.gz

The scheduled GitHub Actions refresh keeps this archive as an asset of the
`refresh-cache` release and replaces it after every successful run. It holds:

- raw Jolpica responses (only the current season is refetched; past seasons
  come from these files),
- the extracted FastF1 race and sprint tables (re-extracting every race since
  2018 would take hours; only new races are downloaded),
- the race model's checkpointed fits and held-out targets (the nine validation
  fits take hours on CPU; they are reused while their training data are unchanged).

Everything in it can be regenerated from the public sources. Posterior caches
are checked against their inputs before reuse, so a stale cache costs time,
never correctness.
"""
import argparse
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PATHS = ("data/raw/jolpica", "data/raw/fastf1_tables", "data/raw/wikipedia", "data/raw/openmeteo",
         "outputs/race_total/cache")
# Not needed by a refresh: the bulk Jolpica dump (oldlaps), synthetic recovery fits and
# failed sampler attempts kept for diagnosis.
EXCLUDE = ("data/raw/jolpica/dumps/", "outputs/race_total/cache/recovery_")


def members(root: Path = ROOT) -> list[Path]:
    files = []
    for base in PATHS:
        for path in sorted((root / base).rglob("*")):
            name = path.relative_to(root).as_posix()
            if path.is_file() and not name.startswith(EXCLUDE) and ".failed_" not in path.name:
                files.append(path)
    return files


def pack(target: Path, root: Path = ROOT) -> int:
    files = members(root)
    if not files:
        raise SystemExit("Nothing to pack: run fetch and the race extraction first")
    with tarfile.open(target, "w:gz") as tar:
        for path in files:
            tar.add(path, arcname=path.relative_to(root).as_posix(), recursive=False)
    return len(files)


def unpack(source: Path, root: Path = ROOT) -> int:
    with tarfile.open(source, "r:gz") as tar:
        names = tar.getnames()
        bad = [n for n in names if not n.startswith(tuple(p + "/" for p in PATHS))]
        if bad:
            raise ValueError(f"Unexpected paths in the cache archive: {bad[:3]}")
        tar.extractall(root, filter="data")
    return len(names)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["pack", "unpack"])
    p.add_argument("archive", type=Path)
    args = p.parse_args()
    n = pack(args.archive) if args.command == "pack" else unpack(args.archive)
    print(f"{args.command}ed {n} files: {args.archive}")


if __name__ == "__main__":
    main()
