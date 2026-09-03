import json, pathlib

seeds_dir = pathlib.Path("results/multi_seed")
count = 0
for d in sorted(seeds_dir.iterdir()):
    if d.is_dir() and d.name.startswith("seed_"):
        f = d / "seed_results.json"
        if f.exists():
            data = json.loads(f.read_text())
            if "DRC-CL" in data:
                del data["DRC-CL"]
                f.write_text(json.dumps(data, indent=2))
                count += 1
                print(f"  {d.name}: deleted DRC-CL cache")
print(f"\nCleaned {count} seeds.")