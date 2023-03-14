from pathlib import Path

path = Path(__file__).parent
here = Path()
for f in path.iterdir():
    if f.is_file() and f.suffix == ".ipynb":
        name = f.name
        destination = here / name
        if destination.exists():
            print(f"Ignoring {name} because file already exists.")
        else:
            destination.write_text(f.read_text())
            print(f"Copied {name} to current working directory!")
