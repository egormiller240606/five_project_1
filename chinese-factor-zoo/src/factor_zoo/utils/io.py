from pathlib import Path
import yaml

def load_yaml(path):
    with open(path,"r",encoding="utf-8") as f: return yaml.safe_load(f)

def ensure_parent(path): Path(path).parent.mkdir(parents=True,exist_ok=True)
