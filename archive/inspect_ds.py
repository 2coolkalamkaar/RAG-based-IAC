
from datasets import load_dataset
try:
    ds = load_dataset("autoiac-project/iac-eval")
    print("Dataset keys:", ds.keys())
    print("Columns:", ds[list(ds.keys())[0]].column_names)
    print("Example entry:", ds[list(ds.keys())[0]][0])
except Exception as e:
    print(e)
