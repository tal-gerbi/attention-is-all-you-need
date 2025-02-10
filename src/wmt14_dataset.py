from pprint import pprint
from pathlib import Path

from datasets import DatasetDict, load_dataset, load_from_disk


def wmt14_en_de_dataset() -> DatasetDict:
    dataset_path = Path(__file__).parent.parent / "datasets" / "wmt14" / "en-de"
    if dataset_path.exists():
        return load_from_disk(dataset_path)

    ds = load_dataset("wmt14", "de-en")
    ds.save_to_disk(dataset_path)
    return ds


if __name__ == "__main__":
    ds = wmt14_en_de_dataset()
    stats = {
        'train_set_len': len(ds['train']),
        'validation_set_len': len(ds['validation']),
        'test_set_len': len(ds['test'])
    }
    pprint(stats)
    pprint(ds['train'][:10])
