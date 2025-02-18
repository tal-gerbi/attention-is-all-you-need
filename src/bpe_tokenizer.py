from pathlib import Path
from typing import Iterator

from datasets import DatasetDict
from tokenizers import Tokenizer
from tokenizers.implementations import ByteLevelBPETokenizer

from wmt14_dataset import wmt14_en_de_dataset


def train_tokenizer(ds: DatasetDict) -> ByteLevelBPETokenizer:
    # Build a tokenizer
    bpe_tokenizer = ByteLevelBPETokenizer()
    # bpe_tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    # bpe_tokenizer.normalizer = normalizers.Lowercase()

    # Build an iterator over this dataset
    def train_set_iterator(ds: DatasetDict) -> Iterator[str]:
        batch_length = 1000
        for i in range(0, len(ds["train"]), batch_length):
            translations = ds["train"][i:i + batch_length]["translation"]
            yield [
                t["de"]
                for t in translations
            ]
            yield [
                t["en"]
                for t in translations
            ]

    # And finally train
    bpe_tokenizer.train_from_iterator(
        train_set_iterator(ds),
        length=len(ds["train"]),
        vocab_size=50000
    )
    return bpe_tokenizer


def get_tokenizer() -> Tokenizer:
    tokenizer_json_path = Path(__file__).parent.parent / "tokenizers" / "bpe_tokenizer.tok"
    if not tokenizer_json_path.exists():
        raise Exception("tokenzier json not found")
    return Tokenizer.from_file(str(tokenizer_json_path))


if __name__ == "__main__":
    ds = wmt14_en_de_dataset()
    tokenizer = train_tokenizer(ds)

    tokenizers_dir = Path(__file__).parent.parent / "tokenizers"
    tokenizers_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(tokenizers_dir / "bpe_tokenizer.tok"))
