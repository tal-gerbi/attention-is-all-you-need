import torch
import torch.nn as nn
import bpe_tokenizer


class Embedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x)


class PositionalEncoding(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


tokenizer = bpe_tokenizer.get_tokenizer()
tokens = tokenizer.encode("My name is Slim Shady").ids


tensor = torch.LongTensor(tokens)
d = tensor.dim()
s = tensor.shape
e = Embedding(tokenizer.get_vocab_size(), 512)
embedded = e(tensor)
print(embedded.shape)
print(embedded)

