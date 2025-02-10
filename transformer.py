import torch
import torch.nn as nn


class Embedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x)


tensor = torch.LongTensor([[0, 1, 2, 3], [0, 1, 2, 3]])
d = tensor.dim()
s = tensor.shape
e = Embedding(4, 5)
print(e(tensor))

