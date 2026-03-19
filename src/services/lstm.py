import torch
from torch import nn


class NextTokenLSTM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
        )
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim, vocab_size)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.1)

        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
                hidden_size = param.shape[0] // 4
                param.data[hidden_size : 2 * hidden_size].fill_(1.0)

        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        x = self.embedding(input_ids)
        lstm_out, hidden_state = self.lstm(x, hidden)
        last_step = lstm_out[:, -1, :]
        logits = self.output(self.dropout(last_step))
        return logits, hidden_state

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        do_sample: bool = True,
    ) -> torch.Tensor:
        if max_new_tokens <= 0:
            return input_ids
        if temperature <= 0:
            raise ValueError("temperature must be > 0")

        self.eval()
        generated = input_ids
        hidden = None

        for _ in range(max_new_tokens):
            logits, hidden = self.forward(generated[:, -1:].contiguous(), hidden)
            logits = logits / temperature

            if do_sample:
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)  # [B, 1]
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)  # [B, 1]

            generated = torch.cat([generated, next_token], dim=1)

        return generated
