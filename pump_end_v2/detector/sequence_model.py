from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset


SEQUENCE_BATCH_SIZE = 256
SEQUENCE_MAX_EPOCHS = 40
SEQUENCE_EARLY_STOPPING_PATIENCE = 5
SEQUENCE_LEARNING_RATE = 1e-3
SEQUENCE_WEIGHT_DECAY = 1e-4
SEQUENCE_DROPOUT = 0.10
SEQUENCE_HIDDEN_CHANNELS = 64
SEQUENCE_KERNEL_SIZE = 3
SEQUENCE_DILATIONS: tuple[int, ...] = (1, 2, 4)


class CausalConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.left_padding = int((kernel_size - 1) * dilation)
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = F.pad(x, (self.left_padding, 0))
        return self.conv(x)


class TemporalResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self.proj = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        residual = self.proj(x)
        out = self.conv1(x)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.activation(out)
        out = self.dropout(out)
        return self.activation(out + residual)


class SequenceTCNBinaryClassifier(nn.Module):
    def __init__(
        self,
        input_feature_count: int,
        hidden_channels: int = SEQUENCE_HIDDEN_CHANNELS,
        kernel_size: int = SEQUENCE_KERNEL_SIZE,
        dilations: tuple[int, ...] = SEQUENCE_DILATIONS,
        dropout: float = SEQUENCE_DROPOUT,
    ):
        super().__init__()
        channels = int(hidden_channels)
        blocks: list[nn.Module] = []
        in_channels = int(input_feature_count)
        for dilation in dilations:
            blocks.append(
                TemporalResidualBlock(
                    in_channels=in_channels,
                    out_channels=channels,
                    kernel_size=int(kernel_size),
                    dilation=int(dilation),
                    dropout=float(dropout),
                )
            )
            in_channels = channels
        self.tcn = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.Linear(channels, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        temporal = x.transpose(1, 2)
        encoded = self.tcn(temporal)
        last_step = encoded[:, :, -1]
        logits = self.head(last_step).squeeze(-1)
        return logits


@dataclass(slots=True)
class SequenceTrainStats:
    train_positive_rate: float
    pos_weight: float
    batch_size: int
    max_epochs: int
    early_stopping_patience: int
    best_val_loss: float
    best_epoch: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "train_positive_rate": float(self.train_positive_rate),
            "pos_weight": float(self.pos_weight),
            "batch_size": int(self.batch_size),
            "max_epochs": int(self.max_epochs),
            "early_stopping_patience": int(self.early_stopping_patience),
            "best_val_loss": float(self.best_val_loss),
            "best_epoch": int(self.best_epoch),
        }


def train_sequence_model(
    model: SequenceTCNBinaryClassifier,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray | None,
    y_eval: np.ndarray | None,
    random_seed: int,
    batch_size: int = SEQUENCE_BATCH_SIZE,
    max_epochs: int = SEQUENCE_MAX_EPOCHS,
    early_stopping_patience: int = SEQUENCE_EARLY_STOPPING_PATIENCE,
    learning_rate: float = SEQUENCE_LEARNING_RATE,
    weight_decay: float = SEQUENCE_WEIGHT_DECAY,
) -> SequenceTrainStats:
    torch.manual_seed(int(random_seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    x_train_t = torch.tensor(x_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    train_ds = TensorDataset(x_train_t, y_train_t)
    train_loader = DataLoader(
        train_ds,
        batch_size=int(batch_size),
        shuffle=True,
        drop_last=False,
    )
    positives = float(np.sum(y_train > 0.5))
    negatives = float(len(y_train) - positives)
    pos_weight_value = float(negatives / positives) if positives > 0.0 else 1.0
    pos_weight = torch.tensor(pos_weight_value, dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    eval_data: tuple[Tensor, Tensor] | None = None
    if x_eval is not None and y_eval is not None and len(x_eval) > 0:
        eval_data = (
            torch.tensor(x_eval, dtype=torch.float32, device=device),
            torch.tensor(y_eval, dtype=torch.float32, device=device),
        )
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, Tensor] | None = None
    patience_left = int(early_stopping_patience)
    for epoch in range(1, int(max_epochs) + 1):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            if eval_data is not None:
                eval_logits = model(eval_data[0])
                eval_loss = float(criterion(eval_logits, eval_data[1]).item())
            else:
                train_logits = model(x_train_t.to(device))
                eval_loss = float(criterion(train_logits, y_train_t.to(device)).item())
        if eval_loss < best_loss:
            best_loss = eval_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = int(early_stopping_patience)
        else:
            patience_left -= 1
            if patience_left <= 0:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    train_positive_rate = float(np.mean(y_train > 0.5)) if len(y_train) > 0 else 0.0
    return SequenceTrainStats(
        train_positive_rate=train_positive_rate,
        pos_weight=pos_weight_value,
        batch_size=int(batch_size),
        max_epochs=int(max_epochs),
        early_stopping_patience=int(early_stopping_patience),
        best_val_loss=float(best_loss if np.isfinite(best_loss) else 0.0),
        best_epoch=int(best_epoch),
    )


def predict_sequence_model_proba(
    model: SequenceTCNBinaryClassifier, x: np.ndarray
) -> np.ndarray:
    if len(x) == 0:
        return np.zeros(0, dtype=np.float32)
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        x_t = torch.tensor(x, dtype=torch.float32, device=device)
        logits = model(x_t)
        probs = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)
    return probs
