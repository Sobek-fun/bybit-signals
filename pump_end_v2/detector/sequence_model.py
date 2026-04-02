from dataclasses import dataclass, field

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
    mean_sample_weight_train: float
    batch_size: int
    max_epochs: int
    early_stopping_patience: int
    best_val_loss: float
    best_epoch: int
    epochs_ran: int
    monitor_name: str
    train_rows_total: int
    eval_rows_total: int
    eval_positive_rate: float
    stopped_early: bool
    training_history: list[dict[str, float | int]] = field(default_factory=list)

    def to_dict(self) -> dict[str, float | int]:
        return {
            "train_positive_rate": float(self.train_positive_rate),
            "mean_sample_weight_train": float(self.mean_sample_weight_train),
            "batch_size": int(self.batch_size),
            "max_epochs": int(self.max_epochs),
            "early_stopping_patience": int(self.early_stopping_patience),
            "best_val_loss": float(self.best_val_loss),
            "best_epoch": int(self.best_epoch),
            "epochs_ran": int(self.epochs_ran),
            "monitor_name": str(self.monitor_name),
            "train_rows_total": int(self.train_rows_total),
            "eval_rows_total": int(self.eval_rows_total),
            "eval_positive_rate": float(self.eval_positive_rate),
            "stopped_early": bool(self.stopped_early),
        }


def train_sequence_model(
    model: SequenceTCNBinaryClassifier,
    x_train: np.ndarray,
    y_train: np.ndarray,
    sample_weight_train: np.ndarray,
    x_eval: np.ndarray | None,
    y_eval: np.ndarray | None,
    sample_weight_eval: np.ndarray | None,
    random_seed: int,
    batch_size: int = SEQUENCE_BATCH_SIZE,
    max_epochs: int = SEQUENCE_MAX_EPOCHS,
    early_stopping_patience: int = SEQUENCE_EARLY_STOPPING_PATIENCE,
    learning_rate: float = SEQUENCE_LEARNING_RATE,
    weight_decay: float = SEQUENCE_WEIGHT_DECAY,
) -> SequenceTrainStats:
    if not np.isfinite(x_train).all():
        raise ValueError("train_sequence_model received non-finite x_train")
    if not np.isfinite(y_train).all():
        raise ValueError("train_sequence_model received non-finite y_train")
    if not np.isfinite(sample_weight_train).all():
        raise ValueError("train_sequence_model received non-finite sample_weight_train")
    if x_eval is not None and not np.isfinite(x_eval).all():
        raise ValueError("train_sequence_model received non-finite x_eval")
    if y_eval is not None and not np.isfinite(y_eval).all():
        raise ValueError("train_sequence_model received non-finite y_eval")
    if sample_weight_eval is not None and not np.isfinite(sample_weight_eval).all():
        raise ValueError("train_sequence_model received non-finite sample_weight_eval")
    torch.manual_seed(int(random_seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    x_train_t = torch.tensor(x_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    w_train_t = torch.tensor(sample_weight_train, dtype=torch.float32)
    train_ds = TensorDataset(x_train_t, y_train_t, w_train_t)
    train_loader = DataLoader(
        train_ds,
        batch_size=int(batch_size),
        shuffle=True,
        drop_last=False,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    eval_data: tuple[Tensor, Tensor, Tensor] | None = None
    monitor_name = "train_loss_fallback"
    if x_eval is not None and y_eval is not None and len(x_eval) > 0:
        if sample_weight_eval is None:
            sample_weight_eval = np.ones(len(y_eval), dtype=np.float32)
        eval_data = (
            torch.tensor(x_eval, dtype=torch.float32, device=device),
            torch.tensor(y_eval, dtype=torch.float32, device=device),
            torch.tensor(sample_weight_eval, dtype=torch.float32, device=device),
        )
        monitor_name = "eval_loss"
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, Tensor] | None = None
    patience_left = int(early_stopping_patience)
    epochs_ran = 0
    stopped_early = False
    training_history: list[dict[str, float | int]] = []
    for epoch in range(1, int(max_epochs) + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for batch_x, batch_y, batch_w in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_w = batch_w.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = _weighted_bce_loss(logits, batch_y, batch_w)
            if not torch.isfinite(loss):
                raise ValueError(f"non-finite train loss detected: loss={float(loss.item())}")
            loss.backward()
            optimizer.step()
            batch_size_local = int(batch_y.shape[0])
            train_loss_sum += float(loss.item()) * float(batch_size_local)
            train_count += batch_size_local
        train_loss = float(train_loss_sum / float(max(train_count, 1)))
        model.eval()
        with torch.no_grad():
            if eval_data is not None:
                eval_logits = model(eval_data[0])
                eval_loss = float(
                    _weighted_bce_loss(eval_logits, eval_data[1], eval_data[2]).item()
                )
            else:
                train_logits = model(x_train_t.to(device))
                eval_loss = float(
                    _weighted_bce_loss(
                        train_logits,
                        y_train_t.to(device),
                        w_train_t.to(device),
                    ).item()
                )
        if not np.isfinite(eval_loss):
            raise ValueError(f"non-finite eval loss detected: eval_loss={eval_loss}")
        is_best_epoch = False
        if eval_loss < best_loss:
            best_loss = eval_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = int(early_stopping_patience)
            is_best_epoch = True
        else:
            patience_left -= 1
            if patience_left <= 0:
                epochs_ran = int(epoch)
                stopped_early = True
                training_history.append(
                    {
                        "epoch": int(epoch),
                        "train_loss": float(train_loss),
                        "eval_loss": float(eval_loss),
                        "is_best_epoch": int(is_best_epoch),
                    }
                )
                break
        epochs_ran = int(epoch)
        training_history.append(
            {
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "eval_loss": float(eval_loss),
                "is_best_epoch": int(is_best_epoch),
            }
        )
    if best_state is None:
        raise ValueError("sequence model training failed: best_state is None")
    model.load_state_dict(best_state)
    train_positive_rate = float(np.mean(y_train > 0.5)) if len(y_train) > 0 else 0.0
    eval_positive_rate = (
        float(np.mean(y_eval > 0.5))
        if y_eval is not None and len(y_eval) > 0
        else 0.0
    )
    return SequenceTrainStats(
        train_positive_rate=train_positive_rate,
        mean_sample_weight_train=float(np.mean(sample_weight_train)),
        batch_size=int(batch_size),
        max_epochs=int(max_epochs),
        early_stopping_patience=int(early_stopping_patience),
        best_val_loss=float(best_loss if np.isfinite(best_loss) else 0.0),
        best_epoch=int(best_epoch),
        epochs_ran=int(epochs_ran),
        monitor_name=monitor_name,
        train_rows_total=int(len(y_train)),
        eval_rows_total=int(0 if y_eval is None else len(y_eval)),
        eval_positive_rate=float(eval_positive_rate),
        stopped_early=bool(stopped_early),
        training_history=training_history,
    )


def _weighted_bce_loss(logits: Tensor, targets: Tensor, sample_weight: Tensor) -> Tensor:
    per_row_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    weights = torch.clamp(sample_weight, min=0.0)
    denom = torch.sum(weights)
    if float(denom.item()) <= 0.0:
        return torch.mean(per_row_loss)
    return torch.sum(per_row_loss * weights) / denom


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
