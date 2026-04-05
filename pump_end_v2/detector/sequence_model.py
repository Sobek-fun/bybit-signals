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
    best_classification_loss: float
    best_ranking_loss: float
    final_classification_loss: float
    final_ranking_loss: float
    ranking_pairs_train_total: int
    ranking_pairs_eval_total: int
    hard_negative_rows_train_total: int
    hard_negative_rows_eval_total: int
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
            "best_classification_loss": float(self.best_classification_loss),
            "best_ranking_loss": float(self.best_ranking_loss),
            "final_classification_loss": float(self.final_classification_loss),
            "final_ranking_loss": float(self.final_ranking_loss),
            "ranking_pairs_train_total": int(self.ranking_pairs_train_total),
            "ranking_pairs_eval_total": int(self.ranking_pairs_eval_total),
            "hard_negative_rows_train_total": int(self.hard_negative_rows_train_total),
            "hard_negative_rows_eval_total": int(self.hard_negative_rows_eval_total),
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
    ranking_pairs_train: dict[str, np.ndarray] | None = None,
    ranking_pairs_eval: dict[str, np.ndarray] | None = None,
    ranking_lambda: float = 0.5,
    hard_negative_rows_train_total: int = 0,
    hard_negative_rows_eval_total: int = 0,
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
    ranking_lambda = float(ranking_lambda)
    if ranking_lambda <= 0.0:
        raise ValueError("ranking_lambda must be positive")
    ranking_pairs_train = _normalize_ranking_pairs(ranking_pairs_train, "ranking_pairs_train")
    ranking_pairs_eval = _normalize_ranking_pairs(ranking_pairs_eval, "ranking_pairs_eval")
    torch.manual_seed(int(random_seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    x_train_t = torch.tensor(x_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    w_train_t = torch.tensor(sample_weight_train, dtype=torch.float32)
    idx_train_t = torch.arange(len(y_train), dtype=torch.long)
    x_train_device = x_train_t.to(device)
    y_train_device = y_train_t.to(device)
    w_train_device = w_train_t.to(device)
    train_ds = TensorDataset(x_train_t, y_train_t, w_train_t, idx_train_t)
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
    best_classification_loss = 0.0
    best_ranking_loss = 0.0
    final_classification_loss = 0.0
    final_ranking_loss = 0.0
    best_epoch = 0
    best_state: dict[str, Tensor] | None = None
    patience_left = int(early_stopping_patience)
    epochs_ran = 0
    stopped_early = False
    training_history: list[dict[str, float | int]] = []
    for epoch in range(1, int(max_epochs) + 1):
        model.train()
        train_loss_sum = 0.0
        train_classification_sum = 0.0
        train_ranking_sum = 0.0
        train_count = 0
        for batch_x, batch_y, batch_w, batch_idx in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_w = batch_w.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            classification_loss = _weighted_bce_loss(logits, batch_y, batch_w)
            if not torch.isfinite(classification_loss):
                raise ValueError(
                    f"non-finite train classification loss detected: loss={float(classification_loss.item())}"
                )
            batch_pairs = _extract_batch_ranking_pairs(
                ranking_pairs_train, batch_idx.detach().cpu().numpy()
            )
            ranking_loss_batch = _pairwise_ranking_loss(logits, batch_pairs, device=device)
            total_loss_batch = classification_loss + ranking_lambda * ranking_loss_batch
            if not torch.isfinite(total_loss_batch):
                raise ValueError(
                    f"non-finite train total loss detected: loss={float(total_loss_batch.item())}"
                )
            total_loss_batch.backward()
            optimizer.step()
            batch_size_local = int(batch_y.shape[0])
            train_loss_sum += float(total_loss_batch.item()) * float(batch_size_local)
            train_classification_sum += float(classification_loss.item()) * float(batch_size_local)
            train_ranking_sum += float(ranking_loss_batch.item()) * float(batch_size_local)
            train_count += batch_size_local
        train_classification_epoch = float(
            train_classification_sum / float(max(train_count, 1))
        )
        train_ranking_epoch = float(train_ranking_sum / float(max(train_count, 1)))
        train_loss_epoch = float(train_loss_sum / float(max(train_count, 1)))
        model.eval()
        with torch.no_grad():
            train_logits_monitor = model(x_train_device)
            train_classification_monitor = float(
                _weighted_bce_loss(train_logits_monitor, y_train_device, w_train_device).item()
            )
            train_ranking_monitor = float(
                _pairwise_ranking_loss(
                    train_logits_monitor, ranking_pairs_train, device=device
                ).item()
            )
            train_loss_monitor = float(
                train_classification_monitor + ranking_lambda * train_ranking_monitor
            )
            if eval_data is not None:
                eval_logits = model(eval_data[0])
                eval_classification_loss = float(
                    _weighted_bce_loss(eval_logits, eval_data[1], eval_data[2]).item()
                )
                eval_ranking_loss = float(
                    _pairwise_ranking_loss(
                        eval_logits, ranking_pairs_eval, device=device
                    ).item()
                )
            else:
                eval_classification_loss = float(train_classification_monitor)
                eval_ranking_loss = float(train_ranking_monitor)
            eval_loss = float(eval_classification_loss + ranking_lambda * eval_ranking_loss)
        if not np.isfinite(eval_loss):
            raise ValueError(f"non-finite eval total loss detected: eval_loss={eval_loss}")
        is_best_epoch = False
        if eval_loss < best_loss:
            best_loss = eval_loss
            best_epoch = epoch
            best_classification_loss = float(eval_classification_loss)
            best_ranking_loss = float(eval_ranking_loss)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = int(early_stopping_patience)
            is_best_epoch = True
        else:
            patience_left -= 1
            if patience_left <= 0:
                epochs_ran = int(epoch)
                stopped_early = True
                final_classification_loss = float(eval_classification_loss)
                final_ranking_loss = float(eval_ranking_loss)
                training_history.append(
                    {
                        "epoch": int(epoch),
                        "train_loss": float(train_loss_monitor),
                        "eval_loss": float(eval_loss),
                        "train_classification_loss": float(train_classification_epoch),
                        "train_ranking_loss": float(train_ranking_epoch),
                        "eval_classification_loss": float(eval_classification_loss),
                        "eval_ranking_loss": float(eval_ranking_loss),
                        "is_best_epoch": int(is_best_epoch),
                    }
                )
                break
        epochs_ran = int(epoch)
        final_classification_loss = float(eval_classification_loss)
        final_ranking_loss = float(eval_ranking_loss)
        training_history.append(
            {
                "epoch": int(epoch),
                "train_loss": float(train_loss_epoch),
                "eval_loss": float(eval_loss),
                "train_classification_loss": float(train_classification_epoch),
                "train_ranking_loss": float(train_ranking_epoch),
                "eval_classification_loss": float(eval_classification_loss),
                "eval_ranking_loss": float(eval_ranking_loss),
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
        best_classification_loss=float(best_classification_loss),
        best_ranking_loss=float(best_ranking_loss),
        final_classification_loss=float(final_classification_loss),
        final_ranking_loss=float(final_ranking_loss),
        ranking_pairs_train_total=int(ranking_pairs_train["pair_weight"].size),
        ranking_pairs_eval_total=int(ranking_pairs_eval["pair_weight"].size),
        hard_negative_rows_train_total=int(hard_negative_rows_train_total),
        hard_negative_rows_eval_total=int(hard_negative_rows_eval_total),
        training_history=training_history,
    )


def _weighted_bce_loss(logits: Tensor, targets: Tensor, sample_weight: Tensor) -> Tensor:
    per_row_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    weights = torch.clamp(sample_weight, min=0.0)
    denom = torch.sum(weights)
    if float(denom.item()) <= 0.0:
        return torch.mean(per_row_loss)
    return torch.sum(per_row_loss * weights) / denom


def _pairwise_ranking_loss(
    logits: Tensor, ranking_pairs: dict[str, np.ndarray], device: torch.device
) -> Tensor:
    better_idx = ranking_pairs["better_idx"]
    worse_idx = ranking_pairs["worse_idx"]
    pair_weight = ranking_pairs["pair_weight"]
    if pair_weight.size == 0:
        return logits.new_tensor(0.0)
    better_idx_t = torch.tensor(better_idx, dtype=torch.long, device=device)
    worse_idx_t = torch.tensor(worse_idx, dtype=torch.long, device=device)
    pair_weight_t = torch.tensor(pair_weight, dtype=torch.float32, device=device)
    better_logits = logits.index_select(0, better_idx_t)
    worse_logits = logits.index_select(0, worse_idx_t)
    diff = better_logits - worse_logits
    per_pair_loss = F.softplus(-diff)
    weights = torch.clamp(pair_weight_t, min=0.0)
    denom = torch.sum(weights)
    if float(denom.item()) <= 0.0:
        return torch.mean(per_pair_loss)
    return torch.sum(per_pair_loss * weights) / denom


def _normalize_ranking_pairs(
    ranking_pairs: dict[str, np.ndarray] | None, name: str
) -> dict[str, np.ndarray]:
    if ranking_pairs is None:
        return {
            "better_idx": np.zeros(0, dtype=np.int64),
            "worse_idx": np.zeros(0, dtype=np.int64),
            "pair_weight": np.zeros(0, dtype=np.float32),
        }
    better_idx = np.asarray(ranking_pairs.get("better_idx", np.zeros(0)), dtype=np.int64)
    worse_idx = np.asarray(ranking_pairs.get("worse_idx", np.zeros(0)), dtype=np.int64)
    pair_weight = np.asarray(
        ranking_pairs.get("pair_weight", np.zeros(0)), dtype=np.float32
    )
    if not (len(better_idx) == len(worse_idx) == len(pair_weight)):
        raise ValueError(f"{name} arrays length mismatch")
    if len(better_idx) > 0 and np.any(better_idx < 0):
        raise ValueError(f"{name}.better_idx must be non-negative")
    if len(worse_idx) > 0 and np.any(worse_idx < 0):
        raise ValueError(f"{name}.worse_idx must be non-negative")
    if not np.isfinite(pair_weight).all():
        raise ValueError(f"{name}.pair_weight must be finite")
    return {
        "better_idx": better_idx,
        "worse_idx": worse_idx,
        "pair_weight": pair_weight,
    }


def _extract_batch_ranking_pairs(
    ranking_pairs: dict[str, np.ndarray], batch_indices: np.ndarray
) -> dict[str, np.ndarray]:
    if batch_indices.size == 0:
        return {
            "better_idx": np.zeros(0, dtype=np.int64),
            "worse_idx": np.zeros(0, dtype=np.int64),
            "pair_weight": np.zeros(0, dtype=np.float32),
        }
    better_global = ranking_pairs["better_idx"]
    worse_global = ranking_pairs["worse_idx"]
    pair_weight = ranking_pairs["pair_weight"]
    if pair_weight.size == 0:
        return {
            "better_idx": np.zeros(0, dtype=np.int64),
            "worse_idx": np.zeros(0, dtype=np.int64),
            "pair_weight": np.zeros(0, dtype=np.float32),
        }
    local_pos = {int(global_idx): int(local_idx) for local_idx, global_idx in enumerate(batch_indices)}
    better_local: list[int] = []
    worse_local: list[int] = []
    weights_local: list[float] = []
    for b_idx, w_idx, weight in zip(better_global, worse_global, pair_weight, strict=False):
        b_local = local_pos.get(int(b_idx))
        w_local = local_pos.get(int(w_idx))
        if b_local is None or w_local is None:
            continue
        better_local.append(int(b_local))
        worse_local.append(int(w_local))
        weights_local.append(float(weight))
    return {
        "better_idx": np.asarray(better_local, dtype=np.int64),
        "worse_idx": np.asarray(worse_local, dtype=np.int64),
        "pair_weight": np.asarray(weights_local, dtype=np.float32),
    }


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
