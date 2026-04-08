from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import TensorDataset

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
        self.entry_head = nn.Sequential(
            nn.Linear(channels, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.outcome_head = nn.Sequential(
            nn.Linear(channels, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        )
        self.skip_bias = nn.Parameter(torch.zeros(1, dtype=torch.float32))

    def forward(self, x: Tensor, readout_index: Tensor) -> tuple[Tensor, Tensor]:
        temporal = x.transpose(1, 2)
        encoded = self.tcn(temporal)
        if readout_index.ndim != 1:
            raise ValueError("readout_index must be 1D")
        if int(readout_index.shape[0]) != int(encoded.shape[0]):
            raise ValueError("readout_index length must match batch size")
        max_step = int(encoded.shape[2]) - 1
        gather_idx = torch.clamp(readout_index.to(dtype=torch.long), min=0, max=max_step)
        batch_idx = torch.arange(encoded.shape[0], device=encoded.device, dtype=torch.long)
        context_step = encoded[batch_idx, :, gather_idx]
        entry_logits = self.entry_head(context_step).squeeze(-1)
        outcome_logits = self.outcome_head(context_step)
        return entry_logits, outcome_logits


@dataclass(slots=True)
class SequenceTrainStats:
    batch_size: int
    max_epochs: int
    early_stopping_patience: int
    best_val_loss: float
    best_epoch: int
    epochs_ran: int
    monitor_name: str
    train_rows_total: int
    eval_rows_total: int
    train_episodes_total: int
    eval_episodes_total: int
    stopped_early: bool
    best_choice_loss: float
    best_ranking_loss: float
    best_outcome_loss: float
    final_choice_loss: float
    final_ranking_loss: float
    final_outcome_loss: float
    outcome_aux_lambda: float
    skip_bias: float
    ranking_pairs_train_total: int
    ranking_pairs_eval_total: int
    hard_negative_rows_train_total: int
    hard_negative_rows_eval_total: int
    train_positive_rate: float
    eval_positive_rate: float
    mean_sample_weight_train: float
    training_history: list[dict[str, float | int]] = field(default_factory=list)

    def to_dict(self) -> dict[str, float | int]:
        return {
            "batch_size": int(self.batch_size),
            "max_epochs": int(self.max_epochs),
            "early_stopping_patience": int(self.early_stopping_patience),
            "best_val_loss": float(self.best_val_loss),
            "best_epoch": int(self.best_epoch),
            "epochs_ran": int(self.epochs_ran),
            "monitor_name": str(self.monitor_name),
            "train_rows_total": int(self.train_rows_total),
            "eval_rows_total": int(self.eval_rows_total),
            "train_episodes_total": int(self.train_episodes_total),
            "eval_episodes_total": int(self.eval_episodes_total),
            "stopped_early": bool(self.stopped_early),
            "best_choice_loss": float(self.best_choice_loss),
            "best_ranking_loss": float(self.best_ranking_loss),
            "best_outcome_loss": float(self.best_outcome_loss),
            "final_choice_loss": float(self.final_choice_loss),
            "final_ranking_loss": float(self.final_ranking_loss),
            "final_outcome_loss": float(self.final_outcome_loss),
            "outcome_aux_lambda": float(self.outcome_aux_lambda),
            "skip_bias": float(self.skip_bias),
            "best_classification_loss": float(self.best_choice_loss),
            "best_ranking_loss_legacy": float(self.best_ranking_loss),
            "final_classification_loss": float(self.final_choice_loss),
            "final_ranking_loss_legacy": float(self.final_ranking_loss),
            "ranking_pairs_train_total": int(self.ranking_pairs_train_total),
            "ranking_pairs_eval_total": int(self.ranking_pairs_eval_total),
            "hard_negative_rows_train_total": int(self.hard_negative_rows_train_total),
            "hard_negative_rows_eval_total": int(self.hard_negative_rows_eval_total),
            "train_positive_rate": float(self.train_positive_rate),
            "eval_positive_rate": float(self.eval_positive_rate),
            "mean_sample_weight_train": float(self.mean_sample_weight_train),
        }


def train_sequence_model(
        model: SequenceTCNBinaryClassifier,
        x_train: np.ndarray,
        y_train: np.ndarray,
        sample_weight_train: np.ndarray,
        train_episode_ids: np.ndarray,
        train_decision_row_ids: np.ndarray | None = None,
        train_readout_index: np.ndarray | None = None,
        train_episode_target_row_id_map: dict[str, str | None] | None = None,
        train_outcome_targets: np.ndarray | None = None,
        train_outcome_weights: np.ndarray | None = None,
        x_eval: np.ndarray | None = None,
        y_eval: np.ndarray | None = None,
        sample_weight_eval: np.ndarray | None = None,
        eval_episode_ids: np.ndarray | None = None,
        eval_decision_row_ids: np.ndarray | None = None,
        eval_readout_index: np.ndarray | None = None,
        eval_episode_target_row_id_map: dict[str, str | None] | None = None,
        eval_outcome_targets: np.ndarray | None = None,
        eval_outcome_weights: np.ndarray | None = None,
        random_seed: int = 0,
        batch_size: int = SEQUENCE_BATCH_SIZE,
        max_epochs: int = SEQUENCE_MAX_EPOCHS,
        early_stopping_patience: int = SEQUENCE_EARLY_STOPPING_PATIENCE,
        learning_rate: float = SEQUENCE_LEARNING_RATE,
        weight_decay: float = SEQUENCE_WEIGHT_DECAY,
        outcome_aux_lambda: float = 0.25,
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
    if len(train_episode_ids) != len(y_train):
        raise ValueError("train_episode_ids length must match y_train length")
    if train_readout_index is None or len(train_readout_index) != len(y_train):
        raise ValueError("train_readout_index length must match y_train length")
    if x_eval is not None and not np.isfinite(x_eval).all():
        raise ValueError("train_sequence_model received non-finite x_eval")
    if y_eval is not None and not np.isfinite(y_eval).all():
        raise ValueError("train_sequence_model received non-finite y_eval")
    if sample_weight_eval is not None and not np.isfinite(sample_weight_eval).all():
        raise ValueError("train_sequence_model received non-finite sample_weight_eval")
    if x_eval is not None and eval_readout_index is not None and len(eval_readout_index) != len(x_eval):
        raise ValueError("eval_readout_index length must match x_eval length")
    ranking_lambda = float(ranking_lambda)
    if ranking_lambda <= 0.0:
        raise ValueError("ranking_lambda must be positive")
    outcome_aux_lambda = float(outcome_aux_lambda)
    if outcome_aux_lambda < 0.0:
        raise ValueError("outcome_aux_lambda must be non-negative")
    ranking_pairs_train = _normalize_ranking_pairs(ranking_pairs_train, "ranking_pairs_train")
    ranking_pairs_eval = _normalize_ranking_pairs(ranking_pairs_eval, "ranking_pairs_eval")
    if train_outcome_targets is None:
        train_outcome_targets = np.full(len(y_train), -1, dtype=np.int64)
    if train_outcome_weights is None:
        train_outcome_weights = np.ones(len(y_train), dtype=np.float32)
    if len(train_outcome_targets) != len(y_train):
        raise ValueError("train_outcome_targets length must match y_train length")
    if len(train_outcome_weights) != len(y_train):
        raise ValueError("train_outcome_weights length must match y_train length")
    if not np.isfinite(train_outcome_weights).all():
        raise ValueError("train_sequence_model received non-finite train_outcome_weights")
    if x_eval is not None and y_eval is not None and len(x_eval) > 0:
        if eval_outcome_targets is None:
            eval_outcome_targets = np.full(len(y_eval), -1, dtype=np.int64)
        if eval_outcome_weights is None:
            eval_outcome_weights = np.ones(len(y_eval), dtype=np.float32)
        if not np.isfinite(eval_outcome_weights).all():
            raise ValueError("train_sequence_model received non-finite eval_outcome_weights")
    torch.manual_seed(int(random_seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    x_train_t = torch.tensor(x_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    w_train_t = torch.tensor(sample_weight_train, dtype=torch.float32)
    readout_train_t = torch.tensor(train_readout_index, dtype=torch.long)
    outcome_target_train_t = torch.tensor(train_outcome_targets, dtype=torch.long)
    outcome_weight_train_t = torch.tensor(train_outcome_weights, dtype=torch.float32)
    x_train_device = x_train_t.to(device)
    y_train_device = y_train_t.to(device)
    w_train_device = w_train_t.to(device)
    readout_train_device = readout_train_t.to(device)
    outcome_target_train_device = outcome_target_train_t.to(device)
    outcome_weight_train_device = outcome_weight_train_t.to(device)
    idx_train_t = torch.arange(len(y_train), dtype=torch.long)
    train_ds = TensorDataset(
        x_train_t,
        y_train_t,
        w_train_t,
        readout_train_t,
        outcome_target_train_t,
        outcome_weight_train_t,
        idx_train_t,
    )
    episode_batches = _build_episode_index_batches(
        train_episode_ids=train_episode_ids,
        batch_size=int(batch_size),
        random_seed=int(random_seed),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    eval_data: tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor] | None = None
    monitor_name = "train_loss_fallback"
    if (
            x_eval is not None
            and y_eval is not None
            and eval_readout_index is not None
            and eval_outcome_targets is not None
            and eval_outcome_weights is not None
            and len(x_eval) > 0
    ):
        if sample_weight_eval is None:
            sample_weight_eval = np.ones(len(y_eval), dtype=np.float32)
        eval_data = (
            torch.tensor(x_eval, dtype=torch.float32, device=device),
            torch.tensor(y_eval, dtype=torch.float32, device=device),
            torch.tensor(sample_weight_eval, dtype=torch.float32, device=device),
            torch.tensor(eval_readout_index, dtype=torch.long, device=device),
            torch.tensor(eval_outcome_targets, dtype=torch.long, device=device),
            torch.tensor(eval_outcome_weights, dtype=torch.float32, device=device),
        )
        monitor_name = "eval_loss"
    best_loss = float("inf")
    best_choice_loss = 0.0
    best_ranking_loss = 0.0
    best_outcome_loss = 0.0
    final_choice_loss = 0.0
    final_ranking_loss = 0.0
    final_outcome_loss = 0.0
    best_epoch = 0
    best_state: dict[str, Tensor] | None = None
    patience_left = int(early_stopping_patience)
    epochs_ran = 0
    stopped_early = False
    training_history: list[dict[str, float | int]] = []
    for epoch in range(1, int(max_epochs) + 1):
        model.train()
        train_loss_sum = 0.0
        train_choice_sum = 0.0
        train_ranking_sum = 0.0
        train_outcome_sum = 0.0
        train_count = 0
        for batch_indices in episode_batches:
            (
                batch_x,
                batch_y,
                batch_w,
                batch_readout,
                batch_outcome_targets,
                batch_outcome_weights,
                batch_idx,
            ) = _slice_batch(train_ds, batch_indices)
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_w = batch_w.to(device)
            batch_readout = batch_readout.to(device)
            batch_outcome_targets = batch_outcome_targets.to(device)
            batch_outcome_weights = batch_outcome_weights.to(device)
            optimizer.zero_grad(set_to_none=True)
            entry_logits, outcome_logits = model(batch_x, batch_readout)
            classification_loss = _weighted_bce_loss(entry_logits, batch_y, batch_w)
            if not torch.isfinite(classification_loss):
                raise ValueError(
                    f"non-finite train classification loss detected: loss={float(classification_loss.item())}"
                )
            batch_pairs = _extract_batch_ranking_pairs_by_global_index(
                ranking_pairs_train, batch_idx.detach().cpu().numpy()
            )
            ranking_loss_batch = _pairwise_ranking_loss(
                entry_logits, batch_pairs, device=device
            )
            outcome_loss = _weighted_outcome_ce_loss(
                logits=outcome_logits,
                targets=batch_outcome_targets,
                sample_weight=batch_outcome_weights,
            )
            total_loss_batch = (
                    classification_loss
                    + ranking_lambda * ranking_loss_batch
                    + outcome_aux_lambda * outcome_loss
            )
            if not torch.isfinite(total_loss_batch):
                raise ValueError(
                    f"non-finite train total loss detected: loss={float(total_loss_batch.item())}"
                )
            total_loss_batch.backward()
            optimizer.step()
            batch_size_local = int(batch_y.shape[0])
            train_loss_sum += float(total_loss_batch.item()) * float(batch_size_local)
            train_choice_sum += float(classification_loss.item()) * float(batch_size_local)
            train_ranking_sum += float(ranking_loss_batch.item()) * float(batch_size_local)
            train_outcome_sum += float(outcome_loss.item()) * float(batch_size_local)
            train_count += batch_size_local
        train_choice_epoch = float(train_choice_sum / float(max(train_count, 1)))
        train_ranking_epoch = float(train_ranking_sum / float(max(train_count, 1)))
        train_outcome_epoch = float(train_outcome_sum / float(max(train_count, 1)))
        train_loss_epoch = float(train_loss_sum / float(max(train_count, 1)))
        model.eval()
        with torch.no_grad():
            train_entry_logits, train_outcome_logits = model(
                x_train_device, readout_train_device
            )
            train_choice_monitor = float(
                _weighted_bce_loss(train_entry_logits, y_train_device, w_train_device).item()
            )
            train_ranking_monitor = float(
                _pairwise_ranking_loss(
                    train_entry_logits, ranking_pairs_train, device=device
                ).item()
            )
            train_outcome_monitor = float(
                _weighted_outcome_ce_loss(
                    logits=train_outcome_logits,
                    targets=outcome_target_train_device,
                    sample_weight=outcome_weight_train_device,
                ).item()
            )
            train_loss_monitor = float(
                train_choice_monitor
                + ranking_lambda * train_ranking_monitor
                + outcome_aux_lambda * train_outcome_monitor
            )
            if eval_data is not None:
                eval_entry_logits, eval_outcome_logits = model(eval_data[0], eval_data[3])
                eval_choice_loss = float(
                    _weighted_bce_loss(eval_entry_logits, eval_data[1], eval_data[2]).item()
                )
                eval_ranking_loss = float(
                    _pairwise_ranking_loss(
                        eval_entry_logits, ranking_pairs_eval, device=device
                    ).item()
                )
                eval_outcome_loss = float(
                    _weighted_outcome_ce_loss(
                        logits=eval_outcome_logits,
                        targets=eval_data[4],
                        sample_weight=eval_data[5],
                    ).item()
                )
            else:
                eval_choice_loss = float(train_choice_monitor)
                eval_ranking_loss = float(train_ranking_monitor)
                eval_outcome_loss = float(train_outcome_monitor)
            eval_loss = float(
                eval_choice_loss
                + ranking_lambda * eval_ranking_loss
                + outcome_aux_lambda * eval_outcome_loss
            )
        if not np.isfinite(eval_loss):
            raise ValueError(f"non-finite eval total loss detected: eval_loss={eval_loss}")
        is_best_epoch = False
        if eval_loss < best_loss:
            best_loss = eval_loss
            best_epoch = epoch
            best_choice_loss = float(eval_choice_loss)
            best_ranking_loss = float(eval_ranking_loss)
            best_outcome_loss = float(eval_outcome_loss)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = int(early_stopping_patience)
            is_best_epoch = True
        else:
            patience_left -= 1
            if patience_left <= 0:
                epochs_ran = int(epoch)
                stopped_early = True
                final_choice_loss = float(eval_choice_loss)
                final_ranking_loss = float(eval_ranking_loss)
                final_outcome_loss = float(eval_outcome_loss)
                training_history.append(
                    {
                        "epoch": int(epoch),
                        "train_loss": float(train_loss_monitor),
                        "eval_loss": float(eval_loss),
                        "train_choice_loss": float(train_choice_epoch),
                        "train_ranking_loss": float(train_ranking_epoch),
                        "train_outcome_loss": float(train_outcome_epoch),
                        "eval_choice_loss": float(eval_choice_loss),
                        "eval_ranking_loss": float(eval_ranking_loss),
                        "eval_outcome_loss": float(eval_outcome_loss),
                        "is_best_epoch": int(is_best_epoch),
                    }
                )
                break
        epochs_ran = int(epoch)
        final_choice_loss = float(eval_choice_loss)
        final_ranking_loss = float(eval_ranking_loss)
        final_outcome_loss = float(eval_outcome_loss)
        training_history.append(
            {
                "epoch": int(epoch),
                "train_loss": float(train_loss_epoch),
                "eval_loss": float(eval_loss),
                "train_choice_loss": float(train_choice_epoch),
                "train_ranking_loss": float(train_ranking_epoch),
                "train_outcome_loss": float(train_outcome_epoch),
                "eval_choice_loss": float(eval_choice_loss),
                "eval_ranking_loss": float(eval_ranking_loss),
                "eval_outcome_loss": float(eval_outcome_loss),
                "is_best_epoch": int(is_best_epoch),
            }
        )
    if best_state is None:
        raise ValueError("sequence model training failed: best_state is None")
    model.load_state_dict(best_state)
    return SequenceTrainStats(
        batch_size=int(batch_size),
        max_epochs=int(max_epochs),
        early_stopping_patience=int(early_stopping_patience),
        best_val_loss=float(best_loss if np.isfinite(best_loss) else 0.0),
        best_epoch=int(best_epoch),
        epochs_ran=int(epochs_ran),
        monitor_name=monitor_name,
        train_rows_total=int(len(y_train)),
        eval_rows_total=int(0 if y_eval is None else len(y_eval)),
        train_episodes_total=int(np.unique(np.asarray(train_episode_ids, dtype=object).astype(str)).size),
        eval_episodes_total=(
            int(np.unique(np.asarray(eval_episode_ids, dtype=object).astype(str)).size)
            if eval_episode_ids is not None
            else 0
        ),
        stopped_early=bool(stopped_early),
        best_choice_loss=float(best_choice_loss),
        best_ranking_loss=float(best_ranking_loss),
        best_outcome_loss=float(best_outcome_loss),
        final_choice_loss=float(final_choice_loss),
        final_ranking_loss=float(final_ranking_loss),
        final_outcome_loss=float(final_outcome_loss),
        outcome_aux_lambda=float(outcome_aux_lambda),
        skip_bias=float(model.skip_bias.detach().cpu().item()),
        ranking_pairs_train_total=int(len(ranking_pairs_train["pair_weight"])),
        ranking_pairs_eval_total=int(len(ranking_pairs_eval["pair_weight"])),
        hard_negative_rows_train_total=int(hard_negative_rows_train_total),
        hard_negative_rows_eval_total=int(hard_negative_rows_eval_total),
        train_positive_rate=float(np.mean(y_train > 0.5)) if len(y_train) > 0 else 0.0,
        eval_positive_rate=(
            float(np.mean(y_eval > 0.5))
            if y_eval is not None and len(y_eval) > 0
            else 0.0
        ),
        mean_sample_weight_train=(
            float(np.mean(sample_weight_train)) if len(sample_weight_train) > 0 else 0.0
        ),
        training_history=training_history,
    )


def _episode_choice_loss(
        entry_logits: Tensor,
        episode_ids: np.ndarray,
        decision_row_ids: np.ndarray,
        target_row_id_map: dict[str, str | None],
        skip_bias: Tensor,
) -> Tensor:
    if entry_logits.numel() == 0:
        return entry_logits.new_tensor(0.0)
    episode_np = np.asarray(episode_ids, dtype=object).astype(str)
    row_id_np = np.asarray(decision_row_ids, dtype=object).astype(str)
    losses: list[Tensor] = []
    start = 0
    while start < len(episode_np):
        end = start + 1
        current_episode = str(episode_np[start])
        while end < len(episode_np) and str(episode_np[end]) == current_episode:
            end += 1
        row_logits = entry_logits[start:end]
        candidate_logits = torch.cat(
            [row_logits, skip_bias.view(1).to(device=entry_logits.device)]
        )
        target_row_id = target_row_id_map.get(current_episode)
        if target_row_id is None:
            target_idx = int(row_logits.shape[0])
        else:
            local_ids = row_id_np[start:end]
            matched = np.where(local_ids == str(target_row_id))[0]
            target_idx = int(matched[0]) if matched.size > 0 else int(row_logits.shape[0])
        target_t = torch.tensor([target_idx], dtype=torch.long, device=entry_logits.device)
        losses.append(F.cross_entropy(candidate_logits.unsqueeze(0), target_t))
        start = end
    if not losses:
        return entry_logits.new_tensor(0.0)
    return torch.stack(losses).mean()


def _weighted_bce_loss(logits: Tensor, targets: Tensor, sample_weight: Tensor) -> Tensor:
    per_row_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    weights = torch.clamp(sample_weight, min=0.0)
    denom = torch.sum(weights)
    if float(denom.item()) <= 0.0:
        return torch.zeros((), dtype=per_row_loss.dtype, device=per_row_loss.device)
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


def _weighted_outcome_ce_loss(
        logits: Tensor, targets: Tensor, sample_weight: Tensor
) -> Tensor:
    valid = targets >= 0
    if not bool(valid.any().item()):
        return logits.new_tensor(0.0)
    logits_valid = logits[valid]
    targets_valid = targets[valid]
    weights = torch.clamp(sample_weight[valid], min=0.0)
    per_row_loss = F.cross_entropy(logits_valid, targets_valid, reduction="none")
    denom = torch.sum(weights)
    if float(denom.item()) <= 0.0:
        return torch.mean(per_row_loss)
    return torch.sum(per_row_loss * weights) / denom


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


def _extract_batch_ranking_pairs_by_global_index(
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


def _build_episode_index_batches(
        train_episode_ids: np.ndarray,
        batch_size: int,
        random_seed: int,
) -> list[np.ndarray]:
    episode_ids = np.asarray(train_episode_ids, dtype=object).astype(str)
    groups: dict[str, list[int]] = {}
    for idx, episode_id in enumerate(episode_ids.tolist()):
        groups.setdefault(str(episode_id), []).append(int(idx))
    rng = np.random.default_rng(int(random_seed))
    episode_keys = np.asarray(list(groups.keys()), dtype=object)
    if episode_keys.size > 0:
        rng.shuffle(episode_keys)
    batches: list[np.ndarray] = []
    current: list[int] = []
    current_size = 0
    target_batch_size = max(int(batch_size), 1)
    for key in episode_keys.tolist():
        episode_indices = groups[str(key)]
        episode_len = len(episode_indices)
        if current and (current_size + episode_len) > target_batch_size:
            batches.append(np.asarray(current, dtype=np.int64))
            current = []
            current_size = 0
        current.extend(episode_indices)
        current_size += episode_len
        if current_size >= target_batch_size:
            batches.append(np.asarray(current, dtype=np.int64))
            current = []
            current_size = 0
    if current:
        batches.append(np.asarray(current, dtype=np.int64))
    if not batches:
        return [np.arange(len(episode_ids), dtype=np.int64)]
    return batches


def _slice_batch(
        dataset: TensorDataset, batch_indices: np.ndarray
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    idx = torch.tensor(batch_indices, dtype=torch.long)
    x_all, y_all, w_all, readout_all, outcome_target_all, outcome_weight_all, i_all = dataset.tensors
    return (
        x_all.index_select(0, idx),
        y_all.index_select(0, idx),
        w_all.index_select(0, idx),
        readout_all.index_select(0, idx),
        outcome_target_all.index_select(0, idx),
        outcome_weight_all.index_select(0, idx),
        i_all.index_select(0, idx),
    )


def predict_sequence_model_outputs(
        model: SequenceTCNBinaryClassifier, x: np.ndarray, readout_index: np.ndarray
) -> dict[str, np.ndarray]:
    if len(x) == 0:
        return {
            "p_good": np.zeros(0, dtype=np.float32),
            "p_tp_row": np.zeros(0, dtype=np.float32),
            "p_timeout_row": np.zeros(0, dtype=np.float32),
            "p_sl_row": np.zeros(0, dtype=np.float32),
        }
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        x_t = torch.tensor(x, dtype=torch.float32, device=device)
        readout_t = torch.tensor(readout_index, dtype=torch.long, device=device)
        entry_logits, outcome_logits = model(x_t, readout_t)
        p_good = torch.sigmoid(entry_logits).detach().cpu().numpy().astype(np.float32)
        outcome_probs = torch.softmax(outcome_logits, dim=1).detach().cpu().numpy().astype(
            np.float32
        )
    return {
        "p_good": p_good,
        "p_tp_row": outcome_probs[:, 0],
        "p_timeout_row": outcome_probs[:, 1],
        "p_sl_row": outcome_probs[:, 2],
    }


def predict_sequence_model_proba(
        model: SequenceTCNBinaryClassifier, x: np.ndarray, readout_index: np.ndarray
) -> np.ndarray:
    return predict_sequence_model_outputs(model, x, readout_index)["p_good"]
