"""
ARMADA end-to-end training/evaluation script (EMBER feature-vector variant).

Pipeline:
  1. Load EMBER X_train/y_train (source domain, labeled) and X_test/y_test
     (target domain -- used UNLABELED during adaptation, labels only for eval).
  2. Preprocess per feature-group (log1p + standardize + missing-sentinel mask).
  3. Train encoder + classifier + dual-discriminator (marginal + conditional)
     adversarial domain adaptation, source labeled / target unlabeled.
  4. Evaluate: (a) no-TTT baseline, (b) with TTT adaptation at inference.

Run:
    python armada_train_eval.py --data_dir /path/to/ember/ --epochs 15
"""

import argparse
import copy
import gc
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# ----------------------------------------------------------------------
# 1. EMBER feature-group schema (official boundaries, feature_version=2)
# ----------------------------------------------------------------------
EMBER_GROUP_SLICES = {
    "byte_hist":    slice(0, 256),
    "byte_entropy": slice(256, 512),
    "strings":      slice(512, 616),
    "general":      slice(616, 626),
    "header":       slice(626, 688),
    "section":      slice(688, 943),
    "imports":      slice(943, 2223),
    "exports":      slice(2223, 2351),
    "data_dirs":    slice(2351, 2381),
}
EMBER_GROUP_DIMS = {name: s.stop - s.start for name, s in EMBER_GROUP_SLICES.items()}
LOG_TRANSFORM_GROUPS = {"byte_hist", "byte_entropy", "strings", "imports", "exports"}
SENTINEL_VALUE = -1.0
TOTAL_DIM = 2381


# ----------------------------------------------------------------------
# 2. Data loading
# ----------------------------------------------------------------------
def load_ember_split(data_dir, split, max_samples=None, seed=42):
    """split: 'train' or 'test'. Returns (X, y) as numpy arrays.
    max_samples: if set, randomly subsample N rows (not just the first N --
    EMBER's on-disk row order is not guaranteed class-balanced or representative,
    so a random subsample gives a much more honest smoke test)."""
    X_path = os.path.join(data_dir, f"X_{split}.dat")
    y_path = os.path.join(data_dir, f"y_{split}.dat")
    X = np.memmap(X_path, dtype=np.float32, mode="r").reshape(-1, TOTAL_DIM)
    y = np.memmap(y_path, dtype=np.float32, mode="r")
    if max_samples is not None and max_samples < X.shape[0]:
        rng = np.random.default_rng(seed)
        idx = rng.choice(X.shape[0], size=max_samples, replace=False)
        idx.sort()  # keep memmap reads sequential-ish
        X = X[idx]
        y = y[idx]
    return np.array(X), np.array(y)


# ----------------------------------------------------------------------
# 3. Preprocessing (per feature-group)
# ----------------------------------------------------------------------
def preprocess_group(X_group, group_name, stats=None, fit=False):
    X_group = X_group.astype(np.float32).copy()
    missing_mask = (X_group == SENTINEL_VALUE).astype(np.float32)
    X_group[X_group == SENTINEL_VALUE] = 0.0

    if group_name in LOG_TRANSFORM_GROUPS:
        X_group = np.log1p(np.clip(X_group, a_min=0, a_max=None))

    if fit:
        mean = X_group.mean(axis=0, keepdims=True)
        std = X_group.std(axis=0, keepdims=True) + 1e-6
        stats = {"mean": mean, "std": std}
    X_scaled = (X_group - stats["mean"]) / stats["std"]
    X_out = np.concatenate([X_scaled, missing_mask], axis=1)
    return X_out, stats


def preprocess_all_groups(X_raw, group_slices=EMBER_GROUP_SLICES, fit_stats=None, fit=False):
    processed, stats_out = {}, {}
    for name, s in group_slices.items():
        group_stats = fit_stats[name] if (fit_stats and not fit) else None
        X_group, stats = preprocess_group(X_raw[:, s], name, stats=group_stats, fit=fit)
        processed[name] = X_group
        stats_out[name] = stats
    return processed, stats_out


def groups_to_tensor_dict(group_arrays, device):
    return {k: torch.tensor(v, dtype=torch.float32, device=device) for k, v in group_arrays.items()}


# ----------------------------------------------------------------------
# 4. Encoder (grouped self-attention over EMBER feature groups)
# ----------------------------------------------------------------------
class EmberGroupedEncoder(nn.Module):
    def __init__(self, group_dims, d_model=128, n_heads=4, n_layers=2, out_dim=256):
        super().__init__()
        self.group_names = list(group_dims.keys())
        self.norms = nn.ModuleDict({name: nn.LayerNorm(dim) for name, dim in group_dims.items()})
        self.projections = nn.ModuleDict({name: nn.Linear(dim, d_model) for name, dim in group_dims.items()})
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=0.1, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.pool = nn.Linear(d_model * len(group_dims), out_dim)

    def forward(self, group_tensors):
        tokens = []
        for name in self.group_names:
            x = self.norms[name](group_tensors[name])
            x = self.projections[name](x)
            tokens.append(x)
        tokens = torch.stack(tokens, dim=1)
        encoded = self.transformer(tokens)
        flat = encoded.flatten(start_dim=1)
        return self.pool(flat)


class Classifier(nn.Module):
    def __init__(self, in_dim=256, hidden=128, n_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)


# ----------------------------------------------------------------------
# 5. Dual-discriminator adversarial domain adaptation
# ----------------------------------------------------------------------
class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambd, None


def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x, lambd)


class MarginalDiscriminator(nn.Module):
    def __init__(self, in_dim=256, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, feat, lambd):
        return self.net(grad_reverse(feat, lambd)).squeeze(-1)


class ConditionalDiscriminator(nn.Module):
    def __init__(self, feat_dim=256, n_classes=2, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim * n_classes, hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, feat, class_probs, lambd):
        joint = torch.bmm(class_probs.unsqueeze(2), feat.unsqueeze(1))
        joint = joint.view(joint.size(0), -1)
        return self.net(grad_reverse(joint, lambd)).squeeze(-1)


def grl_lambda(step, total_steps, gamma=10.0):
    p = step / max(total_steps, 1)
    return float(2.0 / (1.0 + np.exp(-gamma * p)) - 1.0)


# ----------------------------------------------------------------------
# 6. Test-time training (masked feature reconstruction)
# ----------------------------------------------------------------------
class MaskedReconstructionHead(nn.Module):
    def __init__(self, emb_dim=256, input_dim=None, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(emb_dim, hidden), nn.ReLU(), nn.Linear(hidden, input_dim))

    def forward(self, embedding):
        return self.net(embedding)


def mask_input(X, mask_ratio=0.15):
    mask = (torch.rand_like(X) < mask_ratio).float()
    return X * (1 - mask), mask, X


def ssl_loss(recon, target, mask):
    return F.mse_loss(recon * mask, target * mask, reduction="sum") / (mask.sum() + 1e-6)


class TTTWrapper:
    def __init__(self, encoder, classifier, recon_head, lr=1e-4, steps=5, mask_ratio=0.15):
        self.base_encoder_state = copy.deepcopy(encoder.state_dict())
        self.encoder = encoder
        self.classifier = classifier
        self.recon_head = recon_head
        self.lr = lr
        self.steps = steps
        self.mask_ratio = mask_ratio

    def reset_to_base(self):
        self.encoder.load_state_dict(self.base_encoder_state)

    def adapt_and_predict(self, X_flat_batch):
        # X_flat_batch here is the CONCATENATED processed group tensor (for masking),
        # so reconstruction target dim must match encoder input width.
        params = list(self.encoder.parameters()) + list(self.recon_head.parameters())
        optimizer = torch.optim.SGD(params, lr=self.lr)
        self.encoder.train()
        loss_val = 0.0
        for _ in range(self.steps):
            optimizer.zero_grad()
            X_masked, mask, target = mask_input(X_flat_batch, self.mask_ratio)
            emb = self.encoder.forward_flat(X_masked)
            recon = self.recon_head(emb)
            loss = ssl_loss(recon, target, mask)
            if not torch.isfinite(loss):
                # Don't silently continue on a diverged step -- reset this batch's
                # adaptation back to the base checkpoint and skip further TTT steps,
                # so a bad batch degrades to the (still valid) base-model prediction
                # rather than producing a NaN that would corrupt reported metrics.
                self.reset_to_base()
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()
            loss_val = loss.item()
        self.encoder.eval()
        with torch.no_grad():
            emb = self.encoder.forward_flat(X_flat_batch)
            logits = self.classifier(emb)
        return logits, loss_val


# ----------------------------------------------------------------------
# Helper: flatten group dict <-> tensor, and a forward_flat() for TTT masking
# ----------------------------------------------------------------------
def attach_forward_flat(encoder, group_dims):
    """TTT masks a flat vector, so add a forward_flat() that re-splits into groups."""
    boundaries = []
    start = 0
    for name, dim in group_dims.items():
        boundaries.append((name, start, start + dim))
        start += dim

    def forward_flat(self, X_flat):
        group_tensors = {name: X_flat[:, s:e] for name, s, e in boundaries}
        return self._forward_groups(group_tensors)

    encoder._forward_groups = encoder.forward
    encoder.forward_flat = forward_flat.__get__(encoder)
    return encoder


def groups_dict_to_flat(group_tensor_dict, group_names):
    return torch.cat([group_tensor_dict[name] for name in group_names], dim=1)


# ----------------------------------------------------------------------
# 7. Training loop
# ----------------------------------------------------------------------
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading EMBER data...")
    if args.max_samples is not None:
        print(f"SMOKE TEST MODE: loading only the first {args.max_samples} rows of each split")
    X_train_raw, y_train_raw = load_ember_split(args.data_dir, "train", max_samples=args.max_samples)

    # Source domain: labeled training samples only (drop unlabeled y == -1)
    labeled_mask = y_train_raw != -1
    X_src_raw, y_src = X_train_raw[labeled_mask], y_train_raw[labeled_mask].astype(np.int64)
    del X_train_raw, y_train_raw, labeled_mask
    gc.collect()

    X_test_raw, y_test_raw = load_ember_split(args.data_dir, "test", max_samples=args.max_samples)
    # Target domain: test set features used UNLABELED for adaptation.
    # y_test is held out and used only at evaluation time.
    X_tgt_raw = X_test_raw
    y_test = y_test_raw.astype(np.int64)
    del X_test_raw, y_test_raw
    gc.collect()

    n_src_raw = X_src_raw.shape[0]
    n_tgt_raw = X_tgt_raw.shape[0]
    print(f"Source (train, labeled): {n_src_raw} samples")
    print(f"Target (test, for adaptation, unlabeled): {n_tgt_raw} samples")

    # Preprocess: fit stats on SOURCE only, apply same stats to target/test.
    # Free each raw array immediately after it's consumed -- on 8GB machines,
    # holding raw + processed + tensor copies simultaneously is what causes OOM kills.
    src_groups, fit_stats = preprocess_all_groups(X_src_raw, fit=True)
    del X_src_raw
    gc.collect()
    tgt_groups, _ = preprocess_all_groups(X_tgt_raw, fit_stats=fit_stats, fit=False)
    del X_tgt_raw
    gc.collect()

    processed_dims = {name: dim * 2 for name, dim in EMBER_GROUP_DIMS.items()}  # scaled + mask

    encoder = EmberGroupedEncoder(processed_dims, d_model=args.d_model,
                                   n_heads=4, n_layers=2, out_dim=args.emb_dim).to(device)
    encoder = attach_forward_flat(encoder, processed_dims)
    classifier = Classifier(in_dim=args.emb_dim, n_classes=2).to(device)
    d_marg = MarginalDiscriminator(in_dim=args.emb_dim).to(device)
    d_cond = ConditionalDiscriminator(feat_dim=args.emb_dim, n_classes=2).to(device)

    params = (list(encoder.parameters()) + list(classifier.parameters())
              + list(d_marg.parameters()) + list(d_cond.parameters()))
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=1e-5)

    n_src = n_src_raw
    n_tgt = n_tgt_raw
    if args.batch_size > n_src:
        print(f"WARNING: batch_size ({args.batch_size}) > available source samples ({n_src}); "
              f"reducing batch_size to {n_src} for this run.")
        args.batch_size = n_src
    steps_per_epoch = max(1, n_src // args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    global_step = 0

    src_tensors = groups_to_tensor_dict(src_groups, device)
    tgt_tensors = groups_to_tensor_dict(tgt_groups, device)
    del src_groups, tgt_groups
    gc.collect()
    y_src_t = torch.tensor(y_src, dtype=torch.long, device=device)

    print("Starting training...")
    for epoch in range(args.epochs):
        perm_src = torch.randperm(n_src, device=device)
        perm_tgt = torch.randperm(n_tgt, device=device)
        epoch_losses = {"cls": 0.0, "marg": 0.0, "cond": 0.0}

        for step in range(steps_per_epoch):
            idx_src = perm_src[step * args.batch_size:(step + 1) * args.batch_size]
            idx_tgt = perm_tgt[(step * args.batch_size) % n_tgt: (step * args.batch_size) % n_tgt + args.batch_size]
            if idx_tgt.shape[0] < args.batch_size:
                idx_tgt = perm_tgt[:args.batch_size]

            batch_src = {k: v[idx_src] for k, v in src_tensors.items()}
            batch_tgt = {k: v[idx_tgt] for k, v in tgt_tensors.items()}
            y_batch = y_src_t[idx_src]

            lambd = grl_lambda(global_step, total_steps)

            optimizer.zero_grad()
            feat_src = encoder(batch_src)
            feat_tgt = encoder(batch_tgt)

            logits_src = classifier(feat_src)
            cls_loss = F.cross_entropy(logits_src, y_batch)

            domain_src = torch.zeros(feat_src.size(0), device=device)
            domain_tgt = torch.ones(feat_tgt.size(0), device=device)

            d_marg_src = d_marg(feat_src, lambd)
            d_marg_tgt = d_marg(feat_tgt, lambd)
            marg_loss = (F.binary_cross_entropy_with_logits(d_marg_src, domain_src)
                         + F.binary_cross_entropy_with_logits(d_marg_tgt, domain_tgt))

            probs_src = F.softmax(logits_src, dim=1)
            with torch.no_grad():
                probs_tgt = F.softmax(classifier(feat_tgt), dim=1)
            d_cond_src = d_cond(feat_src, probs_src, lambd)
            d_cond_tgt = d_cond(feat_tgt, probs_tgt, lambd)
            cond_loss = (F.binary_cross_entropy_with_logits(d_cond_src, domain_src)
                         + F.binary_cross_entropy_with_logits(d_cond_tgt, domain_tgt))

            total_loss = cls_loss + args.marg_weight * marg_loss + args.cond_weight * cond_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
            optimizer.step()

            epoch_losses["cls"] += cls_loss.item()
            epoch_losses["marg"] += marg_loss.item()
            epoch_losses["cond"] += cond_loss.item()
            global_step += 1

        n = steps_per_epoch
        print(f"Epoch {epoch+1}/{args.epochs} | cls_loss={epoch_losses['cls']/n:.4f} "
              f"marg_loss={epoch_losses['marg']/n:.4f} cond_loss={epoch_losses['cond']/n:.4f} "
              f"lambd={lambd:.3f}")

    # ------------------------------------------------------------------
    # 8. Evaluation: no-TTT baseline vs. with-TTT, on the held-out test labels
    # ------------------------------------------------------------------
    print("\nEvaluating (no-TTT baseline)...")
    encoder.eval()
    classifier.eval()
    with torch.no_grad():
        logits = classifier(encoder(tgt_tensors))
        probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()

    valid_eval = y_test != -1  # some test rows may be unlabeled too
    baseline_metrics = {
        "accuracy": accuracy_score(y_test[valid_eval], preds[valid_eval]),
        "f1": f1_score(y_test[valid_eval], preds[valid_eval]),
        "auc": roc_auc_score(y_test[valid_eval], probs[valid_eval]),
    }
    print(f"Baseline (no TTT): {baseline_metrics}")

    print("\nEvaluating (with TTT)...")
    recon_head = MaskedReconstructionHead(
        emb_dim=args.emb_dim, input_dim=sum(processed_dims.values())
    ).to(device)
    ttt = TTTWrapper(encoder, classifier, recon_head,
                      lr=args.ttt_lr, steps=args.ttt_steps, mask_ratio=args.ttt_mask_ratio)

    group_names = list(processed_dims.keys())
    X_tgt_flat = groups_dict_to_flat(tgt_tensors, group_names)

    all_preds, all_probs = [], []
    for i in range(0, X_tgt_flat.size(0), args.batch_size):
        batch = X_tgt_flat[i:i + args.batch_size]
        ttt.reset_to_base()  # stateless adaptation per batch (see design note)
        logits, _ = ttt.adapt_and_predict(batch)
        probs_b = F.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        preds_b = logits.argmax(dim=1).detach().cpu().numpy()
        all_probs.append(probs_b)
        all_preds.append(preds_b)

    ttt_preds = np.concatenate(all_preds)
    ttt_probs = np.concatenate(all_probs)
    ttt_metrics = {
        "accuracy": accuracy_score(y_test[valid_eval], ttt_preds[valid_eval]),
        "f1": f1_score(y_test[valid_eval], ttt_preds[valid_eval]),
        "auc": roc_auc_score(y_test[valid_eval], ttt_probs[valid_eval]),
    }
    print(f"With TTT: {ttt_metrics}")

    results = {"baseline_no_ttt": baseline_metrics, "with_ttt": ttt_metrics}
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    torch.save({
        "encoder": encoder.state_dict(),
        "classifier": classifier.state_dict(),
        "fit_stats": {k: {kk: vv.tolist() for kk, vv in v.items()} for k, v in fit_stats.items()},
    }, os.path.join(args.output_dir, "armada_checkpoint.pt"))

    print(f"\nSaved results to {args.output_dir}/results.json")
    print(f"Saved checkpoint to {args.output_dir}/armada_checkpoint.pt")
    return results


# ----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                         help="Directory containing X_train.dat, y_train.dat, X_test.dat, y_test.dat")
    parser.add_argument("--max_samples", type=int, default=None,
                         help="If set, only load this many rows per split -- use for a quick smoke test "
                              "(e.g. --max_samples 5000) before running on the full dataset.")
    parser.add_argument("--output_dir", type=str, default="./armada_output")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--emb_dim", type=int, default=256)
    parser.add_argument("--marg_weight", type=float, default=0.5)
    parser.add_argument("--cond_weight", type=float, default=0.5)
    parser.add_argument("--ttt_lr", type=float, default=1e-4)
    parser.add_argument("--ttt_steps", type=int, default=5)
    parser.add_argument("--ttt_mask_ratio", type=float, default=0.15)
    args = parser.parse_args()
    train(args)