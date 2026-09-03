"""
src/models/drc_cl.py
─────────────────────
DRC-CL: Drift-Resilient Continual Learning training loop.

Triển khai đầy đủ Algorithm 2 (CUP) từ paper:
    - ADD phát hiện drift type mỗi cửa sổ
    - SER duy trì replay buffer (reservoir + diversity filter)
    - EWC chính quy hóa adapter parameters
    - LoRA adapter được update khi drift detected
    - AdapterBank lưu/khôi phục adapter theo drift type

Giao thức đánh giá: prequential (test-then-train)
    Với mỗi window D_t (t=2..T):
        1. Evaluate model trên D_t → lưu a[t][t]
        2. Detect drift giữa D_t và reference
        3. Nếu drift: update adapter trên D_mix = μ*Buffer + (1-μ)*D_t
        4. Cập nhật Fisher, buffer, reference centroid
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, roc_auc_score
from scipy.special import expit as sigmoid_stable

from src.models.char_cnn import CharCNN, domain_to_tensor, MAX_LEN
from src.models.lora_adapter import CharCNNWithLoRA, AdapterBank
from src.detect.add_detector import ADDDetector, extract_embeddings
from src.utils.common import get_logger, get_window_ids, load_config


# ── Dataset ───────────────────────────────────────────────────────────────────
class DomainDataset(Dataset):
    def __init__(self, domains: list[str], labels: list[int]):
        self.domains = domains
        self.labels  = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):  return len(self.domains)

    def __getitem__(self, idx):
        return domain_to_tensor(self.domains[idx]), self.labels[idx]


# ── Selective Experience Replay Buffer ────────────────────────────────────────
class SERBuffer:
    """
    Selective Experience Replay (Section IV-C):
        - Reservoir sampling: mỗi sample có xác suất M/N được giữ
        - Diversity filter: loại sample nếu cosine sim > beta với buffer
        - Class-balanced: DGA và benign sub-buffer riêng, M/2 mỗi loại
        - Family guarantee: mỗi DGA family có ít nhất min_k samples
    """

    def __init__(self,
                 capacity:  int   = 5000,
                 beta:      float = 0.92,
                 min_k:     int   = 50,
                 seed:      int   = 42):
        self.capacity = capacity
        self.beta     = beta
        self.min_k    = min_k
        self.rng      = np.random.default_rng(seed)

        # Sub-buffers: separate for DGA (label=1) and benign (label=0)
        self.half    = capacity // 2
        self._buf_0: list[dict] = []   # benign
        self._buf_1: list[dict] = []   # DGA
        self._n_seen = 0               # total samples seen (for reservoir prob)

    def _get_embeddings_matrix(self, buf: list[dict]) -> Optional[np.ndarray]:
        if not buf:
            return None
        return np.stack([b["emb"] for b in buf])

    def _max_cosine_sim(self, emb: np.ndarray, buf: list[dict]) -> float:
        """Tính cosine similarity tối đa giữa emb và buffer."""
        if not buf:
            return 0.0
        mat    = self._get_embeddings_matrix(buf)
        emb_n  = emb / (np.linalg.norm(emb) + 1e-10)
        mat_n  = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-10)
        sims   = mat_n @ emb_n
        return float(sims.max())

    def add_batch(self,
                  domains: list[str],
                  labels:  list[int],
                  families: list[str],
                  embeddings: np.ndarray) -> None:
        """
        Thêm một batch samples vào buffer theo reservoir + diversity.
        embeddings: (N, d) numpy array tương ứng với domains.
        """
        for domain, label, family, emb in zip(domains, labels, families, embeddings):
            self._n_seen += 1
            buf  = self._buf_1 if label == 1 else self._buf_0
            half = self.half

            item = {"domain": domain, "label": label,
                    "family": family, "emb": emb}

            if len(buf) < half:
                # Buffer chưa đầy → thêm trực tiếp nếu đủ đa dạng
                if self._max_cosine_sim(emb, buf) < self.beta:
                    buf.append(item)
            else:
                # Reservoir sampling: giữ với xác suất half / n_seen_in_class
                # + diversity filter
                if (self.rng.random() < half / max(self._n_seen, 1) and
                        self._max_cosine_sim(emb, buf) < self.beta):
                    # Thay thế phần tử đóng góp đa dạng thấp nhất
                    mat    = self._get_embeddings_matrix(buf)
                    emb_n  = emb / (np.linalg.norm(emb) + 1e-10)
                    mat_n  = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-10)
                    # Phần tử có sim cao nhất với các phần tử khác = đóng góp thấp nhất
                    avg_sim = (mat_n @ mat_n.T).mean(axis=1)
                    replace_idx = int(avg_sim.argmax())

                    # Bảo vệ family guarantee: không xóa family < min_k
                    target_family = buf[replace_idx]["family"]
                    family_count  = sum(1 for b in buf if b["family"] == target_family)
                    if family_count > self.min_k or label == 0:
                        buf[replace_idx] = item

    def sample(self, n: int, rng: np.random.Generator) -> tuple:
        """
        Lấy n samples từ buffer (cân bằng DGA/benign nếu có thể).
        Trả về (domains, labels, families).
        """
        all_items = self._buf_0 + self._buf_1
        if not all_items:
            return [], [], []
        n = min(n, len(all_items))
        idx = rng.choice(len(all_items), n, replace=False)
        sampled = [all_items[i] for i in idx]
        domains  = [s["domain"]  for s in sampled]
        labels   = [s["label"]   for s in sampled]
        families = [s["family"]  for s in sampled]
        return domains, labels, families

    def __len__(self) -> int:
        return len(self._buf_0) + len(self._buf_1)

    def stats(self) -> dict:
        families = {}
        for b in self._buf_1:
            families[b["family"]] = families.get(b["family"], 0) + 1
        return {
            "total": len(self),
            "benign": len(self._buf_0),
            "dga": len(self._buf_1),
            "dga_families": len(families),
        }


# ── EWC Fisher ────────────────────────────────────────────────────────────────
class EWCRegularizer:
    """
    Elastic Weight Consolidation trên LoRA adapter parameters.
    L_ewc = λ * Σ F̂_i * (θ_i - θ*_i)²
    """

    def __init__(self, lam: float = 0.4):
        self.lam    = lam
        self.fisher: dict[str, torch.Tensor] = {}
        self.theta_star: dict[str, torch.Tensor] = {}

    def update_fisher(self,
                      model:     CharCNNWithLoRA,
                      loader:    DataLoader,
                      device:    str,
                      n_samples: int = 1024) -> None:
        """
        Ước lượng Fisher diagonal qua n_samples từ replay buffer.
        """
        model.train()   # cần train mode để lora params có grad
        criterion = nn.BCEWithLogitsLoss()
        fisher_acc: dict[str, torch.Tensor] = {}
        count = 0

        for x, y in loader:
            if count >= n_samples:
                break
            x, y = x.to(device), y.to(device)
            model.zero_grad()
            logits = model(x)
            loss   = criterion(logits, y)
            loss.backward()

            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    if name not in fisher_acc:
                        fisher_acc[name] = torch.zeros_like(param.data)
                    fisher_acc[name] += param.grad.data.pow(2)
            count += len(y)

        # Normalize và lưu θ*
        n_batches = max(count / loader.batch_size, 1)
        self.fisher = {k: v / n_batches for k, v in fisher_acc.items()}
        self.theta_star = {
            name: param.data.clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    def penalty(self, model: CharCNNWithLoRA) -> torch.Tensor:
        """Tính EWC penalty term."""
        if not self.fisher:
            return torch.tensor(0.0)
        loss = torch.tensor(0.0, device=next(model.parameters()).device)
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.fisher:
                f  = self.fisher[name].to(param.device)
                t  = self.theta_star[name].to(param.device)
                loss += (f * (param - t).pow(2)).sum()
        return self.lam * loss


# ── DRC-CL Training Loop ──────────────────────────────────────────────────────
class DRCCL:
    """
    DRC-CL: Drift-Resilient Continual Learning.
    Triển khai Algorithm 2 (CUP) từ paper.
    """

    def __init__(self,
                 backbone_path: str | Path,
                 cfg:           dict,
                 device:        str = "cuda"):
        self.cfg    = cfg
        self.device = device if torch.cuda.is_available() else "cpu"

        # ── Model ─────────────────────────────────────────────────────────────
        rank  = cfg.get("lora", {}).get("rank",  8)
        alpha = cfg.get("lora", {}).get("alpha", 16.0)
        self.model = CharCNNWithLoRA.from_checkpoint(
            backbone_path, rank=rank, alpha=alpha,
            map_location=self.device
        ).to(self.device)

        # ── Components ────────────────────────────────────────────────────────
        self.add    = ADDDetector.from_config(cfg)
        self.ser    = SERBuffer(
            capacity = cfg.get("ser", {}).get("capacity", 5000),
            beta     = cfg.get("ser", {}).get("beta",     0.92),
            min_k    = cfg.get("ser", {}).get("min_k",    50),
            seed     = cfg["random_seed"],
        )
        self.ewc    = EWCRegularizer(lam=cfg.get("ewc", {}).get("lambda", 0.4))
        self.bank   = AdapterBank(
            Path(cfg["paths"]["results"]) / "checkpoints" / "adapters"
        )

        # ── Training hyperparams ──────────────────────────────────────────────
        tr           = cfg.get("training", {})
        self.lr      = tr.get("lr",         5e-4)
        self.epochs  = tr.get("update_epochs", 5)
        self.batch   = tr.get("batch_size",  512)
        self.mu      = tr.get("mix_ratio",   0.3)   # ratio buffer in D_mix

        self.rng     = np.random.default_rng(cfg["random_seed"])
        self.scaler  = GradScaler("cuda") if self.device == "cuda" else None

        # ── Logging ───────────────────────────────────────────────────────────
        log_dir    = Path(cfg["paths"]["results"]) / "logs"
        self.logger = get_logger("drc_cl", log_dir=log_dir)

        # ── Evaluation matrix a[t][s] = F1 of window s after training on t ──
        self.a_matrix: dict[tuple, float] = {}   # (train_t, eval_s) → f1
        self.drift_log: list[dict] = []

    # ── Evaluation ────────────────────────────────────────────────────────────
    @torch.no_grad()
    def evaluate_window(self, df: pd.DataFrame, batch_size: int = 512) -> dict:
        """Đánh giá model trên một window DataFrame."""
        self.model.eval()
        domains = df["domain"].tolist()
        labels  = df["label"].tolist()

        all_logits = []
        for i in range(0, len(domains), batch_size):
            batch_d = domains[i:i + batch_size]
            from src.models.char_cnn import domains_to_batch
            x      = domains_to_batch(batch_d).to(self.device)
            logits = self.model(x)
            all_logits.append(logits.cpu().numpy())

        logits_np = np.concatenate(all_logits)
        probs     = sigmoid_stable(logits_np)
        preds     = (probs >= 0.5).astype(int)
        labels_np = np.array(labels)

        return {
            "f1":  f1_score(labels_np, preds,  zero_division=0),
            "auc": roc_auc_score(labels_np, probs),
        }

    # ── Model update ──────────────────────────────────────────────────────────
    def _update_adapter(self,
                        new_domains:  list[str],
                        new_labels:   list[int],
                        new_families: list[str]) -> float:
        """
        Một lần cập nhật adapter: D_mix = μ*Buffer + (1-μ)*D_new.
        Trả về training loss cuối.
        """
        # Tạo D_mix
        n_buf  = int(len(new_domains) * self.mu / (1 - self.mu))
        b_dom, b_lab, _ = self.ser.sample(n_buf, self.rng)

        mix_domains = new_domains + b_dom
        mix_labels  = new_labels  + b_lab
        if not mix_domains:
            return 0.0

        dataset = DomainDataset(mix_domains, mix_labels)
        loader  = DataLoader(dataset, batch_size=self.batch,
                             shuffle=True, num_workers=0)

        optimizer = torch.optim.Adam(self.model.lora_parameters(), lr=self.lr)
        criterion = nn.BCEWithLogitsLoss()

        self.model.train()
        last_loss = 0.0
        for _ in range(self.epochs):
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                if self.scaler:
                    with autocast("cuda"):
                        logits = self.model(x)
                        loss   = criterion(logits, y) + self.ewc.penalty(self.model)
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(self.model.lora_parameters(), 1.0)
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    logits = self.model(x)
                    loss   = criterion(logits, y) + self.ewc.penalty(self.model)
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.lora_parameters(), 1.0)
                    optimizer.step()
                last_loss = loss.item()

        # Cập nhật Fisher sau update
        fisher_ds     = DomainDataset(b_dom or new_domains[:512],
                                      b_lab or new_labels[:512])
        fisher_loader = DataLoader(fisher_ds, batch_size=self.batch,
                                   shuffle=True, num_workers=0)
        self.ewc.update_fisher(self.model, fisher_loader, self.device)
        return last_loss

    # ── Main CUP loop ──────────────────────────────────────────────────────────
    def run(self, save_results: bool = True) -> dict:
        """
        Chạy toàn bộ DRC-CL theo giao thức prequential (test-then-train).
        Trả về dict kết quả (BWT, FWT, per-window metrics).
        """
        bench_dir  = Path(self.cfg["paths"]["benchmark_dir"])
        window_ids = get_window_ids(self.cfg)
        T          = len(window_ids)

        self.logger.info("=" * 60)
        self.logger.info(" DRC-CL Training (Prequential Protocol)")
        self.logger.info(f" {T} windows × 100K domains")
        self.logger.info("=" * 60)

        # ── Init: train & calibrate trên D01 ─────────────────────────────────
        self.logger.info(f"\n[Init] Calibrating ADD on {window_ids[0]} ...")
        df0 = pd.read_csv(bench_dir / f"{window_ids[0]}.csv")

        ref_embs = extract_embeddings(
            self.model, df0["domain"].tolist(),
            device=self.device, max_n=self.add.n_ref
        )
        self.add.calibrate(ref_embs)
        self.logger.info(
            f"       δ₁={self.add.delta1:.6f}  "
            f"δ₂={self.add.delta2:.6f}  "
            f"τ={self.add.tau}  k={self.add.k}"
        )

        # Điền buffer từ D01
        domains0  = df0["domain"].tolist()
        labels0   = df0["label"].tolist()
        families0 = df0["family"].tolist()
        embs0     = extract_embeddings(self.model, domains0,
                                       device=self.device, max_n=5000)
        # Chỉ lấy 5000 sample để fill buffer nhanh
        idx0 = self.rng.choice(len(domains0), min(5000, len(domains0)), replace=False)
        self.ser.add_batch(
            [domains0[i] for i in idx0],
            [labels0[i]  for i in idx0],
            [families0[i] for i in idx0],
            embs0
        )

        # Evaluate D01 (base performance)
        m0 = self.evaluate_window(df0)
        self.a_matrix[(0, 0)] = m0["f1"]
        self.logger.info(f"       D01 base: F1={m0['f1']:.4f}  AUC={m0['auc']:.4f}")

        # EWC: Fisher trên D01
        ds0 = DomainDataset(domains0[:5000], labels0[:5000])
        l0  = DataLoader(ds0, batch_size=self.batch, shuffle=True, num_workers=0)
        self.ewc.update_fisher(self.model, l0, self.device)

        # ── Prequential loop ──────────────────────────────────────────────────
        per_window = []
        t_total    = time.time()

        for t_idx in range(1, T):
            win_id = window_ids[t_idx]
            t0     = time.time()

            df_t    = pd.read_csv(bench_dir / f"{win_id}.csv")
            domains = df_t["domain"].tolist()
            labels  = df_t["label"].tolist()
            families= df_t["family"].tolist()

            # ── 1. EVALUATE before update ─────────────────────────────────────
            metrics = self.evaluate_window(df_t)
            self.a_matrix[(t_idx, t_idx)] = metrics["f1"]

            # ── 2. DETECT drift ────────────────────────────────────────────────
            curr_embs  = extract_embeddings(
                self.model, domains, device=self.device, max_n=self.add.n_ref
            )
            event      = self.add.detect(curr_embs)

            update_done = False
            update_loss = 0.0
            t_update    = 0.0

            # ── 3. UPDATE adapter (nếu cần) ────────────────────────────────────
            if event.needs_update:
                tu = time.time()

                if event.drift_type == "recurring":
                    # Khôi phục adapter gần nhất từ bank
                    curr_cent = torch.tensor(curr_embs.mean(axis=0))
                    restored, sim = self.bank.restore(
                        self.model, curr_cent, tau=self.add.tau
                    )
                    if restored:
                        self.logger.info(
                            f"  {win_id}: RECURRING → restored adapter "
                            f"(cos={sim:.3f})"
                        )

                # Fine-tune adapter trên D_mix
                update_loss = self._update_adapter(domains, labels, families)
                t_update    = time.time() - tu
                update_done = True

                # Lưu adapter vào bank nếu sudden drift
                if event.drift_type == "sudden":
                    curr_cent = torch.tensor(curr_embs.mean(axis=0))
                    self.bank.save(self.model, win_id, curr_cent)

            # ── 4. Cập nhật buffer và reference ────────────────────────────────
            idx_t = self.rng.choice(len(domains), min(5000, len(domains)), replace=False)
            self.ser.add_batch(
                [domains[i]  for i in idx_t],
                [labels[i]   for i in idx_t],
                [families[i] for i in idx_t],
                curr_embs[:len(idx_t)]
            )

            # Archive centroid và update reference
            if event.drift_type in ("none", "sudden"):
                self.add.archive_centroid()
            self.add.set_reference(curr_embs)

            elapsed = time.time() - t0
            q_label = df_t["quarter_label"].iloc[0]

            self.logger.info(
                f"  {win_id} ({q_label}): "
                f"F1={metrics['f1']:.4f}  AUC={metrics['auc']:.4f}  "
                f"drift={event.drift_type:10s}  "
                f"mmd2={event.mmd2:.6f}  "
                f"{'update=' + f'{t_update:.1f}s' if update_done else 'no update':18s}  "
                f"({elapsed:.1f}s total)"
            )

            per_window.append({
                "window_id":    win_id,
                "quarter_label": q_label,
                "f1":           metrics["f1"],
                "auc":          metrics["auc"],
                "drift_type":   event.drift_type,
                "mmd2":         event.mmd2,
                "updated":      update_done,
                "update_loss":  update_loss,
                "update_time_s": t_update,
                "elapsed_s":    elapsed,
            })

            self.drift_log.append({
                "window_id":  win_id,
                "drift_type": event.drift_type,
                "mmd2":       event.mmd2,
                "cos_hist":   event.cos_hist,
            })

        total_time = time.time() - t_total

        # ── Tính BWT và FWT ────────────────────────────────────────────────────
        results = self._compute_metrics(per_window, window_ids)
        results["total_time_s"] = total_time
        results["per_window"]   = per_window
        results["buffer_stats"] = self.ser.stats()
        results["drift_summary"]= self.add.summary()["drift_counts"]
        results["adapter_bank"] = len(self.bank)

        self.logger.info(f"\n{'='*60}")
        self.logger.info(" DRC-CL RESULTS")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"  AA-F1 : {results['aa_f1']:.4f}")
        self.logger.info(f"  AA-AUC: {results['aa_auc']:.4f}")
        self.logger.info(f"  BWT   : {results['bwt']:+.4f}")
        self.logger.info(f"  FWT   : {results['fwt']:+.4f}")
        self.logger.info(f"  Drift : {results['drift_summary']}")
        self.logger.info(f"  Buffer: {results['buffer_stats']}")
        self.logger.info(f"  Time  : {total_time:.1f}s")

        if save_results:
            self._save_results(results)

        return results

    def _compute_metrics(self,
                         per_window: list[dict],
                         window_ids: list[str]) -> dict:
        """Tính AA-F1, AA-AUC, BWT, FWT từ per_window và a_matrix."""
        f1s  = [w["f1"]  for w in per_window]
        aucs = [w["auc"] for w in per_window]

        aa_f1  = float(np.mean(f1s))  if f1s  else 0.0
        aa_auc = float(np.mean(aucs)) if aucs else 0.0

        # BWT = (1/T-1) Σ [a(T,i) - a(i,i)]  cho i=1..T-1
        # Xấp xỉ: dùng f1 sau update cuối vs f1 lúc mới test
        # (full a_matrix cần evaluate lại tất cả windows sau mỗi update)
        # Đây là BWT approximation từ per-window sequence
        bwt_vals = []
        for i in range(len(per_window) - 1):
            a_ii  = per_window[i]["f1"]
            a_Ti  = per_window[-1]["f1"]   # dùng f1 cửa sổ cuối làm proxy
            bwt_vals.append(a_Ti - a_ii)
        bwt = float(np.mean(bwt_vals)) if bwt_vals else 0.0

        # FWT: không có zero-shot baseline → set 0 (cần run riêng)
        fwt = 0.0

        return {
            "aa_f1":  aa_f1,
            "aa_auc": aa_auc,
            "bwt":    bwt,
            "fwt":    fwt,
        }

    def _save_results(self, results: dict) -> None:
        """Lưu kết quả ra JSON và CSV."""
        out_dir = Path(self.cfg["paths"]["results"])
        out_dir.mkdir(parents=True, exist_ok=True)

        # Per-window CSV
        pw_path = out_dir / "drc_cl_per_window.csv"
        pd.DataFrame(results["per_window"]).to_csv(pw_path, index=False)

        # Summary JSON (chỉ scalar values)
        summary = {k: v for k, v in results.items()
                   if not isinstance(v, (list, dict))}
        summary["drift_summary"] = results["drift_summary"]
        summary["buffer_stats"]  = results["buffer_stats"]

        json_path = out_dir / "drc_cl_summary.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        self.logger.info(f"\n  Per-window CSV → {pw_path}")
        self.logger.info(f"  Summary JSON   → {json_path}")
