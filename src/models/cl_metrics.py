"""CL metrics from accuracy matrix: AA, BWT, FWT, Forgetting."""
from __future__ import annotations
import json, numpy as np, pandas as pd, torch
from pathlib import Path
from sklearn.metrics import f1_score, roc_auc_score
from scipy.special import expit as sigmoid_stable
from src.models.char_cnn import domains_to_batch
from src.utils.dga_taxonomy import split_by_dga_type

@torch.no_grad()
def evaluate_on_test(model, test_df, device, batch_size=512, is_transformer=False, tokenizer=None):
    model.eval()
    domains = test_df["domain"].tolist()
    labels  = np.array(test_df["label"].tolist())
    all_logits = []
    for i in range(0, len(domains), batch_size):
        batch_d = domains[i:i+batch_size]
        if is_transformer and tokenizer:
            enc = tokenizer(batch_d, padding="max_length", truncation=True, max_length=64, return_tensors="pt")
            logits = model(enc["input_ids"].to(device), enc["attention_mask"].to(device))
        else:
            x = domains_to_batch(batch_d).to(device)
            logits = model(x)
        all_logits.append(logits.cpu().numpy())
    logits_np = np.concatenate(all_logits)
    probs = sigmoid_stable(logits_np)
    preds = (probs >= 0.5).astype(int)
    result = {"f1": f1_score(labels, preds, zero_division=0),
              "auc": roc_auc_score(labels, probs) if labels.sum()>0 and (1-labels).sum()>0 else 0.0}
    df_char, df_word = split_by_dga_type(test_df)
    for sname, sdf in [("f1_char", df_char), ("f1_word", df_word)]:
        if len(sdf) < 10 or sdf["label"].nunique() < 2:
            result[sname] = float("nan"); continue
        sub_idx = test_df.index.isin(sdf.index)
        sub_p = probs[sub_idx[:len(probs)]]
        sub_l = np.array(sdf["label"].tolist())
        n = min(len(sub_p), len(sub_l))
        result[sname] = f1_score(sub_l[:n], (sub_p[:n]>=0.5).astype(int), zero_division=0)
    return result

def build_accuracy_row(model, split_dir, window_ids, up_to_t, device, batch_size=512, is_transformer=False, tokenizer=None):
    row = {}
    for s in range(up_to_t + 1):
        test_path = Path(split_dir) / f"{window_ids[s]}_test.csv"
        if not test_path.exists(): continue
        row[window_ids[s]] = evaluate_on_test(model, pd.read_csv(test_path), device, batch_size, is_transformer, tokenizer)
    return row

class AccuracyMatrix:
    def __init__(self, window_ids):
        self.window_ids = window_ids; self.T = len(window_ids)
        self.f1_matrix = np.full((self.T, self.T), np.nan)
        self.auc_matrix = np.full((self.T, self.T), np.nan)
        self.f1_char_matrix = np.full((self.T, self.T), np.nan)
        self.f1_word_matrix = np.full((self.T, self.T), np.nan)

    def add_row(self, t, row):
        for s, wid in enumerate(self.window_ids):
            if wid in row:
                self.f1_matrix[t][s] = row[wid]["f1"]
                self.auc_matrix[t][s] = row[wid].get("auc", np.nan)
                self.f1_char_matrix[t][s] = row[wid].get("f1_char", np.nan)
                self.f1_word_matrix[t][s] = row[wid].get("f1_word", np.nan)

    def compute_metrics(self):
        T, a = self.T, self.f1_matrix
        diag = [a[i][i] for i in range(T) if not np.isnan(a[i][i])]
        aa = float(np.mean(diag)) if diag else 0.0
        bwt_v = [a[T-1][i]-a[i][i] for i in range(T-1) if not np.isnan(a[T-1][i]) and not np.isnan(a[i][i])]
        bwt = float(np.mean(bwt_v)) if bwt_v else 0.0
        fwt_v = [a[i-1][i]-0.5 for i in range(1,T) if not np.isnan(a[i-1][i])]
        fwt = float(np.mean(fwt_v)) if fwt_v else 0.0
        forg_v = []
        for i in range(T-1):
            col = [a[j][i] for j in range(T) if not np.isnan(a[j][i])]
            if col and not np.isnan(a[T-1][i]): forg_v.append(max(col)-a[T-1][i])
        forgetting = float(np.mean(forg_v)) if forg_v else 0.0
        degrad = float(a[0][0]-a[T-1][T-1]) if not np.isnan(a[0][0]) and not np.isnan(a[T-1][T-1]) else 0.0
        def _ml(mat):
            v=[x for x in mat[T-1] if not np.isnan(x)]; return float(np.mean(v)) if v else float("nan")
        return {"aa_f1":round(aa,4),"bwt":round(bwt,4),"fwt":round(fwt,4),"forgetting":round(forgetting,4),
                "degrad":round(degrad,4),"f1_old":round(float(a[T-1][0]),4) if not np.isnan(a[T-1][0]) else 0.0,
                "f1_first":round(float(a[0][0]),4) if not np.isnan(a[0][0]) else None,
                "f1_last":round(float(a[T-1][T-1]),4) if not np.isnan(a[T-1][T-1]) else None,
                "aa_f1_char":round(_ml(self.f1_char_matrix),4),"aa_f1_word":round(_ml(self.f1_word_matrix),4)}

    def compute_per_type_metrics(self):
        T = self.T
        def _m(mat):
            diag=[mat[i][i] for i in range(T) if not np.isnan(mat[i][i])]
            aa=float(np.mean(diag)) if diag else float("nan")
            bv=[mat[T-1][i]-mat[i][i] for i in range(T-1) if not np.isnan(mat[T-1][i]) and not np.isnan(mat[i][i])]
            bwt=float(np.mean(bv)) if bv else float("nan")
            return {"aa_f1":round(aa,4),"bwt":round(bwt,4)}
        return {"char_based":_m(self.f1_char_matrix),"word_based":_m(self.f1_word_matrix)}

    def to_dataframe(self):
        return pd.DataFrame(self.f1_matrix, index=[f"T{i+1}" for i in range(self.T)],
                            columns=[f"{w}_test" for w in self.window_ids])

    def save(self, out_dir, prefix=""):
        out_dir=Path(out_dir); out_dir.mkdir(parents=True,exist_ok=True)
        p=prefix+"_" if prefix else ""
        self.to_dataframe().to_csv(out_dir/f"{p}accuracy_matrix.csv")
        m=self.compute_metrics(); m["per_type"]=self.compute_per_type_metrics()
        with open(out_dir/f"{p}cl_metrics.json","w") as f: json.dump(m,f,indent=2)

def print_metrics_table(results, logger):
    header=f"{'Method':<28} {'AA-F1':>7} {'BWT':>8} {'Forg.':>8} {'Degrad.':>8} {'F1-Old':>7} {'F1-Char':>7} {'F1-Word':>7}"
    sep="-"*95; logger.info(sep); logger.info(header); logger.info(sep)
    for r in results:
        f1c=f"{r.get('aa_f1_char',0):>7.4f}" if r.get('aa_f1_char') and not np.isnan(r.get('aa_f1_char',float('nan'))) else "    N/A"
        f1w=f"{r.get('aa_f1_word',0):>7.4f}" if r.get('aa_f1_word') and not np.isnan(r.get('aa_f1_word',float('nan'))) else "    N/A"
        logger.info(f"{r['method']:<28} {r['aa_f1']:>7.4f} {r['bwt']:>+8.4f} {r.get('forgetting',0):>+8.4f} {r['degrad']:>+8.4f} {r.get('f1_old',0):>7.4f} {f1c} {f1w}")
    logger.info(sep)

def print_per_type_table(all_per_type, logger):
    logger.info("\n  Supplementary: Per-DGA-Type Metrics")
    sep="-"*65; logger.info(f"  {sep}")
    logger.info(f"  {'Method':<28} {'Char AA-F1':>10} {'Char BWT':>10} {'Word AA-F1':>10} {'Word BWT':>10}")
    logger.info(f"  {sep}")
    for method, pt in all_per_type.items():
        c,w=pt.get("char_based",{}),pt.get("word_based",{})
        def _f(v): return f"{v:>10.4f}" if v is not None and v==v else "       N/A"
        logger.info(f"  {method:<28} {_f(c.get('aa_f1'))} {_f(c.get('bwt'))} {_f(w.get('aa_f1'))} {_f(w.get('bwt'))}")
    logger.info(f"  {sep}")
