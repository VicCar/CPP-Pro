"""run_plm4cpps_v6.py — pLM4CPPs' PUBLISHED ESM2-640 (150M) model on the v6 TEST set.

This is the Panel-A "old dataset" point: pLM4CPPs' own trained weights (best_model_640.h5),
run on our v6 holdout — NOT retrained. Faithful to pLM4CPPs/esm_640_Latest.ipynb:
  1. ESM-2 t30 150M (640-d) mean-pool (residues only — THEIR convention) of pLM4CPPs train pool + v6 test.
  2. train_test_split(test_size=0.2, random_state=123) on the pool -> MinMaxScaler fit on the train part.
  3. Transform v6 test, reshape (N, 640, 1), predict with best_model_640.h5 @ thr=0.5.
  4. Report MCC / AUROC / AUPRC / F1 and save to results/plm4cpps_640_on_v6_test.{json,npz}.

Run (CUDA needed for the ESM-2 embedding step):
    ~/miniconda3/bin/python benchmarks/run_plm4cpps_v6.py
"""
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from pathlib import Path
import json, time
import numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (matthews_corrcoef, roc_auc_score, average_precision_score,
                             f1_score, accuracy_score, confusion_matrix)
import tensorflow as tf

CUR = Path(__file__).resolve().parent.parent                         # CPPro_current/
LEGACY = CUR.parent / 'CPPro_legacy'
PLM = LEGACY / 'pLM4CPPs'
V6_TEST = CUR / 'dataset' / 'splits_v6' / 'test.csv'
V6_VAL = CUR / 'dataset' / 'splits_v6' / 'val.csv'
EMB_CACHE = LEGACY / 'classifier_experiments' / 'embeddings' / 'plm4cpps_train_640.npz'
RESULTS = CUR / 'results'; RESULTS.mkdir(parents=True, exist_ok=True)

tokfile = LEGACY / 'classifier_experiments' / 'HFToken.txt'
if tokfile.exists():
    from huggingface_hub import login
    login(token=tokfile.read_text().strip(), add_to_git_credential=False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- 1. pLM4CPPs original pool ---
plm_cpp = pd.read_excel(PLM / 'dataset' / 'pLM4CPPs_dataset_CPP.xlsx')
plm_neg = pd.read_excel(PLM / 'dataset' / 'pLM4CPPs_dataset_Non-CPP.xlsx')
sc = [c for c in plm_cpp.columns if 'seq' in c.lower()][0]
sn = [c for c in plm_neg.columns if 'seq' in c.lower()][0]
plm_train = pd.concat([
    pd.DataFrame({'sequence': plm_cpp[sc].astype(str).str.upper().str.strip(), 'label': 1}),
    pd.DataFrame({'sequence': plm_neg[sn].astype(str).str.upper().str.strip(), 'label': 0}),
], ignore_index=True)
print(f"pLM4CPPs pool: {len(plm_train)} ({(plm_train.label==1).sum()}+/{(plm_train.label==0).sum()}-)")

test = pd.read_csv(V6_TEST)
val = pd.read_csv(V6_VAL)
print(f"v6 test: {len(test)} ({(test.label==1).sum()}+/{(test.label==0).sum()}-)  val: {len(val)}")

# --- 2. ESM-2 150M (640-d) mean-pool, residues only (THEIR convention) ---
MODEL_ID = 'facebook/esm2_t30_150M_UR50D'
print(f"Loading {MODEL_ID}...", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModel.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to(device).eval()
HID = model.config.hidden_size
assert HID == 640

@torch.no_grad()
def embed(seqs, label='set'):
    out = np.zeros((len(seqs), HID), dtype=np.float32)
    t0 = time.time()
    for i, s in enumerate(seqs):
        enc = tok(s, return_tensors='pt', add_special_tokens=True).to(device)
        h = model(**enc).last_hidden_state[0]            # (L+2, HID)
        out[i] = h[1:-1].float().mean(0).cpu().numpy()   # drop CLS/EOS (their pooling)
        if (i + 1) % 500 == 0:
            print(f"  {label}: {i+1}/{len(seqs)} ({(i+1)/(time.time()-t0):.1f} seq/s)", flush=True)
    return out

if EMB_CACHE.exists():
    print(f"Loading cached pLM4CPPs train embeddings from {EMB_CACHE}")
    plm_emb = np.load(EMB_CACHE)['emb']
    assert plm_emb.shape[0] == len(plm_train), "cache size mismatch — delete cache + rerun"
else:
    print("Embedding pLM4CPPs train pool ...")
    plm_emb = embed(plm_train.sequence.tolist(), 'plm_train')
    EMB_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(EMB_CACHE, emb=plm_emb)

print("Embedding v6 test + val ...")
test_emb = embed(test.sequence.tolist(), 'v6_test')
val_emb = embed(val.sequence.tolist(), 'v6_val')

# --- 3. pLM4CPPs scaler (fit on their train split) + predict test & val ---
X_tr, _, _, _ = train_test_split(plm_emb, plm_train.label.values, test_size=0.2,
                                 random_state=123, stratify=plm_train.label.values)
scaler = MinMaxScaler().fit(X_tr)
mdl = tf.keras.models.load_model(str(PLM / 'models' / 'ESM2-640' / 'best_model_640.h5'), compile=False)
test_probs = mdl.predict(scaler.transform(test_emb).reshape(len(test), HID, 1), batch_size=64, verbose=0).flatten()
val_probs = mdl.predict(scaler.transform(val_emb).reshape(len(val), HID, 1), batch_size=64, verbose=0).flatten()

# --- 4. val-tuned threshold (max v6-val MCC), report test @ tau and @ 0.5 ---
yv = val.label.values
grid = np.unique(np.clip(val_probs, 1e-4, 1 - 1e-4))
tau = float(grid[int(np.argmax([matthews_corrcoef(yv, (val_probs >= t).astype(int)) for t in grid]))])

y = test.label.values
def block(thr):
    pred = (test_probs >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(threshold=float(thr), mcc=float(matthews_corrcoef(y, pred)),
                f1=float(f1_score(y, pred, zero_division=0)),
                auc=float(roc_auc_score(y, test_probs)), auprc=float(average_precision_score(y, test_probs)),
                fpr=float(fp / (fp + tn)) if (fp + tn) else 0.0,
                tpr=float(tp / (tp + fn)) if (tp + fn) else 0.0,
                acc=float(accuracy_score(y, pred)), tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn))

res = dict(model='plm4cpps_640_published', backbone='esm2_t30_150M_UR50D', eval_set='v6_test',
           n=int(len(y)), tau=tau, test_at_tau=block(tau), test_at_05=block(0.5))
print(f"\n=== pLM4CPPs (published 640) on v6 test  (tau={tau:.3f} tuned on v6 val) ===")
print(f"  @tau: MCC {res['test_at_tau']['mcc']:.4f}  F1 {res['test_at_tau']['f1']:.4f}  "
      f"FPR {res['test_at_tau']['fpr']:.3f}  TPR {res['test_at_tau']['tpr']:.3f}")
print(f"  @0.5: MCC {res['test_at_05']['mcc']:.4f}  |  AUC {res['test_at_tau']['auc']:.4f}  "
      f"AUPRC {res['test_at_tau']['auprc']:.4f}")

with open(RESULTS / 'plm4cpps_640_on_v6_test.json', 'w') as f:
    json.dump(res, f, indent=2)
np.savez(RESULTS / 'plm4cpps_640_on_v6_test.npz',
         y_true=y, y_prob=test_probs, len_bin=test.len_bin.values.astype(str), tau=tau)
print(f"\nSaved -> {RESULTS}/plm4cpps_640_on_v6_test.{{json,npz}}")
