import pickle, numpy as np, sys

BASE = "/mnt/data/siping/runs/nogmc_ref_weekend/uav_cache_base.pkl"
ARM  = "/mnt/data/siping/runs/npu20ms_seed2/trials/trial_0032/gate/epoch20/uav_cache.pkl"

def load(path):
    rows = pickle.load(open(path, "rb"))
    out = {}
    for r in rows:
        out[r["im_name"]] = r
    return out, rows[0].keys()

def match(rec, th, tol, filt):
    sc = np.asarray(rec.get("pred_scores", []), dtype=np.float64).reshape(-1)
    pt = np.asarray(rec.get("pred_points", []), dtype=np.float64).reshape(-1, 2)
    gt = np.asarray(rec.get("gt_pts", []), dtype=np.float64).reshape(-1, 2)
    if len(sc) == 0:
        return 0, 0, len(gt)
    keep = np.where(sc >= th)[0]
    if len(keep) == 0:
        return 0, 0, len(gt)
    keep = keep[np.argsort(-sc[keep], kind="stable")]
    if filt == "nms8":
        chosen = []
        for i in keep:
            if not chosen:
                chosen.append(i); continue
            d = pt[chosen] - pt[i]
            if np.min((d * d).sum(1)) > tol * tol:
                chosen.append(i)
        keep = np.array(chosen)
    elif filt == "nms16":
        chosen = []
        for i in keep:
            if not chosen:
                chosen.append(i); continue
            d = pt[chosen] - pt[i]
            if np.min((d * d).sum(1)) > 16 * 16:
                chosen.append(i)
        keep = np.array(chosen)
    elif filt == "top1":
        keep = keep[:1]
    gts = gt
    used = np.zeros(len(gts), dtype=bool)
    tp = fp = 0
    tol2 = tol * tol
    for i in keep:
        p = pt[i]
        if len(gts):
            d = (gts - p)
            d2 = (d * d).sum(1)
            j = int(np.argmin(d2))
            if d2[j] <= tol2 and not used[j]:
                used[j] = True; tp += 1; continue
        fp += 1
    return tp, fp, int((~used).sum())

def run(cache, th, filt, tol=8.0):
    tp = fp = fn = 0
    for rec in cache.values():
        a, b, c = match(rec, th, tol, filt)
        tp += a; fp += b; fn += c
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return tp, fp, fn, p, r, f1

A, keysA = load(BASE)
B, keysB = load(ARM)
print("cache record keys:", sorted(keysB))
print(f"A frames={len(A)}  B frames={len(B)}")

ths = [0.05, 0.15, 0.25, 0.35, 0.5, 0.75]
for filt in ["none", "top1", "nms8", "nms16"]:
    print(f"\n=== filter={filt} ===")
    for th in ths:
        ta, fa, fna, pa, ra, f1a = run(A, th, filt)
        tb, fb, fnb, pb, rb, f1b = run(B, th, filt)
        print(f" th={th:<5} | BASE tp={ta:<6} fp={fa:<7} F1={f1a:.6f} | ARM(trial_0032) tp={tb:<6} fp={fb:<7} F1={f1b:.6f} dF1={f1b-f1a:+.6f}")
