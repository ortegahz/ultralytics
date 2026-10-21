import json

SEQ = "VIDEO00009_19700101_001531"
rows = []
for ep in (0, 1, 2, 3, 4, 5):
    d = json.load(open(f"runs/mixed_ft_route_b/gate/epoch{ep}/track2_golden.json"))
    s = d["per_sequence"][SEQ]
    rows.append((ep, s))
    print(
        "Ep%d %s: gt=%s det=%s tp=%s fp=%s fn=%s fn_absent=%s f1=%.12f rec=%.12f prec=%.12f"
        % (ep, SEQ, s["gt"], s["det"], s["tp"], s["fp"], s["fn"], s["fn_absent"],
           s["f1"], s["recall"], s["precision"])
    )

base = rows[0][1]
print("\n逐字段与 Epoch0 完全相同的 epoch：",
      [ep for ep, s in rows if s == base])
print("该序列是否真的被评测（gt>0 且 det>0）：", base["gt"] > 0, base["det"] > 0)
print("基线 F1 =", base["f1"], " 绝对量级参考：非零、非平凡")
