(uav) huangzhe@test-Super-Server:/tmp/pycharm_project_10ae9e2e$ python manu/generate_osd_videos.py \
>     --cache-file runs/badcase_analysis/inference_cache.pkl \
>     --output-dir runs/badcase_analysis/osd_videos \
>     --dist-thresh 8.0 \
>     --conf 0.20 \
>     --fps 25

>>> Loading cached inferences from: /tmp/pycharm_project_10ae9e2e/runs/badcase_analysis/inference_cache.pkl
Loaded 31613 image predictions from cache.

Total sequences to render: 24
Indexing dataset images on disk...
Indexed 31613 images.
[OK] 01_4485_1167-2666_osd.mp4                     | GT:981  TP:947  FP:217  FN:34   | Rec: 96.5% Prec: 81.4%                                                                                                             
[OK] 02_6321_0274-2773_osd.mp4                     | GT:1498 TP:1079 FP:115  FN:419  | Rec: 72.0% Prec: 90.4%                                                                                                             
[OK] 25_1_osd.mp4                                  | GT:660  TP:652  FP:11   FN:8    | Rec: 98.8% Prec: 98.3%                                                                                                             
[OK] 34_1_osd.mp4                                  | GT:503  TP:503  FP:28   FN:0    | Rec:100.0% Prec: 94.7%                                                                                                             
[OK] 3700000000002_102004_3_osd.mp4                | GT:1498 TP:1498 FP:0    FN:0    | Rec:100.0% Prec:100.0%                                                                                                             
[OK] 3700000000002_153918_1_osd.mp4                | GT:1193 TP:1036 FP:154  FN:157  | Rec: 86.8% Prec: 87.1%                                                                                                             
[OK] 43_1_osd.mp4                                  | GT:623  TP:623  FP:2    FN:0    | Rec:100.0% Prec: 99.7%                                                                                                             
[OK] 5_1_osd.mp4                                   | GT:1393 TP:1382 FP:56   FN:11   | Rec: 99.2% Prec: 96.1%                                                                                                             
[OK] DJI_0051_2_osd.mp4                            | GT:578  TP:111  FP:270  FN:467  | Rec: 19.2% Prec: 29.1%                                                                                                             
[OK] DJI_0175_2_osd.mp4                            | GT:448  TP:137  FP:34   FN:311  | Rec: 30.6% Prec: 80.1%                                                                                                             
[OK] wg2022_ir_010_split_01_osd.mp4                | GT:1427 TP:1261 FP:80   FN:166  | Rec: 88.4% Prec: 94.0%                                                                                                             
[OK] wg2022_ir_011_split_02_osd.mp4                | GT:1169 TP:960  FP:198  FN:209  | Rec: 82.1% Prec: 82.9%                                                                                                             
[OK] wg2022_ir_011_split_03_osd.mp4                | GT:184  TP:34   FP:113  FN:150  | Rec: 18.5% Prec: 23.1%                                                                                                             
[OK] wg2022_ir_012_split_08_osd.mp4                | GT:1356 TP:1120 FP:64   FN:236  | Rec: 82.6% Prec: 94.6%                                                                                                             
[OK] wg2022_ir_015_split_01_osd.mp4                | GT:1312 TP:1151 FP:52   FN:161  | Rec: 87.7% Prec: 95.7%                                                                                                             
[OK] wg2022_ir_020_split_01_osd.mp4                | GT:1474 TP:1335 FP:77   FN:139  | Rec: 90.6% Prec: 94.5%                                                                                                             
[OK] wg2022_ir_020_split_03_osd.mp4                | GT:500  TP:0    FP:172  FN:500  | Rec:  0.0% Prec:  0.0%                                                                                                             
[OK] wg2022_ir_020_split_05_osd.mp4                | GT:0    TP:0    FP:27   FN:0    | Rec:  0.0% Prec:  0.0%                                                                                                             
[OK] wg2022_ir_020_split_07_osd.mp4                | GT:1349 TP:1072 FP:268  FN:277  | Rec: 79.5% Prec: 80.0%                                                                                                             
[OK] wg2022_ir_040_split_04_osd.mp4                | GT:1498 TP:1473 FP:6    FN:25   | Rec: 98.3% Prec: 99.6%                                                                                                             
[OK] wg2022_ir_041_split_06_osd.mp4                | GT:1445 TP:1443 FP:19   FN:2    | Rec: 99.9% Prec: 98.7%                                                                                                             
[OK] wg2022_ir_047_split_01_osd.mp4                | GT:1340 TP:1298 FP:94   FN:42   | Rec: 96.9% Prec: 93.2%                                                                                                             
[OK] wg2022_ir_047_split_02_osd.mp4                | GT:1498 TP:1498 FP:57   FN:0    | Rec:100.0% Prec: 96.3%                                                                                                             
[OK] wg2022_ir_052_split_08_osd.mp4                | GT:1184 TP:1128 FP:210  FN:56   | Rec: 95.3% Prec: 84.3%                                                                                                             

===========================================================================
ALL VIDEOS GENERATED IN: /tmp/pycharm_project_10ae9e2e/runs/badcase_analysis/osd_videos
===========================================================================