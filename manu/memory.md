# 红外弱小无人机检测器算法攻坚与演进记忆 (Memory & Strategy)

## 一、当前检测器技术基线与大盘现状

1. **核心模型定位**：
   - **大目标检测**：YOLO26（用于中近距大目标集成与提供 Bounding Box）。
   - **弱小/点目标主干**：`YOLO26HeatmapDetector`（基于 YOLO26 骨干 + P2 颈部 + 高分辨率 Stride=2 热图点回归与亚像素偏移预测）。
2. **当前大盘实测指标**：
   - 全体验证集（24 个典型序列，31,613 帧）：
     - **Recall**：93.15%
     - **Precision**：92.61%
     - **FAR（单帧虚警率）**：0.062 个/帧（平均约 16 帧仅 1 个孤立虚警）
   - **分层表现**：
     - 19 个常规及中弱目标序列：表现优异，平均 Recall > 95%，多个序列达成 100% 满检。
     - 4~5 个长尾难例序列（Hard Cases）：指标断崖式偏低，成为系统主要短板。

---

## 二、四大 Hard Case 深度诊断与物理成因

通过抽帧及 `fn_missed_analysis.csv` 的精确数据统计，难例主要分为两类截然不同的物理成因：

| 序列名 | GT 帧数 | 表现 (conf=0.20) | 核心失败机理 |
| :--- | :--- | :--- | :--- |
| **`DJI_0175_2`** | 448 | Rec: 30.6%, Prec: 80.1% | **弱特征已被激活但受门限截断**：漏检中 49% 在目标 1.9px 内存在热图响应（平均 score=0.08），但被 0.20 门限截断。 |
| **`wg2022_ir_011_split_03`** | 184 | Rec: 18.5%, Prec: 23.1% | **高精度微弱响应**：漏检中 57% 在目标 1.2px 处有热图响应（平均 score=0.07），定位极准但峰值未达标。 |
| **`DJI_0051_2`** | 578 | Rec: 19.2%, Prec: 29.1% | **相机抖动 + 地面强杂波**：镜头平移导致地表树林/建筑边缘差分爆出 270+ 个假警，掩盖空中弱目标。 |
| **`wg2022_ir_020_split_03`** | 500 | Rec: 0.0%, Prec: 0.0% | **超低信噪比与像元自抵消**：目标仅 2px 且极暗（SCR < 1.0），短步长差分完全自抵消，单帧静态几近不可分。 |

---

## 三、核心物理洞察与技术演进结论

1. **暗弱运动目标的“像元自抵消定律”**：
   - 在 25fps 下，远距无人机帧间位移仅 0.2~0.4px，目标有 80% 以上面积在前后帧重叠。
   - 简单的相邻单帧差分 $|I_t - I_{t-1}|$ 会产生严重的自相抵消，导致微弱信号归零。
   - **破局准则**：必须拉大时序步长（$\Delta t \ge 4 \sim 12$ 帧），使目标在空间上完全错开，彻底释放 100% 运动能量。
2. **特征构建方式的迭代**：
   - 舍弃死板的手工绝对差分与局部方差归一化（易放大天空白噪声）；
   - 进化为 **3 帧多尺度原始灰度输入**：$[I_t, I_{t-4}, I_{t-12}]$。
   - 在图像通道上直接呈现清晰的“彩色运动拖尾”（当前帧蓝色、历史帧红绿），静止背景保持灰度。
3. **前 12 帧工程回退机制（Clamp Fallback）**：
   - 保持与原验证集划分（Fold 4）100% 一致的 31,613 张图片总数；
   - 对前 12 帧回退至第 0 帧处理，数学上等价于目标初始静止/悬停，既保障了评测公正性，又促使模型学习悬停目标识别。
4. **网络结构的继承性**：
   - 保持原有 YOLO26 `b0 = Conv(3, 16, 3, 2)` 网络架构完全不变；
   - 现有最佳权重 `best.pt` 可以 **100% 严格匹配（strict=True）** 无损加载；
   - 卷积第一层自身即可通过反向传播自发学习时序加权组合。

---

## 四、当前就绪代码与夜间微调挂机方案

1. **数据集生成脚本**：
   - 脚本：`manu/build_yolo_3frame_raw_dataset.py`
   - 功能：严格镜像当前 `/mnt/data/siping/datasets/manu/uav`，从原始视频抽取 $[I_t, I_{t-4}, I_{t-12}]$ 合成 3 通道图片，保留全部标签。
2. **多卡并行超参调优脚本**：
   - 脚本：`manu/optuna_parallel_3frame_heatmap.py`
   - 特性：
     - 支持 4 卡独立并行，支持 SQLite 断点续跑与自动生成汇总 CSV；
     - 搜索空间严格针对 10 Epoch 微调定制（`lr0`: 5e-5~6e-4，`focal_beta`: 2.5~4.2，轻度数据增广）。
3. **执行命令**：
   ```bash
   # 1. 生成 3 帧时序数据集
   python manu/build_yolo_3frame_raw_dataset.py \
       --ref-dataset /mnt/data/siping/datasets/manu/uav \
       --raw-root /mnt/data/siping/datasets/manu/anti-uav \
       --output /mnt/data/siping/datasets/manu/uav_temporal_3frame

   # 2. 启动 4 卡并行 Optuna 挂机微调 (10 Epochs, 60 组试验)
   python manu/optuna_parallel_3frame_heatmap.py \
       --data /mnt/data/siping/datasets/manu/uav_temporal_3frame/data.yaml \
       --weights runs/optuna_heatmap_stride2_640/trial_0031/weights/best.pt \
       --gpus 0,1,2,3 \
       --n-trials 60 \
       --epochs 10 \
       --batch 32 \
       --stride 2 \
       --output-root runs/optuna_3frame_temporal_10ep
   ```

---

## 五、后续更进一步提升的储备方案（Next Milestones）

若当前多时相 3 帧微调仍未彻底解决极限暗弱目标，可启动代际跃升的双杀组合：

1. **前端：局部特征点积相关层（Local Correlation Volume）**：
   - 在 Stride=2（320×320）尺度上，将 $F_t$ 与 $F_{t-4}$ 进行 $5\times 5$（半径 $R=2$）局部点积匹配；
   - 生成 25 个速度假设通道，通过时空相干性将暗弱目标的信噪比提升 5~10 倍，同时将随机白噪声抵消为 0。
2. **后端：运动矢量联合显式监督（Motion Vector Supervision）**：
   - 利用当前与 4 帧前的真值位移 $\vec{v}^* = (x_t - x_{t-4}, y_t - y_{t-4})$ 作为硬监督标签；
   - Head 侧同时输出目标位置与瞬时速度 $(\Delta x, \Delta y)$；
   - 形成“前端相干共振搜索 + 后端速度标签约束”的完整物理闭环。
