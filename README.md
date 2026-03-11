# snack

This repository stores selected simulation assets and scripts using a whitelist `.gitignore` policy.

## Tracked by default whitelist

- `README.md`
- `res/*.mp4`
- `res/*.log`
- `meshes/*.STL`
- `script/*.py`
- `.gitignore`
- `.gitattributes`

## Linear backend (线性后端) — 实现细节与数学原理

概述：线性后端实现的是一种逐段投影（sequential projection / coordinate-descent）近似求解器，用于在保持 x-轴 twist 为只读的情况下快速生成 yz 方向的关节角近似解，主要用于作为更精确拟合器（雅可比/LM）或退火求解器的 warm-start。

主要数据布局与依赖：
- 输入向量 `v` 布局为 `[yz_vars..., head_translation(3), head_rpy(3)]`。
- 需要外部预计算字典 `precomp` 包含：`lengths_m`, `tgt`（目标段中点坐标数组）, `joint_axes`, `joint_signs`, `x_joint_indices_0b`, `yz_joint_indices_0b`, `twist_filtered`（x-twist 只读）以及 `fk_callbacks`（见下）。

FK 回调（必须由调用方提供）
- `fk_init(head_translation, head_rpy, joint_angles_base_x, yz_init, lengths_m, joint_axes)`：初始化内部 FK 缓存。
- `fk_apply_delta(i, delta_rad)`：对第 i 个关节应用增量（局部更新，不立即重新计算全链）。
- `fk_get_midpoint(i)`：返回第 i 段的中点坐标（世界系）。
- `fk_invalidate_from(i)` / `fk_ensure_upto(i)`：用于局部缓存失效与增量计算边界控制。

对齐（alignment）步骤——几何预处理：
1. 若目标点序列 `tgt` 长度至少 2，则取曲线起点 `f0` 与起始切线 `tangent = tgt[1] - tgt[0]`。
2. 计算头部坐标系的 x 轴（`head_axis`），先绕曲线起点旋转，使 `tangent` 与 `head_axis` 对齐（旋转轴为两者叉乘，角度为 arccos 内积）。
3. 将旋转后的曲线整体平移，使旋转后的起点匹配 `head_translation`。
4. 若目标至少含 3 点，则再绕 `head_axis` 在 `head_translation` 处做一次搜索旋转，使曲线在头部处的曲面法线与第一个关节轴方向尽量正交（离散搜索若干 phi，选取使代价最小的 phi 并旋转）。

逐段投影算法（核心）——数学与实现要点：
- 对每一次迭代（`n_pass = cfg.max_iter`）按 `yz_joint_indices_0b` 的顺序遍历每个 yz 关节 j：
	- 目标段索引 `seg_i = clamp(j, 0, #segments-1)`。
	- 使用 FK 回调保证能获取到 `seg_i` 的段中点：`fk_ensure_upto(seg_i+1)`。
	- 当前段代价定义为该段（及下一段）的中点与对齐后目标点的平方距离和：
		\[ C_i = ||m_i - T_i||^2 + ||m_{i+1} - T_{i+1}||^2, \]
		其中 $m_k$ 是通过 `fk_get_midpoint(k)` 得到的当前中点，$T_k$ 是 `aligned_tgt[k]`。
	- 枚举候选增量 \(\pm \delta\)（`step = 2*finite_diff_eps` 的小步长），对每个候选：
		* 将候选角度按关节符号 `joint_signs[j]` 转换为关节增量（并限制累计角度到 \([-\pi/2,\pi/2]\)）。
		* 通过 `fk_apply_delta` 临时施加增量、重新计算段中点并求代价，随后撤销增量以恢复状态。若代价下降则记为当前最佳增量。
	- 将最佳增量应用到 FK，并更新累计已应用角度数组 `applied_deltas` 与局部 `yz_vars`（注意符号反向关系：`yz_vars[k] += actual_apply / sign`）。

约束与数值策略：
- 每个关节的累计应用角度被夹紧到 \(\pm 90^\circ\)（即 \(\pm \pi/2\)），以避免过度扭转导致几何异常。
- 使用小步长的正负候选搜索（类似有限差分的一步坐标下降），代价函数基于段中点与目标点的平方误差。
- 算法为启发式的、局部的、非凸的近似方法：它不是严格求解全局最小化问题，而是用廉价的逐段投影来快速减少局部误差，适合作为更精确求解器的初始化（warm-start）。

复杂度与适用场景：
- 每次完整遍历的复杂度约为 O(n_yz * cost_fk) ，其中 `cost_fk` 包括 FK 增量更新与两次中点查询（常数级），整体随 `cfg.max_iter` 线性放大。
- 优点：实现简单、依赖少、运行快速，适用于实时或批量生成初始猜测。
- 缺点：可能收敛到局部最优、对 step/夹紧阈值敏感、对高频曲线细节不敏感（需要后续雅可比/LM 优化细化）。

注意事项与可改进点：
- 当前实现使用固定的小步长与离散 phi 搜索（73 点）来对齐平面法线，可将其换成解析最小化或更细致的一维优化以提高鲁棒性与性能。
- 累积夹紧策略（\(\pm \pi/2\)）是经验值；在不同机械臂/结构上可能需要调整。
- 若需要更平滑的轨迹，可把目标采样密度或代价函数扩展为覆盖更多相邻段。

参见代码：`script/linear_backend.py`（实现）以及 `script/shapefit_solver.py`（调用、precomp 构建与接口契约）。
