"""
GeoSched NSGA-II + 差分进化 混合算法 (NSGA-DE)
================================================
编码策略：随机键 (Random Keys)
  · 个体 = N 维实数向量，每个基因值 ∈ [0, 1]
  · 解码 = argsort(降序) → 任务考虑顺序 → 贪心解码器
  · 所有解均可行（贪心解码器保证约束满足）

DE 变异：DE/rand/1/bin
  · 变异：v = x_r1 + F * (x_r2 - x_r3)，clip 到 [0,1]
  · 交叉：binomial，概率 CR

NSGA-II 选择（2 目标，等价于 NSGA-III）：
  · 目标 1（主）：最大化 total_profit
  · 目标 2（辅）：最大化完成率 completion_rate
  · 非支配排序 + 拥挤距离选择下一代

关键设计：
  · 第 0 号个体直接编码为贪心排序 → 算法起点 ≥ 贪心基线
  · 纯连续编码，DE 操作无需修改
  · 贪心解码器确保所有约束始终满足
"""

import os
import sys
import time
import numpy as np

# 确保能 import greedy.py（无论从哪里调用）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from greedy import load_tasks, load_satellites, greedy_schedule

# ── 默认参数 ─────────────────────────────────────────────────────────
DEFAULT_POP_SIZE = 50    # 种群大小
DEFAULT_N_GEN    = 120   # 最大迭代代数
DEFAULT_F        = 0.6   # DE 缩放因子 (0.4 ~ 0.9 均可)
DEFAULT_CR       = 0.8   # DE 交叉概率 (0.7 ~ 0.95 均可)
DEFAULT_SEED     = 42


# ── 个体评估 ──────────────────────────────────────────────────────────

def _evaluate(keys: np.ndarray, tasks: list, satellites: list) -> tuple:
    """
    将随机键向量解码为调度方案并计算两个目标值。

    Parameters
    ----------
    keys : shape (N,)，每个基因 ∈ [0,1]

    Returns
    -------
    (total_profit, completion_rate, assignments)
    """
    # 降序 argsort：值越大，任务越早被考虑分配
    order      = np.argsort(-keys).tolist()
    assignments, profit = greedy_schedule(tasks, satellites, order)
    completion = len(assignments) / max(len(tasks), 1)
    return profit, completion, assignments


# ── NSGA-II 选择算子 ──────────────────────────────────────────────────

def _dominates(a: tuple, b: tuple) -> bool:
    """a 支配 b（所有目标 >=，至少一个 >，均为最大化）"""
    return all(ai >= bi for ai, bi in zip(a, b)) and any(ai > bi for ai, bi in zip(a, b))


def _fast_non_dominated_sort(objs: list) -> list:
    """
    快速非支配排序，返回 Pareto 分层列表（每层为下标列表）。
    时间复杂度 O(N²)，N=2P=100 时可忽略不计。
    """
    N         = len(objs)
    dom_count = [0] * N          # 被支配次数
    dom_set   = [[] for _ in range(N)]   # 我支配的个体
    fronts    = [[]]

    for i in range(N):
        for j in range(i + 1, N):
            if _dominates(objs[i], objs[j]):
                dom_set[i].append(j)
                dom_count[j] += 1
            elif _dominates(objs[j], objs[i]):
                dom_set[j].append(i)
                dom_count[i] += 1
        if dom_count[i] == 0:
            fronts[0].append(i)

    k = 0
    while fronts[k]:
        nxt = []
        for i in fronts[k]:
            for j in dom_set[i]:
                dom_count[j] -= 1
                if dom_count[j] == 0:
                    nxt.append(j)
        k += 1
        fronts.append(nxt)

    return fronts[:-1]   # 去掉末尾空列表


def _crowding_distance(front: list, objs: list) -> list:
    """计算拥挤距离，用于同层排序。边界点距离设为 inf。"""
    n_f   = len(front)
    dists = [0.0] * n_f
    if n_f <= 2:
        return [float("inf")] * n_f

    n_obj = len(objs[0])
    for obj_idx in range(n_obj):
        sorted_idx = sorted(range(n_f), key=lambda i: objs[front[i]][obj_idx])
        dists[sorted_idx[0]]  = float("inf")
        dists[sorted_idx[-1]] = float("inf")
        obj_range = objs[front[sorted_idx[-1]]][obj_idx] - objs[front[sorted_idx[0]]][obj_idx]
        if obj_range == 0:
            continue
        for k in range(1, n_f - 1):
            prev_val = objs[front[sorted_idx[k - 1]]][obj_idx]
            next_val = objs[front[sorted_idx[k + 1]]][obj_idx]
            dists[sorted_idx[k]] += (next_val - prev_val) / obj_range

    return dists


# ── 主算法 ────────────────────────────────────────────────────────────

def run_nsga_de(
    tasks:      list = None,
    satellites: list = None,
    pop_size:   int   = DEFAULT_POP_SIZE,
    n_gen:      int   = DEFAULT_N_GEN,
    F:          float = DEFAULT_F,
    CR:         float = DEFAULT_CR,
    seed:       int   = DEFAULT_SEED,
    verbose:    bool  = True,
) -> dict:
    """
    运行 NSGA-DE 混合优化算法。

    Parameters
    ----------
    tasks / satellites : 若为 None，自动从 CSV 加载
    pop_size : 种群大小（建议 40~80）
    n_gen    : 迭代代数（建议 80~200）
    F        : DE 缩放因子
    CR       : DE 交叉概率
    verbose  : 是否打印进度

    Returns
    -------
    标准结果字典，含 total_profit / assignments / convergence 等
    """
    if tasks is None:
        tasks = load_tasks()
    if satellites is None:
        satellites = load_satellites()

    N   = len(tasks)
    rng = np.random.default_rng(seed)

    # ── 1. 种群初始化 ────────────────────────────────────────────────
    pop = rng.random((pop_size, N)).astype(np.float64)

    # 第 0 号个体：编码贪心排序（profit/energy 降序）
    # 这保证算法初始就至少等于贪心基线
    greedy_order = sorted(
        range(N),
        key=lambda i: tasks[i]["profit"] / max(tasks[i]["energy"], 1e-9),
        reverse=True,
    )
    keys_greedy = np.empty(N, dtype=np.float64)
    for rank, task_idx in enumerate(greedy_order):
        keys_greedy[task_idx] = 1.0 - rank / N   # 最高比率 → 接近 1
    pop[0] = keys_greedy

    # ── 2. 评估初始种群 ──────────────────────────────────────────────
    profits     = np.zeros(pop_size)
    completions = np.zeros(pop_size)
    all_assigns = [None] * pop_size

    for i in range(pop_size):
        profits[i], completions[i], all_assigns[i] = _evaluate(pop[i], tasks, satellites)

    convergence = []
    t0 = time.time()

    if verbose:
        print(f"[NSGA-DE] 任务: {N}  卫星: {len(satellites)}  种群: {pop_size}  代数: {n_gen}")
        print(f"[NSGA-DE] 初始最高收益 = {max(profits):.2f}  (个体0贪心解 = {profits[0]:.2f})")

    # ── 3. 主进化循环 ────────────────────────────────────────────────
    for gen in range(n_gen):

        # --- DE/rand/1/bin 变异 + 交叉 ---
        trials        = np.empty_like(pop)
        trial_profits = np.zeros(pop_size)
        trial_comps   = np.zeros(pop_size)
        trial_assigns = [None] * pop_size

        idx_pool = np.arange(pop_size)
        for i in range(pop_size):
            cands = idx_pool[idx_pool != i]
            r1, r2, r3 = rng.choice(cands, 3, replace=False)

            # 变异向量，clip 保持在 [0,1]
            mutant     = pop[r1] + F * (pop[r2] - pop[r3])
            mutant     = np.clip(mutant, 0.0, 1.0)

            # 二项式交叉（至少保证一个基因来自 mutant）
            cross_mask = rng.random(N) < CR
            if not cross_mask.any():
                cross_mask[rng.integers(N)] = True
            trials[i] = np.where(cross_mask, mutant, pop[i])

        # 评估试验种群
        for i in range(pop_size):
            trial_profits[i], trial_comps[i], trial_assigns[i] = \
                _evaluate(trials[i], tasks, satellites)

        # --- 合并父代 + 子代（共 2P 个个体） ---
        combined_pop     = np.vstack([pop, trials])
        combined_profits = np.concatenate([profits, trial_profits])
        combined_comps   = np.concatenate([completions, trial_comps])
        combined_assigns = all_assigns + trial_assigns
        combined_objs    = list(zip(combined_profits.tolist(), combined_comps.tolist()))

        # --- NSGA-II 非支配排序 + 拥挤距离选 P 个存活 ---
        fronts    = _fast_non_dominated_sort(combined_objs)
        survivors = []

        for front in fronts:
            if len(survivors) + len(front) <= pop_size:
                survivors.extend(front)
            else:
                needed = pop_size - len(survivors)
                dists  = _crowding_distance(front, combined_objs)
                ranked = sorted(zip(dists, front), reverse=True)
                survivors.extend(idx for _, idx in ranked[:needed])
                break

        # 更新种群
        pop         = combined_pop[survivors]
        profits     = combined_profits[survivors]
        completions = combined_comps[survivors]
        all_assigns = [combined_assigns[i] for i in survivors]

        best_profit = float(np.max(profits))
        convergence.append(best_profit)

        if verbose and (gen + 1) % 20 == 0:
            elapsed = time.time() - t0
            print(f"[NSGA-DE] Gen {gen+1:4d}/{n_gen}  最高收益 = {best_profit:.2f}"
                  f"  耗时 {elapsed:.1f}s")

    # ── 4. 提取最优解 ────────────────────────────────────────────────
    best_idx    = int(np.argmax(profits))
    best_profit = float(profits[best_idx])
    best_comp   = float(completions[best_idx])
    best_assign = all_assigns[best_idx] or []
    elapsed     = time.time() - t0

    if verbose:
        print(f"\n[NSGA-DE] 完成！耗时 {elapsed:.1f}s")
        print(f"[NSGA-DE] 最终最高收益 = {best_profit:.2f}"
              f"  完成率 = {best_comp*100:.1f}%"
              f"  已分配 {len(best_assign)}/{N}")

    return {
        "algorithm":       "NSGA-DE",
        "total_profit":    round(best_profit, 2),
        "assigned_tasks":  len(best_assign),
        "total_tasks":     N,
        "completion_rate": round(best_comp * 100, 2),
        "assignments":     best_assign,
        "convergence":     convergence,
        "elapsed_sec":     round(elapsed, 2),
        "params": {
            "pop_size": pop_size,
            "n_gen":    n_gen,
            "F":        F,
            "CR":       CR,
        },
    }


if __name__ == "__main__":
    from greedy import run_greedy
    g = run_greedy()
    print(f"[Greedy]  total_profit = {g['total_profit']}")
    print()
    nd = run_nsga_de(verbose=True)
    improvement = (nd["total_profit"] - g["total_profit"]) / max(g["total_profit"], 1e-9) * 100
    print(f"\n{'='*50}")
    print(f"Greedy  : {g['total_profit']}")
    print(f"NSGA-DE : {nd['total_profit']}")
    print(f"提升    : {improvement:+.2f}%")
    print(f"优于贪心: {'✓' if nd['total_profit'] > g['total_profit'] else '✗'}")
