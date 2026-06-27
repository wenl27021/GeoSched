"""
GeoSched 算法统一调用入口 (model.py)
======================================
后端 backend/main.py 通过此模块调用各算法，无需关心内部实现细节。

支持的算法：
  "greedy"   - 贪心调度（基线）
  "nsga_de"  - NSGA-DE 混合优化（主算法）

用法示例：
  from model.model import run_optimization, compare_all
  result = run_optimization("nsga_de")
  report = compare_all()
"""

import os
import sys

# 确保 model/ 目录在 sys.path 中，无论从哪里 import
_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
if _MODEL_DIR not in sys.path:
    sys.path.insert(0, _MODEL_DIR)

from greedy  import load_tasks, load_satellites, greedy_schedule
from nsga_de import run_nsga_de


def run_optimization(algorithm: str = "nsga_de", **kwargs) -> dict:
    """
    统一调度入口。

    Parameters
    ----------
    algorithm : "greedy" | "nsga_de"
    **kwargs  : 传递给底层算法的额外参数
                nsga_de 支持: pop_size, n_gen, F, CR, seed, verbose

    Returns
    -------
    标准结果字典：
    {
        "algorithm":       str,
        "total_profit":    float,
        "assigned_tasks":  int,
        "total_tasks":     int,
        "completion_rate": float,   # 百分比
        "assignments":     list,
        # nsga_de 额外包含:
        "convergence":     list[float],
        "elapsed_sec":     float,
        "params":          dict,
    }
    """
    tasks      = load_tasks()
    satellites = load_satellites()

    if not tasks:
        return {"error": "无法加载任务数据，请检查 data/processed/tasks_1000.csv"}
    if not satellites:
        return {"error": "无可用卫星（所有卫星均处于非服务状态）"}

    alg = algorithm.lower().strip()

    if alg == "greedy":
        assignments, profit = greedy_schedule(tasks, satellites)
        return {
            "algorithm":       "Greedy",
            "total_profit":    round(profit, 2),
            "assigned_tasks":  len(assignments),
            "total_tasks":     len(tasks),
            "completion_rate": round(len(assignments) / max(len(tasks), 1) * 100, 2),
            "assignments":     assignments,
        }

    if alg == "nsga_de":
        return run_nsga_de(tasks=tasks, satellites=satellites, **kwargs)

    raise ValueError(f"未知算法 '{algorithm}'，支持 'greedy' 或 'nsga_de'")


def compare_all(verbose: bool = True, **nsga_kwargs) -> dict:
    """
    依次运行贪心和 NSGA-DE，返回对比报告。

    Parameters
    ----------
    verbose    : 是否打印进度
    **nsga_kwargs : 透传给 run_nsga_de 的参数（如 pop_size, n_gen）

    Returns
    -------
    {
        "greedy":     <greedy result dict>,
        "nsga_de":    <nsga_de result dict>,
        "comparison": {
            "greedy_profit":    float,
            "nsga_de_profit":   float,
            "improvement_pct":  float,
            "beats_greedy":     bool,
        }
    }
    """
    tasks      = load_tasks()
    satellites = load_satellites()

    if verbose:
        print(f"加载完成：{len(tasks)} 个任务，{len(satellites)} 颗可用卫星\n")

    # ── 贪心基线 ────────────────────────────────────────────────────
    if verbose:
        print("━" * 50)
        print("[1/2] 运行贪心算法（基线）...")
    assignments, g_profit = greedy_schedule(tasks, satellites)
    greedy_res = {
        "algorithm":       "Greedy",
        "total_profit":    round(g_profit, 2),
        "assigned_tasks":  len(assignments),
        "total_tasks":     len(tasks),
        "completion_rate": round(len(assignments) / max(len(tasks), 1) * 100, 2),
        "assignments":     assignments,
    }
    if verbose:
        print(f"  贪心收益 = {greedy_res['total_profit']:.2f}"
              f"  完成率 = {greedy_res['completion_rate']:.1f}%")

    # ── NSGA-DE 优化 ─────────────────────────────────────────────────
    if verbose:
        print(f"\n{'━'*50}")
        print("[2/2] 运行 NSGA-DE 算法...")
    nsga_res = run_nsga_de(
        tasks=tasks, satellites=satellites,
        verbose=verbose, **nsga_kwargs
    )

    # ── 对比汇总 ─────────────────────────────────────────────────────
    improvement = (
        (nsga_res["total_profit"] - greedy_res["total_profit"])
        / max(greedy_res["total_profit"], 1e-9)
        * 100
    )
    beats = nsga_res["total_profit"] > greedy_res["total_profit"]

    report = {
        "greedy":  greedy_res,
        "nsga_de": nsga_res,
        "comparison": {
            "greedy_profit":   greedy_res["total_profit"],
            "nsga_de_profit":  nsga_res["total_profit"],
            "improvement_pct": round(improvement, 2),
            "beats_greedy":    beats,
        },
    }

    if verbose:
        print(f"\n{'━'*50}")
        print("对比结果：")
        print(f"  贪心     : {greedy_res['total_profit']:.2f}")
        print(f"  NSGA-DE  : {nsga_res['total_profit']:.2f}")
        print(f"  提升幅度 : {improvement:+.2f}%")
        print(f"  优于贪心 : {'✓ YES' if beats else '✗ NO（需调整参数）'}")

    return report


if __name__ == "__main__":
    compare_all(verbose=True)
