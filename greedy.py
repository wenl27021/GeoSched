"""
GeoSched 贪心调度算法 (Greedy Scheduler)
=========================================
排序策略：按 profit / energy 比率降序，贪心地将每个任务分配给
第一颗能容纳它的可用卫星（时间窗口不冲突 + 能量充足）。

这是 NSGA-DE 的基线对比算法，也用作 NSGA-DE 的初始解种子。
"""

import csv
import os
from bisect import bisect_left

# ── 路径配置 ────────────────────────────────────────────────────────
_BASE       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TASKS_CSV  = os.path.join(_BASE, "data", "processed", "tasks_1000.csv")
_SATS_CSV   = os.path.join(_BASE, "data", "processed", "satellites_20.csv")


# ── 数据加载 ────────────────────────────────────────────────────────

def load_tasks(path: str = _TASKS_CSV) -> list:
    """读取任务 CSV，返回字典列表（跳过无效行）"""
    tasks = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            st, et = int(row["start_time"]), int(row["end_time"])
            if et <= st:          # 过滤无效时间窗口
                continue
            tasks.append({
                "task_id":    row["task_id"],
                "priority":   int(row["priority"]),
                "start_time": st,
                "end_time":   et,
                "profit":     float(row["profit"]),
                "energy":     float(row["energy"]),
                "task_type":  row["task_type"],
            })
    return tasks


def load_satellites(path: str = _SATS_CSV) -> list:
    """读取卫星 CSV，仅返回状态为'服务中'/'operational'/'在轨运行'的卫星"""
    sats = []
    active = {"服务中", "operational", "在轨运行"}
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row["status"] in active:
                sats.append({
                    "sat_id":           row["sat_id"],
                    "energy_available": float(row["energy_available_Wh"]),
                })
    return sats


# ── 时间窗口冲突检测（核心约束） ─────────────────────────────────────

def _can_insert(starts: list, ends: list, ts: int, te: int) -> bool:
    """
    判断区间 [ts, te) 是否可以插入已排好序的不冲突区间列表中。
    starts 和 ends 均按 start 升序维护且保持同步。
    使用二分查找，O(log K)。
    """
    pos = bisect_left(starts, ts)
    # 检查左邻：前一个区间的 end 是否超过 ts
    if pos > 0 and ends[pos - 1] > ts:
        return False
    # 检查右邻：下一个区间的 start 是否小于 te
    if pos < len(starts) and starts[pos] < te:
        return False
    return True


def _insert_interval(starts: list, ends: list, ts: int, te: int) -> None:
    """将区间 [ts, te) 插入排序列表，保持 starts 升序。O(K) 但 K 通常很小。"""
    pos = bisect_left(starts, ts)
    starts.insert(pos, ts)
    ends.insert(pos, te)


# ── 贪心调度核心 ────────────────────────────────────────────────────

def greedy_schedule(tasks: list, satellites: list, order=None) -> tuple:
    """
    执行贪心调度。

    Parameters
    ----------
    tasks      : load_tasks() 返回的任务列表
    satellites : load_satellites() 返回的卫星列表
    order      : 任务下标的排列顺序（list[int]）。
                 默认 None → 按 profit/energy 降序排列（标准贪心）。
                 传入自定义顺序时，贪心解码器按该顺序依次尝试分配，
                 用于 NSGA-DE 的随机键解码。

    Returns
    -------
    (assignments, total_profit)
    assignments : list of {task_id, sat_id, profit, energy}
    total_profit: float
    """
    if order is None:
        order = sorted(
            range(len(tasks)),
            key=lambda i: tasks[i]["profit"] / max(tasks[i]["energy"], 1e-9),
            reverse=True,
        )

    # 初始化每颗卫星的状态：剩余能量 + 已占用时间窗口（有序列表）
    sat_state = {
        s["sat_id"]: {
            "energy_rem": s["energy_available"],
            "starts":     [],   # 已分配任务的 start_time（升序）
            "ends":       [],   # 与 starts 同步的 end_time
        }
        for s in satellites
    }

    assignments  = []
    total_profit = 0.0

    for idx in order:
        t       = tasks[idx]
        ts, te  = t["start_time"], t["end_time"]
        ecost   = t["energy"]
        profit  = t["profit"]

        for s in satellites:
            sid = s["sat_id"]
            st  = sat_state[sid]

            if st["energy_rem"] < ecost:
                continue
            if not _can_insert(st["starts"], st["ends"], ts, te):
                continue

            # 分配成功
            st["energy_rem"] -= ecost
            _insert_interval(st["starts"], st["ends"], ts, te)
            assignments.append({
                "task_id": t["task_id"],
                "sat_id":  sid,
                "profit":  profit,
                "energy":  ecost,
            })
            total_profit += profit
            break

    return assignments, total_profit


# ── 便捷入口 ─────────────────────────────────────────────────────────

def run_greedy() -> dict:
    """直接运行贪心算法，返回标准结果字典"""
    tasks      = load_tasks()
    satellites = load_satellites()
    assignments, profit = greedy_schedule(tasks, satellites)

    return {
        "algorithm":       "Greedy",
        "total_profit":    round(profit, 2),
        "assigned_tasks":  len(assignments),
        "total_tasks":     len(tasks),
        "completion_rate": round(len(assignments) / max(len(tasks), 1) * 100, 2),
        "assignments":     assignments,
    }


if __name__ == "__main__":
    res = run_greedy()
    print(f"[Greedy] total_profit = {res['total_profit']}")
    print(f"         assigned     = {res['assigned_tasks']} / {res['total_tasks']}"
          f"  ({res['completion_rate']}%)")
