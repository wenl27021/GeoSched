import dataclasses
from typing import Callable, List, Dict, Sequence, Tuple
import csv
import io
import os
import copy
import time
from bisect import bisect_left

import torch
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv, MessagePassing
from torch_geometric.utils import add_self_loops, degree

import numpy as np
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.core.population import Population
from pymoo.core.individual import Individual

# 1. --- 数据结构定义 ---
@dataclasses.dataclass
class Task:
    task_id: str
    priority: int
    start_time: int
    end_time: int
    profit: float
    energy: float
    task_type: str
    status: str

@dataclasses.dataclass
class Satellite:
    sat_id: str
    model: str
    energy_available_Wh: float
    status: str
    orbit_height_m: float
    inclination_deg: float
    eccentricity: float
    semi_major_axis: float
    raan: float
    arg_of_perigee: float
    windows_today: int
    completed_tasks: List[Task]
    pending_tasks: List[Task]
    utilization: float
    supported_task_types: List[str]
    total_profit: float = 0.0
    total_task_number: int = 0

# 2. --- 可行性检查工具函数 ---
def check_satellite_status(satellite: Satellite) -> bool:
    return satellite.status == "服务中"

def check_windows_today(satellite: Satellite) -> bool:
    return satellite.windows_today > 0

def check_task_type_support(task: Task, satellite: Satellite) -> bool:
    return task.task_type in satellite.supported_task_types

def check_time_conflict(task: Task, satellite: Satellite) -> bool:
    for existing_task in satellite.completed_tasks + satellite.pending_tasks:
        if max(task.start_time, existing_task.start_time) < min(task.end_time, existing_task.end_time):
            return False
    return True

def check_energy_available(task: Task, satellite: Satellite) -> bool:
    return task.energy <= satellite.energy_available_Wh

def is_feasible(task: Task, satellite: Satellite) -> bool:
    return (
        check_satellite_status(satellite) and
        check_windows_today(satellite) and
        check_task_type_support(task, satellite) and
        check_time_conflict(task, satellite) and
        check_energy_available(task, satellite)
    )

def get_conflicting_tasks(new_task: Task, satellite: Satellite) -> List[Task]:
    conflicts = []
    for existing_task in satellite.pending_tasks:
        if max(new_task.start_time, existing_task.start_time) < min(new_task.end_time, existing_task.end_time):
            conflicts.append(existing_task)
    return conflicts

# 3. --- GNN 和 ASGA 模型定义 ---

class ASGATLayer(MessagePassing):
    """自适应结构图注意力层"""
    def __init__(self, in_channels, out_channels, heads=1, concat=True, dropout=0.6):
        super(ASGATLayer, self).__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.dropout = dropout

        self.weight = Parameter(torch.Tensor(in_channels, heads * out_channels))
        self.att_weight = Parameter(torch.Tensor(1, heads, 2 * out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        torch.nn.init.xavier_uniform_(self.att_weight)

    def forward(self, x, edge_index):
        x = torch.matmul(x, self.weight)
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j, edge_index):
        x_i = x_i.view(-1, self.heads, self.out_channels)
        x_j = x_j.view(-1, self.heads, self.out_channels)
        
        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att_weight).sum(dim=-1)
        alpha = F.leaky_relu(alpha, 0.2)
        alpha = F.softmax(alpha, dim=0) 
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        return (x_j * alpha.view(-1, self.heads, 1)).view(-1, self.heads * self.out_channels)


class ASGAT_GNN(torch.nn.Module):
    """使用ASGA层的GNN模型"""
    def __init__(self, num_node_features, embedding_dim=64):
        super(ASGAT_GNN, self).__init__()
        self.asga1 = ASGATLayer(num_node_features, 128, heads=1)
        self.asga2 = ASGATLayer(128, embedding_dim, heads=1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.asga1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.asga2(x, edge_index)
        return x

# 4. --- 数据加载和图构建 ---
def detect_csv_encoding(file_path: str) -> str:
    for encoding in ("utf-8-sig", "gbk"):
        try:
            with open(file_path, 'r', encoding=encoding) as csvfile:
                csvfile.read(4096)
            return encoding
        except UnicodeDecodeError:
            continue
    return "utf-8-sig"

def get_all_unique_task_types(tasks_file_path: str) -> List[str]:
    unique_task_types = set()
    with open(tasks_file_path, 'r', encoding=detect_csv_encoding(tasks_file_path)) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            unique_task_types.add(row['task_type'])
    return list(unique_task_types)

def get_all_unique_task_types_from_files(tasks_file_paths: Sequence[str]) -> List[str]:
    unique_task_types = set()
    for tasks_file_path in tasks_file_paths:
        if os.path.exists(tasks_file_path):
            unique_task_types.update(get_all_unique_task_types(tasks_file_path))
    return list(unique_task_types)

def load_tasks_from_csv(file_path: str, only_pending: bool = True) -> List[Task]:
    tasks: List[Task] = []
    with open(file_path, 'r', encoding=detect_csv_encoding(file_path)) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if only_pending and row['status'] != "调度中":
                continue
            tasks.append(Task(
                task_id=row['task_id'],
                priority=int(row['priority']),
                start_time=int(row['start_time']),
                end_time=int(row['end_time']),
                profit=float(row['profit']),
                energy=float(row['energy']),
                task_type=row['task_type'],
                status=row['status'],
            ))
    return tasks

def parse_supported_task_types(row: Dict[str, str], all_task_types: List[str]) -> List[str]:
    raw_task_types = row.get('task_type', '')
    if not raw_task_types:
        return list(all_task_types)
    normalized = raw_task_types.replace(';', ',').replace('|', ',').replace(' ', ',')
    supported_task_types = [task_type for task_type in normalized.split(',') if task_type]
    return supported_task_types or list(all_task_types)

def load_satellites(file_path: str, all_task_types: List[str]) -> Dict[str, Satellite]:
    satellites = {}
    with open(file_path, 'r', encoding=detect_csv_encoding(file_path)) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            sat_id = row['sat_id']
            satellites[sat_id] = Satellite(
                sat_id=sat_id, model=row['model'],
                energy_available_Wh=float(row['energy_available_Wh']),
                status=row['status'],
                orbit_height_m=float(row['orbit_height_m']),
                inclination_deg=float(row['inclination_deg']),
                eccentricity=float(row['eccentricity']), 
                semi_major_axis=float(row['semi_major_axis']),
                raan=float(row['raan']), 
                arg_of_perigee=float(row['arg_of_perigee']),
                windows_today=int(row['windows_today']), 
                utilization=float(row['utilization']),
                completed_tasks=[], 
                pending_tasks=[], 
                supported_task_types=parse_supported_task_types(row, all_task_types)
            )
    return satellites

def build_graph(satellites: Dict[str, Satellite], tasks: List[Task]):
    num_features = 10
    node_features, satellite_indices, task_indices = [], {}, {}
    for idx, (sat_id, sat) in enumerate(satellites.items()):
        features = torch.tensor([
            sat.energy_available_Wh / 1000.0,
            sat.windows_today / 10000.0,
            sat.orbit_height_m / 1_000_000.0,
            sat.inclination_deg / 180.0,
            sat.eccentricity,
            sat.semi_major_axis / 10_000_000.0,
            sat.raan / 360.0,
            sat.arg_of_perigee / 360.0,
            sat.utilization,
            0.0,
        ], dtype=torch.float)
        node_features.append(features)
        satellite_indices[sat_id] = idx
    task_start_idx = len(satellites)
    for idx, task in enumerate(tasks):
        duration = max(task.end_time - task.start_time, 1)
        features = torch.tensor([
            task.priority / 10.0,
            task.profit / 250.0,
            task.energy / 10.0,
            duration / 5000.0,
            task.start_time / 10000.0,
            task.end_time / 10000.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ], dtype=torch.float)
        node_features.append(features)
        task_indices[task.task_id] = task_start_idx + idx
    x = torch.stack(node_features)
    edge_index_list = []
    for sat_id, sat in satellites.items():
        sat_idx = satellite_indices[sat_id]
        for task_offset, task in enumerate(tasks):
            if task.task_type in sat.supported_task_types:
                task_idx = task_start_idx + task_offset
                edge_index_list.extend([[sat_idx, task_idx], [task_idx, sat_idx]])
    if not edge_index_list:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
    data = Data(x=x, edge_index=edge_index)
    return data, satellite_indices, task_indices

# 5. --- 模型训练与评估 ---
def normalize_graph_features(data: Data) -> Data:
    data = copy.copy(data)
    mean = data.x.mean(dim=0, keepdim=True)
    std = data.x.std(dim=0, keepdim=True)
    data.x = (data.x - mean) / (std + 1e-6)
    return data

def train_gnn_model(model, data, epochs=50):
    """训练GNN模型"""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    model.train()
    data = normalize_graph_features(data)
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data)
        # 简单的自监督损失：尝试重构邻接矩阵
        adj_pred = torch.sigmoid(torch.matmul(out, out.t()))
        true_adj = torch.zeros(data.num_nodes, data.num_nodes)
        true_adj[data.edge_index[0], data.edge_index[1]] = 1
        loss = F.binary_cross_entropy(adj_pred, true_adj)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f'GNN Training Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')
    model.eval()
    return model

def get_embeddings(model, satellites: Dict[str, Satellite], tasks: List[Task]):
    data, sat_indices, task_indices = build_graph(satellites, tasks)
    data = normalize_graph_features(data)
    with torch.no_grad():
        embeddings = model(data)
    sat_embeddings = {sat_id: embeddings[idx] for sat_id, idx in sat_indices.items()}
    task_embeddings = {task_id: embeddings[idx] for task_id, idx in task_indices.items()}
    return sat_embeddings, task_embeddings

def _sample_list(items: Sequence, sample_size: int, rng: np.random.Generator) -> List:
    if not items:
        return []
    actual_size = min(sample_size, len(items))
    indices = rng.choice(len(items), size=actual_size, replace=False)
    return [copy.deepcopy(items[int(idx)]) for idx in indices]

def _keys_from_numeric_scores(scores: Sequence[float]) -> np.ndarray:
    if not scores:
        return np.array([], dtype=np.float64)
    keys = np.zeros(len(scores), dtype=np.float64)
    ranked_indices = sorted(
        range(len(scores)),
        key=lambda idx: (float(scores[idx]), -idx),
        reverse=True,
    )
    for rank, task_idx in enumerate(ranked_indices):
        keys[task_idx] = 1.0 - rank / max(len(scores), 1)
    return keys

def _best_seed_schedule_for_pretraining(
    tasks: Sequence[Task],
    satellites: Sequence[Satellite],
    rng: np.random.Generator,
) -> Tuple[float, int, float, List[Tuple[int, Task]]]:
    if not tasks or not satellites:
        return 0.0, 0, sum(sat.energy_available_Wh for sat in satellites), []
    seed_keys = _seed_random_key_population(
        tasks,
        pop_size=min(12, max(8, len(tasks))),
        rng=rng,
    )
    best_result = None
    for keys in seed_keys:
        result = _evaluate_random_keys(keys, tasks, satellites)
        if best_result is None or result[:3] > best_result[:3]:
            best_result = result
    return best_result if best_result is not None else (0.0, 0, 0.0, [])

def pretrain_gnn_for_nsga_de(
    task_files: Sequence[str],
    satellite_files: Sequence[str],
    all_task_types: Sequence[str],
    sample_count: int = 24,
    tasks_per_sample: int = 80,
    satellites_per_sample: int = 12,
    epochs_per_sample: int = 1,
    seed: int = 42,
    log_file=None,
) -> ASGAT_GNN:
    """Pretrain GNN as a task-satellite edge scorer for NSGA-DE seeds.

    Each mini-scenario is sampled from the provided task/satellite CSV pools.
    A strong heuristic decoder creates pseudo labels: selected task-satellite
    edges are positive, other candidate edges are negative.
    """
    rng = np.random.default_rng(seed)
    task_pools = [
        load_tasks_from_csv(path, only_pending=True)
        for path in task_files
        if os.path.exists(path)
    ]
    task_pools = [pool for pool in task_pools if pool]
    satellite_pools = [
        list(load_satellites(path, list(all_task_types)).values())
        for path in satellite_files
        if os.path.exists(path)
    ]
    satellite_pools = [pool for pool in satellite_pools if pool]

    model = ASGAT_GNN(num_node_features=10)
    if not task_pools or not satellite_pools:
        if log_file is not None:
            log_file.write("GNN 预训练跳过: 缺少可用任务或卫星预训练数据\n")
        model.eval()
        return model

    optimizer = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-4)
    completed_steps = 0
    model.train()

    for sample_idx in range(sample_count):
        tasks = _sample_list(
            task_pools[int(rng.integers(len(task_pools)))],
            tasks_per_sample,
            rng,
        )
        sampled_satellites = _sample_list(
            satellite_pools[int(rng.integers(len(satellite_pools)))],
            satellites_per_sample,
            rng,
        )
        if not tasks or not sampled_satellites:
            continue

        _, _, _, accepted = _best_seed_schedule_for_pretraining(tasks, sampled_satellites, rng)
        positive_pairs = {(sat_idx, id(task)) for sat_idx, task in accepted}
        if not positive_pairs:
            continue

        satellites_dict = {sat.sat_id: sat for sat in sampled_satellites}
        data, _, _ = build_graph(satellites_dict, tasks)
        if data.edge_index.numel() == 0:
            continue
        data = normalize_graph_features(data)

        sat_node_indices, task_node_indices, labels = [], [], []
        task_start_idx = len(sampled_satellites)
        for task_idx, task in enumerate(tasks):
            for sat_idx, satellite in enumerate(sampled_satellites):
                if task.task_type not in satellite.supported_task_types:
                    continue
                sat_node_indices.append(sat_idx)
                task_node_indices.append(task_start_idx + task_idx)
                labels.append(1.0 if (sat_idx, id(task)) in positive_pairs else 0.0)

        if not labels or max(labels) <= 0:
            continue

        sat_node_tensor = torch.tensor(sat_node_indices, dtype=torch.long)
        task_node_tensor = torch.tensor(task_node_indices, dtype=torch.long)
        label_tensor = torch.tensor(labels, dtype=torch.float)
        pos_count = label_tensor.sum().clamp(min=1.0)
        neg_count = (label_tensor.numel() - label_tensor.sum()).clamp(min=1.0)
        pos_weight = neg_count / pos_count

        for _ in range(max(1, epochs_per_sample)):
            optimizer.zero_grad()
            embeddings = model(data)
            logits = (embeddings[sat_node_tensor] * embeddings[task_node_tensor]).sum(dim=-1)
            loss = F.binary_cross_entropy_with_logits(logits, label_tensor, pos_weight=pos_weight)
            loss.backward()
            optimizer.step()
            completed_steps += 1

        if log_file is not None and (sample_idx + 1) % 5 == 0:
            log_file.write(
                f"GNN 预训练样本 {sample_idx + 1}/{sample_count}, "
                f"loss={loss.item():.4f}, positives={int(pos_count.item())}\n"
            )

    model.eval()
    if log_file is not None:
        log_file.write(f"GNN 预训练完成: optimization_steps={completed_steps}\n")
    return model

def _canonical_paths(paths: Sequence[str]) -> List[str]:
    return [os.path.abspath(path) for path in paths]

def load_or_pretrain_gnn_for_nsga_de(
    cache_path: str,
    task_files: Sequence[str],
    satellite_files: Sequence[str],
    all_task_types: Sequence[str],
    sample_count: int = 24,
    tasks_per_sample: int = 80,
    satellites_per_sample: int = 12,
    epochs_per_sample: int = 1,
    seed: int = 42,
    force_retrain: bool = False,
    log_file=None,
) -> ASGAT_GNN:
    """Load a cached GNN checkpoint, or pretrain and cache it."""
    config = {
        "task_files": _canonical_paths(task_files),
        "satellite_files": _canonical_paths(satellite_files),
        "all_task_types": sorted(str(task_type) for task_type in all_task_types),
        "sample_count": int(sample_count),
        "tasks_per_sample": int(tasks_per_sample),
        "satellites_per_sample": int(satellites_per_sample),
        "epochs_per_sample": int(epochs_per_sample),
        "seed": int(seed),
        "num_node_features": 10,
        "embedding_dim": 64,
    }

    if os.path.exists(cache_path) and not force_retrain:
        try:
            try:
                checkpoint = torch.load(cache_path, map_location="cpu", weights_only=True)
            except TypeError:
                checkpoint = torch.load(cache_path, map_location="cpu")

            if checkpoint.get("config") == config:
                model = ASGAT_GNN(
                    num_node_features=config["num_node_features"],
                    embedding_dim=config["embedding_dim"],
                )
                model.load_state_dict(checkpoint["model_state_dict"])
                model.eval()
                if log_file is not None:
                    log_file.write(f"GNN 缓存已加载: {cache_path}\n")
                return model

            if log_file is not None:
                log_file.write("GNN 缓存配置已变化，将重新预训练\n")
        except Exception as exc:
            if log_file is not None:
                log_file.write(f"GNN 缓存加载失败，将重新预训练: {exc}\n")

    model = pretrain_gnn_for_nsga_de(
        task_files,
        satellite_files,
        all_task_types,
        sample_count=sample_count,
        tasks_per_sample=tasks_per_sample,
        satellites_per_sample=satellites_per_sample,
        epochs_per_sample=epochs_per_sample,
        seed=seed,
        log_file=log_file,
    )
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save(
        {
            "config": config,
            "model_state_dict": model.state_dict(),
        },
        cache_path,
    )
    if log_file is not None:
        log_file.write(f"GNN 缓存已保存: {cache_path}\n")
    return model

def build_gnn_guided_key_vectors(
    gnn_model,
    tasks: Sequence[Task],
    satellites: Sequence[Satellite],
) -> List[np.ndarray]:
    if gnn_model is None or not tasks or not satellites:
        return []

    satellites_dict = {sat.sat_id: sat for sat in satellites}
    data, _, _ = build_graph(satellites_dict, list(tasks))
    if data.edge_index.numel() == 0:
        return []
    data = normalize_graph_features(data)

    gnn_model.eval()
    with torch.no_grad():
        embeddings = gnn_model(data)

    task_start_idx = len(satellites)
    raw_scores = []
    for task_idx, task in enumerate(tasks):
        task_embedding = embeddings[task_start_idx + task_idx]
        candidate_logits = []
        for sat_idx, satellite in enumerate(satellites):
            if satellite.status != "服务中":
                continue
            if task.task_type not in satellite.supported_task_types:
                continue
            candidate_logits.append(torch.dot(embeddings[sat_idx], task_embedding).item())
        raw_scores.append(max(candidate_logits) if candidate_logits else -1e9)

    profit_scores = [score * task.profit for score, task in zip(raw_scores, tasks)]
    density_scores = [
        score * task.profit / _task_duration(task)
        for score, task in zip(raw_scores, tasks)
    ]
    priority_scores = [
        score + task.priority * 0.1 + task.profit / 250.0
        for score, task in zip(raw_scores, tasks)
    ]

    return [
        _keys_from_numeric_scores(raw_scores),
        _keys_from_numeric_scores(profit_scores),
        _keys_from_numeric_scores(density_scores),
        _keys_from_numeric_scores(priority_scores),
    ]

# 6. --- 调度算法 --- 

# (plain_schedule, greedy_schedule, greedy_schedule_agent 函数保持不变, 这里省略)

def try_assign_with_replacement(new_task: Task, satellite: Satellite) -> (bool, List[Task], float):
    if not (check_satellite_status(satellite) and check_windows_today(satellite) and check_task_type_support(new_task, satellite)):
        return False, [], -1.0
    conflicting_tasks = get_conflicting_tasks(new_task, satellite)
    replaced_tasks, energy_after_replacements = [], satellite.energy_available_Wh
    for existing_task in satellite.pending_tasks:
        if existing_task in conflicting_tasks:
            if new_task.priority > existing_task.priority:
                energy_after_replacements += existing_task.energy
                replaced_tasks.append(existing_task)
            else:
                return False, [], -1.0
    if energy_after_replacements < new_task.energy:
        return False, [], -1.0
    return True, replaced_tasks, energy_after_replacements - new_task.energy

def plain_schedule(tasks_file_path: str, all_satellites: Dict[str, Satellite], log_file):
    all_tasks_to_schedule: List[Task] = []
    with open(tasks_file_path, 'r', encoding=detect_csv_encoding(tasks_file_path)) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['status'] == "调度中":
                all_tasks_to_schedule.append(Task(task_id=row['task_id'], priority=int(row['priority']), start_time=int(row['start_time']), end_time=int(row['end_time']), profit=float(row['profit']), energy=float(row['energy']), task_type=row['task_type'], status=row['status']))
    log_file.write("--- 开始顺序调度 ---\n")
    for task in all_tasks_to_schedule:
        log_file.write(f"任务信息: {task}\n调度结果：\n")
        assigned = False
        for sat_id, satellite in all_satellites.items():
            if is_feasible(task, satellite):
                satellite.pending_tasks.append(task)
                satellite.windows_today -= 1
                satellite.energy_available_Wh -= task.energy
                satellite.total_profit += task.profit
                satellite.total_task_number += 1
                log_file.write(f"任务分配给卫星: {satellite.sat_id}\n卫星当前状态: {satellite}\n")
                assigned = True
                break
        if not assigned:
            log_file.write("未找到合适的卫星\n")

def greedy_schedule(tasks_file_path: str, all_satellites: Dict[str, Satellite], log_file):
    all_tasks_to_schedule: List[Task] = []
    with open(tasks_file_path, 'r', encoding=detect_csv_encoding(tasks_file_path)) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['status'] == "调度中":
                all_tasks_to_schedule.append(Task(task_id=row['task_id'], priority=int(row['priority']), start_time=int(row['start_time']), end_time=int(row['end_time']), profit=float(row['profit']), energy=float(row['energy']), task_type=row['task_type'], status=row['status']))
    all_tasks_to_schedule.sort(key=lambda t: (t.priority, t.profit), reverse=True)
    log_file.write("--- 开始贪婪调度 ---\n")
    for task in all_tasks_to_schedule:
        log_file.write(f"任务信息: {task}\n调度结果：\n")
        best_satellite_id, min_replaced_tasks, best_replaced_tasks_for_assignment = None, float('inf'), []
        for sat_id, satellite in all_satellites.items():
            can_assign, current_replaced_tasks, _ = try_assign_with_replacement(task, satellite)
            if can_assign and len(current_replaced_tasks) < min_replaced_tasks:
                min_replaced_tasks = len(current_replaced_tasks)
                best_satellite_id = sat_id
                best_replaced_tasks_for_assignment = current_replaced_tasks
        if best_satellite_id:
            original_satellite = all_satellites[best_satellite_id]
            for replaced_task in best_replaced_tasks_for_assignment:
                original_satellite.energy_available_Wh += replaced_task.energy
                original_satellite.total_profit -= replaced_task.profit
                original_satellite.total_task_number -= 1
                original_satellite.pending_tasks.remove(replaced_task)
            original_satellite.pending_tasks.append(task)
            original_satellite.windows_today -= 1
            original_satellite.energy_available_Wh -= task.energy
            original_satellite.total_profit += task.profit
            original_satellite.total_task_number += 1
            log_file.write(f"任务分配给卫星: {original_satellite.sat_id}\n卫星当前状态: {original_satellite}\n")
        else:
            log_file.write("未找到合适的卫星\n")
    log_file.write("--- 贪婪调度结束 ---\n")

def greedy_schedule_agent(tasks_file_path: str, all_satellites: Dict[str, Satellite], log_file):
    all_tasks_to_schedule: List[Task] = []
    with open(tasks_file_path, 'r', encoding=detect_csv_encoding(tasks_file_path)) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['status'] == "调度中":
                all_tasks_to_schedule.append(Task(task_id=row['task_id'], priority=int(row['priority']), start_time=int(row['start_time']), end_time=int(row['end_time']), profit=float(row['profit']), energy=float(row['energy']), task_type=row['task_type'], status=row['status']))
    all_tasks_to_schedule.sort(key=lambda t: (t.priority, t.profit), reverse=True)
    log_file.write("--- 开始智能贪婪调度 ---\n")
    for task in all_tasks_to_schedule:
        log_file.write(f"任务信息: {task}\n调度结果：\n")
        best_satellite_id, min_replaced_tasks, max_net_profit_gain = None, float('inf'), -float('inf')
        for sat_id, satellite in all_satellites.items():
            can_assign, replaced_tasks, _ = try_assign_with_replacement(task, satellite)
            if can_assign:
                current_net_profit_gain = task.profit - sum(t.profit for t in replaced_tasks)
                num_replaced = len(replaced_tasks)
                if num_replaced < min_replaced_tasks or (num_replaced == min_replaced_tasks and current_net_profit_gain > max_net_profit_gain):
                    min_replaced_tasks = num_replaced
                    max_net_profit_gain = current_net_profit_gain
                    best_satellite_id = sat_id
        if best_satellite_id:
            original_satellite = all_satellites[best_satellite_id]
            conflicting_tasks = get_conflicting_tasks(task, original_satellite)
            for ct in conflicting_tasks:
                if task.priority > ct.priority:
                    original_satellite.energy_available_Wh += ct.energy
                    original_satellite.total_profit -= ct.profit
                    original_satellite.total_task_number -= 1
                    original_satellite.pending_tasks.remove(ct)
            original_satellite.pending_tasks.append(task)
            original_satellite.windows_today -= 1
            original_satellite.energy_available_Wh -= task.energy
            original_satellite.total_profit += task.profit
            original_satellite.total_task_number += 1
            log_file.write(f"任务分配给卫星: {original_satellite.sat_id}\n卫星当前状态: {original_satellite}\n")
        else:
            log_file.write("未找到合适的卫星\n")
    log_file.write("--- 智能贪婪调度结束 ---\n")

class SatelliteTaskSchedulingProblem(Problem):
    def __init__(self, tasks, satellites, task_embeddings, sat_embeddings):
        self.tasks, self.satellites, self.task_embeddings, self.sat_embeddings = tasks, list(satellites.values()), task_embeddings, sat_embeddings
        self.num_tasks, self.num_satellites = len(tasks), len(self.satellites)
        super().__init__(n_var=self.num_tasks, n_obj=2, n_constr=0, xl=0, xu=self.num_satellites - 1, type_var=int)

    def _evaluate(self, X, out, *args, **kwargs):
        F = np.zeros((X.shape[0], self.n_obj))
        for i, x in enumerate(X):
            total_profit, total_tasks, _ = evaluate_assignment(self.tasks, self.satellites, x)
            F[i, 0], F[i, 1] = -total_profit, -total_tasks
        out["F"] = F


def evaluate_assignment(
    tasks: Sequence[Task],
    satellites: Sequence[Satellite],
    assignment: Sequence[int],
) -> Tuple[float, int, List[Tuple[int, Task]]]:
    """Evaluate one assignment using lexicographic scheduling metrics.

    The returned values count only newly accepted tasks. Existing satellite
    pending/completed tasks are treated as occupied resources for conflict
    checks, but are not counted again in the candidate score.
    """
    total_profit, total_tasks = 0.0, 0
    sat_energy = [sat.energy_available_Wh for sat in satellites]
    sat_windows = [sat.windows_today for sat in satellites]
    sat_assigned_tasks = [
        list(sat.completed_tasks) + list(sat.pending_tasks)
        for sat in satellites
    ]
    accepted: List[Tuple[int, Task]] = []

    for task_idx, task in enumerate(tasks):
        if task_idx >= len(assignment):
            break
        sat_idx = int(assignment[task_idx])
        if sat_idx < 0 or sat_idx >= len(satellites):
            continue

        satellite = satellites[sat_idx]
        if satellite.status != "服务中":
            continue
        if sat_windows[sat_idx] <= 0:
            continue
        if task.task_type not in satellite.supported_task_types:
            continue
        if sat_energy[sat_idx] < task.energy:
            continue

        has_conflict = any(
            max(task.start_time, existing_task.start_time) < min(task.end_time, existing_task.end_time)
            for existing_task in sat_assigned_tasks[sat_idx]
        )
        if has_conflict:
            continue

        total_profit += task.profit
        total_tasks += 1
        sat_energy[sat_idx] -= task.energy
        sat_windows[sat_idx] -= 1
        sat_assigned_tasks[sat_idx].append(task)
        accepted.append((sat_idx, task))

    return total_profit, total_tasks, accepted


def select_lexicographic_best_assignment(
    candidate_assignments: Sequence[Sequence[int]],
    tasks: Sequence[Task],
    satellites: Sequence[Satellite],
    profit_tolerance: float = 1e-9,
) -> Tuple[np.ndarray, float, int]:
    """Select the best candidate by profit first, then task count.

    NSGA-III supplies a Pareto set. This selector makes the final decision
    deterministic and strict: maximize total_profit; among equal-profit
    candidates, maximize total_task_number.
    """
    best_assignment = None
    best_profit = -float("inf")
    best_task_count = -1
    best_remaining_energy = -float("inf")

    for assignment in candidate_assignments:
        total_profit, total_tasks, accepted = evaluate_assignment(tasks, satellites, assignment)
        remaining_energy = (
            sum(satellite.energy_available_Wh for satellite in satellites)
            - sum(task.energy for _, task in accepted)
        )
        profit_is_better = total_profit > best_profit + profit_tolerance
        profit_is_equal = abs(total_profit - best_profit) <= profit_tolerance
        task_count_is_better = total_tasks > best_task_count
        task_count_is_equal = total_tasks == best_task_count
        energy_is_better = remaining_energy > best_remaining_energy + profit_tolerance

        if (
            profit_is_better
            or (
                profit_is_equal
                and (
                    task_count_is_better
                    or (task_count_is_equal and energy_is_better)
                )
            )
        ):
            best_assignment = np.asarray(assignment, dtype=int)
            best_profit = total_profit
            best_task_count = total_tasks
            best_remaining_energy = remaining_energy

    if best_assignment is None:
        return np.array([], dtype=int), 0.0, 0

    return best_assignment, best_profit, best_task_count


def apply_assignment_to_satellites(
    tasks: Sequence[Task],
    satellites: Sequence[Satellite],
    assignment: Sequence[int],
    log_file=None,
) -> Tuple[float, int]:
    """Apply one assignment with the same feasibility rules used in evaluation."""
    total_profit, total_tasks, accepted = evaluate_assignment(tasks, satellites, assignment)
    for sat_idx, task in accepted:
        satellite = satellites[sat_idx]
        satellite.pending_tasks.append(task)
        satellite.windows_today -= 1
        satellite.energy_available_Wh -= task.energy
        satellite.total_profit += task.profit
        satellite.total_task_number += 1
        if log_file is not None:
            log_file.write(f"任务 {task.task_id} 分配给卫星 {satellite.sat_id}\n")

    return total_profit, total_tasks


def _task_duration(task: Task) -> int:
    return max(task.end_time - task.start_time, 1)


def _can_insert_interval(starts: List[int], ends: List[int], start: int, end: int) -> bool:
    pos = bisect_left(starts, start)
    if pos > 0 and ends[pos - 1] > start:
        return False
    if pos < len(starts) and starts[pos] < end:
        return False
    return True


def _insert_interval(starts: List[int], ends: List[int], start: int, end: int) -> None:
    pos = bisect_left(starts, start)
    starts.insert(pos, start)
    ends.insert(pos, end)


def decode_task_order(
    tasks: Sequence[Task],
    satellites: Sequence[Satellite],
    order: Sequence[int],
) -> Tuple[float, int, float, List[Tuple[int, Task]]]:
    """Decode a task order into a feasible schedule using profit-preserving best fit.

    The random-key optimizer searches only over task order. This decoder is
    deterministic and keeps every returned schedule feasible by checking status,
    windows, task type, energy, and time conflicts before each insertion.
    """
    sat_energy = [sat.energy_available_Wh for sat in satellites]
    sat_windows = [sat.windows_today for sat in satellites]
    sat_starts: List[List[int]] = []
    sat_ends: List[List[int]] = []

    for sat in satellites:
        intervals = sorted(
            (task.start_time, task.end_time)
            for task in list(sat.completed_tasks) + list(sat.pending_tasks)
        )
        sat_starts.append([start for start, _ in intervals])
        sat_ends.append([end for _, end in intervals])

    total_profit, total_tasks = 0.0, 0
    accepted: List[Tuple[int, Task]] = []

    for task_idx in order:
        if task_idx < 0 or task_idx >= len(tasks):
            continue
        task = tasks[int(task_idx)]
        best_sat_idx = None
        best_score = None

        for sat_idx, satellite in enumerate(satellites):
            if satellite.status != "服务中":
                continue
            if sat_windows[sat_idx] <= 0:
                continue
            if task.task_type not in satellite.supported_task_types:
                continue
            if sat_energy[sat_idx] < task.energy:
                continue
            if not _can_insert_interval(
                sat_starts[sat_idx],
                sat_ends[sat_idx],
                task.start_time,
                task.end_time,
            ):
                continue

            energy_after = sat_energy[sat_idx] - task.energy
            windows_after = sat_windows[sat_idx] - 1
            score = (energy_after, windows_after, satellite.utilization, sat_idx)
            if best_score is None or score < best_score:
                best_score = score
                best_sat_idx = sat_idx

        if best_sat_idx is None:
            continue

        sat_energy[best_sat_idx] -= task.energy
        sat_windows[best_sat_idx] -= 1
        _insert_interval(
            sat_starts[best_sat_idx],
            sat_ends[best_sat_idx],
            task.start_time,
            task.end_time,
        )
        total_profit += task.profit
        total_tasks += 1
        accepted.append((best_sat_idx, task))

    return total_profit, total_tasks, float(sum(sat_energy)), accepted


def apply_decoded_schedule_to_satellites(
    satellites: Sequence[Satellite],
    accepted: Sequence[Tuple[int, Task]],
    log_file=None,
) -> None:
    for sat_idx, task in accepted:
        satellite = satellites[sat_idx]
        satellite.pending_tasks.append(task)
        satellite.windows_today -= 1
        satellite.energy_available_Wh -= task.energy
        satellite.total_profit += task.profit
        satellite.total_task_number += 1
        if log_file is not None:
            log_file.write(f"任务 {task.task_id} 分配给卫星 {satellite.sat_id}\n")


def _keys_from_task_score(
    tasks: Sequence[Task],
    score_fn: Callable[[Task], Tuple[float, ...]],
) -> np.ndarray:
    n_tasks = len(tasks)
    keys = np.zeros(n_tasks, dtype=np.float64)
    ranked_task_indices = sorted(
        range(n_tasks),
        key=lambda idx: (score_fn(tasks[idx]), -idx),
        reverse=True,
    )
    for rank, task_idx in enumerate(ranked_task_indices):
        keys[task_idx] = 1.0 - rank / max(n_tasks, 1)
    return keys


def _seed_random_key_population(
    tasks: Sequence[Task],
    pop_size: int,
    rng: np.random.Generator,
    extra_seed_vectors: Sequence[np.ndarray] = (),
) -> np.ndarray:
    heuristic_scores: List[Callable[[Task], Tuple[float, ...]]] = [
        lambda task: (task.priority, task.profit),
        lambda task: (task.profit,),
        lambda task: (task.profit / _task_duration(task), task.profit),
        lambda task: (task.profit / max(task.energy, 1e-9), task.profit),
        lambda task: (
            task.profit / max(task.energy * _task_duration(task), 1e-9),
            task.profit,
        ),
        lambda task: (task.priority, task.profit / _task_duration(task), task.profit),
        lambda task: (-_task_duration(task), task.profit),
    ]

    base_vectors = [_keys_from_task_score(tasks, score_fn) for score_fn in heuristic_scores]
    population = []
    for seed_vector in extra_seed_vectors:
        seed_array = np.asarray(seed_vector, dtype=np.float64)
        if seed_array.shape == (len(tasks),):
            population.append(np.clip(seed_array, 0.0, 1.0))
        if len(population) >= pop_size:
            break
    for keys in base_vectors:
        population.append(keys)
        if len(population) >= pop_size:
            break

    jitter_scale = 0.03
    while len(population) < min(pop_size, len(base_vectors) * 4):
        base = base_vectors[len(population) % len(base_vectors)]
        jittered = np.clip(base + rng.normal(0.0, jitter_scale, size=len(tasks)), 0.0, 1.0)
        population.append(jittered)

    while len(population) < pop_size:
        population.append(rng.random(len(tasks)))

    return np.vstack(population[:pop_size]).astype(np.float64)


def _evaluate_random_keys(
    keys: np.ndarray,
    tasks: Sequence[Task],
    satellites: Sequence[Satellite],
) -> Tuple[float, int, float, List[Tuple[int, Task]]]:
    order = np.argsort(-keys).tolist()
    return decode_task_order(tasks, satellites, order)


def _teacher_schedule_for_gnn(
    tasks: Sequence[Task],
    satellites: Sequence[Satellite],
) -> List[Tuple[int, Task]]:
    """Use the best deterministic seed order as pseudo labels for GNN pretraining."""
    teacher_vectors = [
        _keys_from_task_score(tasks, lambda task: (task.priority, task.profit)),
        _keys_from_task_score(tasks, lambda task: (task.profit,)),
        _keys_from_task_score(tasks, lambda task: (task.profit / _task_duration(task), task.profit)),
        _keys_from_task_score(tasks, lambda task: (task.profit / max(task.energy, 1e-9), task.profit)),
        _keys_from_task_score(
            tasks,
            lambda task: (
                task.profit / max(task.energy * _task_duration(task), 1e-9),
                task.profit,
            ),
        ),
    ]
    best_profit, best_task_count, best_energy_sum, best_accepted = -float("inf"), -1, -float("inf"), []
    for keys in teacher_vectors:
        total_profit, task_count, energy_sum, accepted = _evaluate_random_keys(keys, tasks, satellites)
        if (
            total_profit > best_profit
            or (
                abs(total_profit - best_profit) <= 1e-9
                and (
                    task_count > best_task_count
                    or (task_count == best_task_count and energy_sum > best_energy_sum)
                )
            )
        ):
            best_profit = total_profit
            best_task_count = task_count
            best_energy_sum = energy_sum
            best_accepted = accepted
    return best_accepted


def _legacy_single_file_pretrain_gnn_for_nsga_de(
    pretrain_tasks_file: str,
    satellites_file: str,
    all_task_types: Sequence[str],
    num_scenarios: int = 12,
    task_sample_size: int = 80,
    epochs_per_scenario: int = 2,
    embedding_dim: int = 64,
    seed: int = 42,
    verbose: bool = True,
):
    """Pretrain ASGAT on random sub-scenarios as a scheduler prior.

    Each sub-scenario is sampled from tasks_1000.csv. Pseudo labels come from
    the strongest deterministic decoder seed, so the GNN learns which task-sat
    edges tend to appear in high-profit feasible schedules.
    """
    pretrain_tasks = load_tasks_from_csv(pretrain_tasks_file, only_pending=True)
    if not pretrain_tasks:
        if verbose:
            print("[GNN] 预训练跳过：没有可用于预训练的调度中任务")
        return None

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    satellites_template = load_satellites(satellites_file, list(all_task_types))
    first_sample_size = min(task_sample_size, len(pretrain_tasks))
    first_tasks = [
        pretrain_tasks[int(idx)]
        for idx in rng.choice(len(pretrain_tasks), size=first_sample_size, replace=False)
    ]
    first_graph, _, _ = build_graph(satellites_template, first_tasks)
    model = ASGAT_GNN(num_node_features=first_graph.x.shape[1], embedding_dim=embedding_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=5e-4)

    model.train()
    for scenario_idx in range(max(1, num_scenarios)):
        sample_size = min(task_sample_size, len(pretrain_tasks))
        sampled_indices = rng.choice(len(pretrain_tasks), size=sample_size, replace=False)
        sampled_tasks = [pretrain_tasks[int(idx)] for idx in sampled_indices]
        satellites = copy.deepcopy(satellites_template)
        satellite_list = list(satellites.values())
        accepted = _teacher_schedule_for_gnn(sampled_tasks, satellite_list)
        positive_edges = {
            (satellite_list[sat_idx].sat_id, task.task_id)
            for sat_idx, task in accepted
        }

        data, sat_indices, task_indices = build_graph(satellites, sampled_tasks)
        if data.edge_index.numel() == 0:
            continue
        data = normalize_graph_features(data)

        edge_pairs = []
        labels = []
        for sat_id, sat_idx in sat_indices.items():
            for task in sampled_tasks:
                if task.task_type not in satellites[sat_id].supported_task_types:
                    continue
                edge_pairs.append((sat_idx, task_indices[task.task_id]))
                labels.append(1.0 if (sat_id, task.task_id) in positive_edges else 0.0)

        if not edge_pairs or not any(label > 0 for label in labels):
            continue

        edge_pairs_tensor = torch.tensor(edge_pairs, dtype=torch.long)
        labels_tensor = torch.tensor(labels, dtype=torch.float)
        pos_count = labels_tensor.sum().clamp(min=1.0)
        neg_count = (labels_tensor.numel() - labels_tensor.sum()).clamp(min=1.0)
        pos_weight = neg_count / pos_count

        for _ in range(max(1, epochs_per_scenario)):
            optimizer.zero_grad()
            embeddings = model(data)
            src_embeddings = embeddings[edge_pairs_tensor[:, 0]]
            dst_embeddings = embeddings[edge_pairs_tensor[:, 1]]
            logits = (src_embeddings * dst_embeddings).sum(dim=1) / np.sqrt(embeddings.shape[1])
            loss = F.binary_cross_entropy_with_logits(
                logits,
                labels_tensor,
                pos_weight=pos_weight,
            )
            loss.backward()
            optimizer.step()

        if verbose and (scenario_idx + 1) % 4 == 0:
            print(f"[GNN] 预训练场景 {scenario_idx + 1}/{num_scenarios}, Loss: {loss.item():.4f}")

    model.eval()
    if verbose:
        print(
            f"[GNN] 预训练完成：来源={os.path.basename(pretrain_tasks_file)}, "
            f"随机场景={num_scenarios}, 每场景任务数={min(task_sample_size, len(pretrain_tasks))}"
        )
    return model


def build_gnn_seed_vectors(
    gnn_model,
    satellites: Sequence[Satellite],
    tasks: Sequence[Task],
) -> List[np.ndarray]:
    if gnn_model is None or not tasks or not satellites:
        return []

    satellite_dict = {satellite.sat_id: satellite for satellite in satellites}
    data, sat_indices, task_indices = build_graph(satellite_dict, list(tasks))
    if data.edge_index.numel() == 0:
        return []
    data = normalize_graph_features(data)

    gnn_model.eval()
    with torch.no_grad():
        embeddings = gnn_model(data)

    task_scores = []
    density_scores = []
    energy_density_scores = []
    for task in tasks:
        task_embedding = embeddings[task_indices[task.task_id]]
        edge_scores = []
        for satellite in satellites:
            if satellite.status != "服务中" or task.task_type not in satellite.supported_task_types:
                continue
            sat_embedding = embeddings[sat_indices[satellite.sat_id]]
            logit = (sat_embedding * task_embedding).sum() / np.sqrt(embeddings.shape[1])
            edge_scores.append(float(torch.sigmoid(logit).item()))
        best_edge_score = max(edge_scores) if edge_scores else 0.0
        task_scores.append(best_edge_score * task.profit)
        density_scores.append(best_edge_score * task.profit / _task_duration(task))
        energy_density_scores.append(best_edge_score * task.profit / max(task.energy, 1e-9))

    def keys_from_scores(scores: Sequence[float]) -> np.ndarray:
        keys = np.zeros(len(scores), dtype=np.float64)
        ranked = sorted(range(len(scores)), key=lambda idx: (scores[idx], -idx), reverse=True)
        for rank, task_idx in enumerate(ranked):
            keys[task_idx] = 1.0 - rank / max(len(scores), 1)
        return keys

    return [
        keys_from_scores(task_scores),
        keys_from_scores(density_scores),
        keys_from_scores(energy_density_scores),
    ]


def _dominates_max(objective_a: Tuple[float, int, float], objective_b: Tuple[float, int, float]) -> bool:
    return (
        all(a >= b for a, b in zip(objective_a, objective_b))
        and any(a > b for a, b in zip(objective_a, objective_b))
    )


def _fast_non_dominated_sort_max(objectives: Sequence[Tuple[float, int, float]]) -> List[List[int]]:
    n_items = len(objectives)
    dominated_count = [0] * n_items
    dominates = [[] for _ in range(n_items)]
    fronts = [[]]

    for i in range(n_items):
        for j in range(i + 1, n_items):
            if _dominates_max(objectives[i], objectives[j]):
                dominates[i].append(j)
                dominated_count[j] += 1
            elif _dominates_max(objectives[j], objectives[i]):
                dominates[j].append(i)
                dominated_count[i] += 1
        if dominated_count[i] == 0:
            fronts[0].append(i)

    front_idx = 0
    while front_idx < len(fronts) and fronts[front_idx]:
        next_front = []
        for i in fronts[front_idx]:
            for j in dominates[i]:
                dominated_count[j] -= 1
                if dominated_count[j] == 0:
                    next_front.append(j)
        fronts.append(next_front)
        front_idx += 1

    return fronts[:-1]


def _crowding_distance_max(front: Sequence[int], objectives: Sequence[Tuple[float, int, float]]) -> List[float]:
    front_size = len(front)
    if front_size <= 2:
        return [float("inf")] * front_size

    distances = [0.0] * front_size
    for obj_idx in range(len(objectives[0])):
        ranked_positions = sorted(range(front_size), key=lambda pos: objectives[front[pos]][obj_idx])
        distances[ranked_positions[0]] = float("inf")
        distances[ranked_positions[-1]] = float("inf")
        min_value = objectives[front[ranked_positions[0]]][obj_idx]
        max_value = objectives[front[ranked_positions[-1]]][obj_idx]
        value_range = max_value - min_value
        if value_range == 0:
            continue
        for pos in range(1, front_size - 1):
            prev_value = objectives[front[ranked_positions[pos - 1]]][obj_idx]
            next_value = objectives[front[ranked_positions[pos + 1]]][obj_idx]
            distances[ranked_positions[pos]] += (next_value - prev_value) / value_range

    return distances


def _profit_first_survivors(
    profits: np.ndarray,
    task_counts: np.ndarray,
    remaining_energy_sums: np.ndarray,
    pop_size: int,
    elite_fraction: float = 0.35,
) -> List[int]:
    objectives = list(zip(
        profits.tolist(),
        task_counts.astype(int).tolist(),
        remaining_energy_sums.tolist(),
    ))
    elite_count = max(1, min(pop_size, int(pop_size * elite_fraction)))
    lexicographic_elites = sorted(
        range(len(objectives)),
        key=lambda idx: (objectives[idx][0], objectives[idx][1], objectives[idx][2]),
        reverse=True,
    )[:elite_count]

    survivors = []
    survivor_set = set()
    for idx in lexicographic_elites:
        survivors.append(idx)
        survivor_set.add(idx)

    for front in _fast_non_dominated_sort_max(objectives):
        remaining_front = [idx for idx in front if idx not in survivor_set]
        if not remaining_front:
            continue
        if len(survivors) + len(remaining_front) <= pop_size:
            survivors.extend(remaining_front)
            survivor_set.update(remaining_front)
            continue

        needed = pop_size - len(survivors)
        distances = _crowding_distance_max(remaining_front, objectives)
        ranked = sorted(
            zip(distances, remaining_front),
            key=lambda item: (
                item[0],
                objectives[item[1]][0],
                objectives[item[1]][1],
                objectives[item[1]][2],
            ),
            reverse=True,
        )
        chosen = [idx for _, idx in ranked[:needed]]
        survivors.extend(chosen)
        survivor_set.update(chosen)
        break

    if len(survivors) < pop_size:
        for idx in sorted(
            range(len(objectives)),
            key=lambda item_idx: (
                objectives[item_idx][0],
                objectives[item_idx][1],
                objectives[item_idx][2],
            ),
            reverse=True,
        ):
            if idx not in survivor_set:
                survivors.append(idx)
                if len(survivors) == pop_size:
                    break

    return survivors[:pop_size]


def run_profit_first_nsga_de(
    tasks: Sequence[Task],
    satellites: Sequence[Satellite],
    pop_size: int = 80,
    n_gen: int = 100,
    differential_weight: float = 0.6,
    crossover_rate: float = 0.85,
    seed: int = 42,
    gnn_model=None,
) -> Dict[str, object]:
    """Run a profit-first NSGA-DE search over random-key task orders."""
    if not tasks or not satellites:
        remaining_energy = sum(satellite.energy_available_Wh for satellite in satellites)
        return {
            "total_profit": 0.0,
            "total_task_number": 0,
            "energy_available_Wh_sum": float(remaining_energy),
            "accepted": [],
            "convergence": [],
            "params": {
                "pop_size": 0,
                "n_gen": 0,
                "F": differential_weight,
                "CR": crossover_rate,
                "seed": seed,
                "gnn_seed_vectors": 0,
            },
        }

    n_tasks = len(tasks)
    pop_size = max(8, int(pop_size))
    n_gen = max(0, int(n_gen))
    rng = np.random.default_rng(seed)

    gnn_seed_vectors = build_gnn_guided_key_vectors(gnn_model, tasks, satellites)
    population = _seed_random_key_population(tasks, pop_size, rng, gnn_seed_vectors)
    profits = np.zeros(pop_size, dtype=np.float64)
    task_counts = np.zeros(pop_size, dtype=np.int64)
    remaining_energy_sums = np.zeros(pop_size, dtype=np.float64)
    accepted_schedules: List[List[Tuple[int, Task]]] = [[] for _ in range(pop_size)]

    for idx in range(pop_size):
        (
            profits[idx],
            task_counts[idx],
            remaining_energy_sums[idx],
            accepted_schedules[idx],
        ) = _evaluate_random_keys(
            population[idx],
            tasks,
            satellites,
        )

    convergence = [float(np.max(profits))]

    for _ in range(n_gen):
        trials = np.empty_like(population)
        index_pool = np.arange(pop_size)

        for idx in range(pop_size):
            candidates = index_pool[index_pool != idx]
            if len(candidates) >= 3:
                r1, r2, r3 = rng.choice(candidates, 3, replace=False)
                mutant = population[r1] + differential_weight * (population[r2] - population[r3])
                mutant = np.clip(mutant, 0.0, 1.0)
            else:
                mutant = rng.random(n_tasks)

            crossover_mask = rng.random(n_tasks) < crossover_rate
            if not crossover_mask.any():
                crossover_mask[rng.integers(n_tasks)] = True
            trials[idx] = np.where(crossover_mask, mutant, population[idx])

        trial_profits = np.zeros(pop_size, dtype=np.float64)
        trial_task_counts = np.zeros(pop_size, dtype=np.int64)
        trial_remaining_energy_sums = np.zeros(pop_size, dtype=np.float64)
        trial_accepted: List[List[Tuple[int, Task]]] = [[] for _ in range(pop_size)]
        for idx in range(pop_size):
            (
                trial_profits[idx],
                trial_task_counts[idx],
                trial_remaining_energy_sums[idx],
                trial_accepted[idx],
            ) = _evaluate_random_keys(
                trials[idx],
                tasks,
                satellites,
            )

        combined_population = np.vstack([population, trials])
        combined_profits = np.concatenate([profits, trial_profits])
        combined_task_counts = np.concatenate([task_counts, trial_task_counts])
        combined_remaining_energy_sums = np.concatenate([
            remaining_energy_sums,
            trial_remaining_energy_sums,
        ])
        combined_accepted = accepted_schedules + trial_accepted

        survivors = _profit_first_survivors(
            combined_profits,
            combined_task_counts,
            combined_remaining_energy_sums,
            pop_size,
        )
        population = combined_population[survivors]
        profits = combined_profits[survivors]
        task_counts = combined_task_counts[survivors]
        remaining_energy_sums = combined_remaining_energy_sums[survivors]
        accepted_schedules = [combined_accepted[idx] for idx in survivors]
        convergence.append(float(np.max(profits)))

    best_idx = max(
        range(pop_size),
        key=lambda idx: (profits[idx], int(task_counts[idx]), remaining_energy_sums[idx]),
    )
    return {
        "total_profit": float(profits[best_idx]),
        "total_task_number": int(task_counts[best_idx]),
        "energy_available_Wh_sum": float(remaining_energy_sums[best_idx]),
        "accepted": accepted_schedules[best_idx],
        "convergence": convergence,
        "params": {
            "pop_size": pop_size,
            "n_gen": n_gen,
            "F": differential_weight,
            "CR": crossover_rate,
            "seed": seed,
            "gnn_seed_vectors": len(gnn_seed_vectors),
        },
    }

def gnn_based_sampling(problem, n_samples):
    X = np.zeros((n_samples, problem.n_var), dtype=int)
    for j, task in enumerate(problem.tasks):
        task_emb = problem.task_embeddings[task.task_id]
        scores = [torch.cosine_similarity(task_emb.unsqueeze(0), problem.sat_embeddings[sat.sat_id].unsqueeze(0)).item() for sat in problem.satellites]
        probs = np.exp(scores) / np.sum(np.exp(scores))
        for i in range(n_samples):
            X[i, j] = np.random.choice(len(problem.satellites), p=probs)
    return Population.new(X=X)

def multi_objective_schedule(
    tasks_file_path: str,
    all_satellites: Dict[str, Satellite],
    log_file,
    gnn_model=None,
    pop_size: int = 80,
    n_gen: int = 100,
    seed: int = 42,
):
    """Profit-first NSGA-DE scheduler.

    `gnn_model` is kept for backwards-compatible callers. The optimizer now
    follows the random-key NSGA-DE design from nsga_de.py and uses total profit
    as the dominant objective throughout selection and final application.
    """
    all_tasks = []
    with open(tasks_file_path, 'r', encoding=detect_csv_encoding(tasks_file_path)) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['status'] == "调度中":
                all_tasks.append(Task(task_id=row['task_id'], priority=int(row['priority']), start_time=int(row['start_time']), end_time=int(row['end_time']), profit=float(row['profit']), energy=float(row['energy']), task_type=row['task_type'], status=row['status']))

    if not all_tasks:
        log_file.write("没有状态为“调度中”的任务\n")
        log_file.write("--- 多目标调度结束 ---\n")
        return
    if not all_satellites:
        log_file.write("没有可用卫星\n")
        log_file.write("--- 多目标调度结束 ---\n")
        return

    baseline_satellites = copy.deepcopy(all_satellites)
    greedy_schedule_agent(tasks_file_path, baseline_satellites, io.StringIO())
    baseline_profit = sum(satellite.total_profit for satellite in baseline_satellites.values())
    baseline_task_count = sum(satellite.total_task_number for satellite in baseline_satellites.values())
    baseline_energy_sum = sum(satellite.energy_available_Wh for satellite in baseline_satellites.values())

    satellite_list = list(all_satellites.values())
    result = run_profit_first_nsga_de(
        all_tasks,
        satellite_list,
        pop_size=pop_size,
        n_gen=n_gen,
        seed=seed,
        gnn_model=gnn_model,
    )

    if result["total_profit"] <= baseline_profit:
        retry_result = run_profit_first_nsga_de(
            all_tasks,
            satellite_list,
            pop_size=max(pop_size, 120),
            n_gen=max(n_gen, 160),
            seed=seed + 1,
            gnn_model=gnn_model,
        )
        if (
            retry_result["total_profit"] > result["total_profit"]
            or (
                abs(retry_result["total_profit"] - result["total_profit"]) <= 1e-9
                and (
                    retry_result["total_task_number"] > result["total_task_number"]
                    or (
                        retry_result["total_task_number"] == result["total_task_number"]
                        and retry_result["energy_available_Wh_sum"] > result["energy_available_Wh_sum"]
                    )
                )
            )
        ):
            result = retry_result

    log_file.write(
        "NSGA-DE 收益优先调度: "
        f"total_profit={result['total_profit']:.2f}, "
        f"total_task_number={result['total_task_number']}, "
        f"energy_available_Wh_sum={result['energy_available_Wh_sum']:.2f}\n"
    )
    log_file.write(f"GNN 初始种子数量: {result['params'].get('gnn_seed_vectors', 0)}\n")
    log_file.write(
        "greedy_schedule_agent 基线: "
        f"total_profit={baseline_profit:.2f}, "
        f"total_task_number={baseline_task_count}, "
        f"energy_available_Wh_sum={baseline_energy_sum:.2f}\n"
    )
    improvement = result["total_profit"] - baseline_profit
    log_file.write(f"相对智能贪婪收益提升: {improvement:.2f}\n")
    if improvement <= 0:
        log_file.write("警告: 当前搜索未找到高于 greedy_schedule_agent 的方案\n")

    apply_decoded_schedule_to_satellites(satellite_list, result["accepted"], log_file)
    log_file.write("--- 多目标调度结束 ---\n")

# 7. --- 统计与输出 ---
def print_satellite_statistics(
    all_satellites: Dict[str, Satellite],
    schedule_name: str,
    log_file,
    elapsed_time_sec: float = None,
):
    console_output = f"\n--- {schedule_name} 统计信息 ---"
    log_file.write(console_output + "\n")
    print(console_output)
    total_profit_sum, total_task_number_sum, total_energy_available_sum = 0.0, 0, 0.0
    for sat_id, satellite in all_satellites.items():
        console_output_sat = (
            f"卫星 {sat_id}:\n"
            f"  总收益: {satellite.total_profit:.2f}\n"
            f"  总任务数: {satellite.total_task_number}\n"
            f"  剩余能量: {satellite.energy_available_Wh:.2f} Wh"
        )
        log_file.write(console_output_sat + "\n")
        print(console_output_sat)
        total_profit_sum += satellite.total_profit
        total_task_number_sum += satellite.total_task_number
        total_energy_available_sum += satellite.energy_available_Wh
    console_output_total = (
        f"\n所有卫星总计:\n"
        f"  总收益: {total_profit_sum:.2f}\n"
        f"  总任务数: {total_task_number_sum}\n"
        f"  剩余总能量: {total_energy_available_sum:.2f} Wh\n"
        f"  所需时间: {elapsed_time_sec if elapsed_time_sec is not None else 0.0:.4f} 秒"
    )
    log_file.write(console_output_total + "\n")
    print(console_output_total)
    console_output_end = f"--- {schedule_name} 统计信息结束 ---\n"
    log_file.write(console_output_end + "\n")
    print(console_output_end)


def calculate_energy_available_sum(schedule_method) -> float:
    """Return the total remaining energy for one scheduling method."""
    if isinstance(schedule_method, dict):
        energy_values = schedule_method.get("energy_available_Wh", [])
    else:
        energy_values = getattr(schedule_method, "energy_available_Wh", [])

    if energy_values is None:
        return 0.0
    if isinstance(energy_values, (int, float, np.number)):
        return float(energy_values)

    total_energy = 0.0
    for value in energy_values:
        if value is None:
            continue
        total_energy += float(value)
    return total_energy


def sort_schedule_methods(schedule_methods: Sequence) -> List:
    """Sort methods by profit, task count, then remaining energy, all descending."""
    def sort_key(method):
        if isinstance(method, dict):
            total_profit = method.get("total_profit", 0.0)
            total_task_number = method.get("total_task_number", method.get("total_tasks", 0))
        else:
            total_profit = getattr(method, "total_profit", 0.0)
            total_task_number = getattr(method, "total_task_number", getattr(method, "total_tasks", 0))
        return (
            float(total_profit),
            float(total_task_number),
            calculate_energy_available_sum(method),
        )

    return sorted(schedule_methods, key=sort_key, reverse=True)


def select_best_schedule_method(schedule_methods: Sequence):
    """Return the best method by profit, then task count, then energy sum."""
    sorted_methods = sort_schedule_methods(schedule_methods)
    if not sorted_methods:
        return None
    return sorted_methods[0]


def build_schedule_method_result(
    method_id: str,
    satellites: Dict[str, Satellite],
    elapsed_time_sec: float = 0.0,
) -> Dict[str, object]:
    return {
        "method_id": method_id,
        "total_profit": sum(satellite.total_profit for satellite in satellites.values()),
        "total_task_number": sum(satellite.total_task_number for satellite in satellites.values()),
        "elapsed_time_sec": float(elapsed_time_sec),
        "energy_available_Wh": [
            satellite.energy_available_Wh
            for satellite in satellites.values()
        ],
    }


def format_schedule_methods_summary(schedule_methods: Sequence) -> str:
    lines = ["--- 所有调度方法总览 ---"]
    for method in schedule_methods:
        if isinstance(method, dict):
            method_id = method.get("method_id", "")
            total_profit = method.get("total_profit", 0.0)
            total_task_number = method.get("total_task_number", method.get("total_tasks", 0))
            elapsed_time_sec = method.get("elapsed_time_sec", 0.0)
        else:
            method_id = getattr(method, "method_id", "")
            total_profit = getattr(method, "total_profit", 0.0)
            total_task_number = getattr(method, "total_task_number", getattr(method, "total_tasks", 0))
            elapsed_time_sec = getattr(method, "elapsed_time_sec", 0.0)

        lines.append(
            f"方法: {method_id}\n"
            f"  总收益: {float(total_profit):.2f}\n"
            f"  总任务数: {float(total_task_number):g}\n"
            f"  剩余总能量: {calculate_energy_available_sum(method):.2f} Wh\n"
            f"  所需时间: {float(elapsed_time_sec):.4f} 秒"
        )
    return "\n".join(lines)

# 8. --- 主执行入口 ---
if __name__ == "__main__":
    base_path = "C:\\Users\\asus\\Desktop\\求真杯"
    output_dir = os.path.join(base_path, "output_logs")
    os.makedirs(output_dir, exist_ok=True)
    satellites_file = os.path.join(base_path, "satellites_20.csv")
    satellite_sample_file = os.path.join(base_path, "satellite_sample_143.csv")
    test_satellite_sample_file = os.path.join(base_path, "satellite_sample_87.csv")
    pretrain_task_files = [
        os.path.join(base_path, "tasks_1000.csv"),
        os.path.join(base_path, "task_sample_5000.csv"),
        os.path.join(base_path, "tasks_500.csv"),
    ]
    pretrain_satellite_files = [
        satellite_sample_file,
        satellites_file,
    ]
    tasks_input_file = os.path.join(base_path, "task_sample_3000.csv")
    test_satellites_file = test_satellite_sample_file

    all_task_types_from_input = get_all_unique_task_types_from_files(pretrain_task_files + [tasks_input_file])
    initial_satellites = load_satellites(test_satellites_file, all_task_types_from_input)

    gnn_pretrain_log_path = os.path.join(output_dir, "gnn_pretrain_log.txt")
    gnn_cache_path = os.path.join(output_dir, "gnn_pretrained_asgat.pt")
    force_retrain = False
    print("加载或预训练 GNN: tasks_1000.csv, task_sample_5000.csv, tasks_500.csv, satellite_sample_143.csv, satellites_20.csv")
    with open(gnn_pretrain_log_path, 'w', encoding='utf-8') as gnn_log_f:
        trained_asgat_model = load_or_pretrain_gnn_for_nsga_de(
            gnn_cache_path,
            pretrain_task_files,
            pretrain_satellite_files,
            all_task_types_from_input,
            sample_count=24,
            tasks_per_sample=80,
            satellites_per_sample=12,
            epochs_per_sample=1,
            seed=42,
            force_retrain=force_retrain,
            log_file=gnn_log_f,
        )
    print(f"GNN 缓存文件: {gnn_cache_path}")
    print(f"GNN 日志已写入: {gnn_pretrain_log_path}")
    print("测试调度数据: task_sample_3000.csv + satellite_sample_87.csv")

    results = []
    schedulers = {
        "plain_schedule": plain_schedule,
        "greedy_schedule": greedy_schedule,
        "greedy_schedule_agent": greedy_schedule_agent,
    }

    for name, func in schedulers.items():
        satellites_copy = copy.deepcopy(initial_satellites)
        log_path = os.path.join(output_dir, f"{name}_log.txt")
        with open(log_path, 'w', encoding='utf-8') as log_f:
            schedule_start_time = time.perf_counter()
            func(tasks_input_file, satellites_copy, log_f)
            elapsed_time_sec = time.perf_counter() - schedule_start_time
            print_satellite_statistics(
                satellites_copy,
                name.replace('_', ' ').title(),
                log_f,
                elapsed_time_sec,
            )
        print(f"{name.replace('_', ' ').title()} 调度结果已写入: {log_path}")
        results.append(build_schedule_method_result(name, satellites_copy, elapsed_time_sec))
    
    # 多目标调度
    multi_obj_satellites = copy.deepcopy(initial_satellites)
    multi_obj_log_file_path = os.path.join(output_dir, "multi_objective_schedule_log.txt")
    with open(multi_obj_log_file_path, 'w', encoding='utf-8') as log_f:
        schedule_start_time = time.perf_counter()
        multi_objective_schedule(
            tasks_input_file,
            multi_obj_satellites,
            log_f,
            trained_asgat_model,
            pop_size=24,
            n_gen=12,
        )
        elapsed_time_sec = time.perf_counter() - schedule_start_time
        print_satellite_statistics(
            multi_obj_satellites,
            "Multi-Objective Schedule",
            log_f,
            elapsed_time_sec,
        )
    print(f"Multi-Objective Schedule 调度结果已写入: {multi_obj_log_file_path}")
    results.append(build_schedule_method_result(
        "multi_objective_schedule",
        multi_obj_satellites,
        elapsed_time_sec,
    ))

    # --- 最终总结 ---
    summary_log_file_path = os.path.join(output_dir, "summary_log.txt")
    with open(summary_log_file_path, 'w', encoding='utf-8') as summary_log_f:
        summary_text = format_schedule_methods_summary(results)
        summary_log_f.write(summary_text + "\n")
        print(summary_text)
        summary_footer = "------------------------"
        summary_log_f.write(summary_footer + "\n")
        print(summary_footer)
    print(f"最终总结已写入: {summary_log_file_path}")
