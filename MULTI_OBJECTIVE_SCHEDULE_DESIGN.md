# Multi-Objective Schedule Design

`multi_objective_schedule` now uses a profit-first GNN-guided NSGA-DE random-key
optimizer inspired by `nsga_de.py`.

The final priority is lexicographic:

1. Maximize `total_profit`.
2. If `total_profit` is equal, maximize `total_task_number`.
3. If both are equal, maximize the sum of remaining `energy_available_Wh`.

The optimizer encodes each individual as a vector of random keys. Sorting those
keys gives a task order, and `decode_task_order` greedily inserts tasks into
satellites using a best-fit rule. The decoder checks status, remaining windows,
task type support, energy, and interval conflicts, so every decoded schedule is
feasible.

The GNN is pretrained before testing:

- pretraining task data: `tasks_1000.csv`, `task_sample_5000.csv`,
  `tasks_500.csv`
- pretraining satellite data: `satellite_sample_143.csv`, `satellites_20.csv`
- test data: `task_sample_3000.csv + satellite_sample_87.csv`
- pretraining samples random task/satellite subsets from the above pools
- pseudo labels come from the best deterministic decoder seed in each sampled
  sub-scenario
- after pretraining, the GNN scores task-satellite edges on
  `task_sample_3000.csv + satellite_sample_87.csv` and contributes four
  random-key seed vectors to the NSGA-DE population
- pretrained weights are cached at `output_logs/gnn_pretrained_asgat.pt`; a
  matching cache is loaded directly, otherwise the GNN is retrained and saved
  again. Set `force_retrain = True` in the main function or delete the cache to
  force a fresh pretrain run.

The initial population includes strong deterministic seeds:

- priority/profit order
- profit order
- profit per duration
- profit per energy
- profit per energy-duration density
- priority plus profit-density
- short-task plus profit order
- raw GNN edge score
- GNN score times profit
- GNN score times profit-duration density
- GNN score plus priority/profit hybrid score

The DE loop uses `DE/rand/1/bin` mutation and crossover. Survival keeps a
profit-first elite subset, then fills the rest with non-dominated sorting and
crowding distance on `(total_profit, total_task_number, energy_available_Wh_sum)`.
This keeps diversity without allowing high task count or high residual energy
but low profit candidates to displace the best profit solutions.

Before optimization, `multi_objective_schedule` computes the
`greedy_schedule_agent` baseline on a copy of the satellites. The selected
GNN-guided NSGA-DE schedule is logged with its improvement over that baseline.
Each method report prints total profit, total task count, remaining total
energy, and elapsed scheduling time. The final overview preserves the main
function run order: `plain_schedule`, `greedy_schedule`,
`greedy_schedule_agent`, then `multi_objective_schedule`.
With the requested multi-file GNN pretraining pool and
`task_sample_3000.csv + satellite_sample_87.csv` for testing, the verified
result is:

- `greedy_schedule_agent`: `total_profit = 41197.30`,
  `total_task_number = 241`, `energy_available_Wh_sum = 42886.04`
- GNN-guided NSGA-DE: `total_profit = 54090.97`,
  `total_task_number = 372`, `energy_available_Wh_sum = 42819.64`
- improvement: `+12893.67`
