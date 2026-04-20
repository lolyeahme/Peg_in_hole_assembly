# Minimal Assembly Simulation & Force Control Repo

这个子仓库模仿 `Peg_in_hole_assembly` 的仿真流程与参数风格，但刻意去掉 RL/DDPG 等算法层内容，仅保留：

1. 装配几何与位姿更新（双 peg-in-hole）
2. 接触力/力矩估计（6 维 wrench）
3. 参考力跟踪的规则力控（增量位姿动作）

## 参数设定（对齐原仓库风格）

- `diameter_hole = 30.0`
- `diameter_peg = 29.9`
- `distance_axes = 200.0`
- `height = 100.0`
- `K = 100.0`（接触刚度）
- `u = 0.05`（摩擦系数）
- `ref_force = [0, 0, -70, 0, 0, 0]`
- `Kpz = 0.015, Krxyz = 0.01, Kpxy = 0.0022`

## 目录结构

- `src/simulation.py`：几何建模、位姿变换、接触力计算
- `src/force_control.py`：规则力控（参考力误差 -> 位姿增量）
- `src/run_demo.py`：主循环示例

## 运行

```bash
cd /home/runner/work/Peg_in_hole_assembly/Peg_in_hole_assembly/minimal_assembly_sim_repo
python src/run_demo.py
```

输出将打印每步 action、force、pose，并在达到插入力阈值或深度阈值时结束。
