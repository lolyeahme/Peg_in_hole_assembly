import numpy as np
from simulation import DualPegInHoleSimulation
from force_control import SimpleForceController


def main():
    sim = DualPegInHoleSimulation(init_transform=np.array([0.05, -0.08, -0.08, 0.01, 0.007, 3.0]))
    ctrl = SimpleForceController()

    max_steps = 80
    for step in range(max_steps):
        action = ctrl.expert_action(sim.force)
        force, pose = sim.apply_transform(action)
        print(f"step={step:02d} action={np.round(action,4)} force={np.round(force,3)} z={pose[2]:.3f}")

        if pose[2] < 100.0:
            print("Finish: reached insertion depth threshold.")
            break
        if force[2] <= -99.0:
            print("Stop: reached Z-force clamp threshold.")
            break


if __name__ == "__main__":
    np.random.seed(0)
    main()
