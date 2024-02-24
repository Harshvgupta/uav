import torch
import matplotlib.pyplot as plt
from uavtraj import UAVTrajectoryPlanner


def demo_minimum_snap_simple():
    waypoints = torch.tensor([[0, 0], [1, 2], [2, -1], [4, 8], [5, 2]], dtype=torch.float64).t()
    start_vel, start_acc, end_vel, end_acc = torch.tensor([0, 0]), torch.tensor([0, 0]), torch.tensor([0, 0]), torch.tensor([0, 0])
    total_time = 5.0
    poly_order = 5

    planner = UAVTrajectoryPlanner(total_time, poly_order, start_vel, start_acc, end_vel, end_acc, dtype=torch.float64, device='cpu')
    polys_x, polys_y, time_stamps = planner(waypoints)
    print(polys_x,polys_y)

    planner.plot_trajectory(waypoints, polys_x, polys_y, time_stamps)

demo_minimum_snap_simple()