import cvxpy as cp
import numpy as np

import cvxpy as cp
import numpy as np

def generate_trajectory(keyframes, order=7):
    num_keyframes = len(keyframes)
    num_segments = num_keyframes - 1
    num_coefficients = order + 1

    # Flatten the coefficient array for cvxpy
    coefficients = cp.Variable((num_segments * num_coefficients, 3))

    # Cost function (sum of squared snap)
    cost = 0  # We will define this later

    # Constraints
    constraints = []

    # Define the time intervals for each segment
    times = [kf[1] for kf in keyframes]
    durations = [times[i+1] - times[i] for i in range(num_segments)]

    # Cost function for snap: Minimize the integral of the snap squared over each segment
    for s in range(num_segments):
        for d in range(3):  # For x, y, z
            # Calculate the coefficients for the snap cost function
            for k in range(4, num_coefficients):
                cost += (
                    cp.square(coefficients[s * num_coefficients + k, d]) *
                    np.math.factorial(k) ** 2 / np.math.factorial(2 * k - 4) *
                    durations[s] ** (2 * k - 4)
                )

    # Add constraints for each keyframe position
    for k in range(num_keyframes):
        for d in range(3):  # For x, y, z
            expr = 0
            for s in range(num_segments):
                if times[k] < times[s+1]:
                    break
                t = times[k] - times[s]
                expr += sum(
                    coefficients[s * num_coefficients + i, d] * t**i
                    for i in range(num_coefficients)
                )
            constraints.append(expr == keyframes[k][0][d])

    # Continuity constraints between segments
    for s in range(1, num_segments):
        for d in range(3):  # For x, y, z
            for i in range(1, order):
                expr1 = sum(
                    coefficients[(s-1) * num_coefficients + j, d] *
                    np.math.factorial(j) / np.math.factorial(j - i) *
                    durations[s-1]**(j - i)
                    for j in range(i, num_coefficients)
                )
                expr2 = coefficients[s * num_coefficients + i, d]
                constraints.append(expr1 == expr2)

    # Solve the optimization problem
    problem = cp.Problem(cp.Minimize(cost), constraints)
    problem.solve(solver=cp.ECOS)

    # Reshape the coefficients into the original shape for plotting
    trajectory = coefficients.value
    return trajectory.reshape((num_segments, num_coefficients, 3))

# Define keyframes and total time
keyframes = [([0, 0, 0], 0), ([1, 1, 1], 1), ([2, 0, 2], 2), ([3, 1, 3], 3), ([4, 0, 4], 4), ([5, 1, 5], 5), ([6, 0, 6], 6)]

# Generate the trajectory
coefficients = generate_trajectory(keyframes)



import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def plot_trajectory(coefficients, keyframes, total_time, resolution=500):
    """
    Plot the trajectory in 3D space.

    :param coefficients: Coefficients of the polynomial trajectory.
    :param keyframes: List of keyframes (each keyframe is a tuple of position and time).
    :param total_time: Total time to complete the trajectory.
    :param resolution: Number of points to plot along the trajectory.
    """
    t = np.linspace(0, total_time, resolution)
    trajectory = np.zeros((resolution, 3))  # x, y, z

    for i in range(len(keyframes) - 1):
        start_time = keyframes[i][1]
        end_time = keyframes[i+1][1]
        segment_indices = (t >= start_time) & (t <= end_time)
        t_segment = t[segment_indices] - start_time

        for d in range(3):  # x, y, z
            coefs = coefficients[i, d, :]
            trajectory[segment_indices, d] = np.polyval(coefs[::-1], t_segment)

    # Extract keyframe positions
    keyframe_positions = np.array([frame[0] for frame in keyframes])

    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], label='Trajectory')
    ax.scatter(keyframe_positions[:, 0], keyframe_positions[:, 1], keyframe_positions[:, 2], color='red', label='Keyframes')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    plt.title('3D Trajectory with Keyframes')
    plt.legend()
    plt.show()

# Use the plot_trajectory function with the trajectory and keyframes
# Example: plot_trajectory(trajectory, keyframes, total_time=2)
# Assuming trajectory and keyframes are obtained from the generate_trajectory function
plot_trajectory(coefficients, keyframes, total_time=2)
