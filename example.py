import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def test_trajectory(keyframes, times, polynomial_order, num_segments):
    # Use linear interpolation for testing
    num_variables_per_segment = polynomial_order * 4  # For x, y, z, ψ
    coefs = np.zeros(num_segments * num_variables_per_segment)

    for i in range(num_segments):
        start_point = keyframes[i]
        end_point = keyframes[i + 1]
        duration = times[i + 1] - times[i]

        # Linear coefficients for x, y, z
        for j in range(3):
            coefs[i * num_variables_per_segment + j * polynomial_order] = start_point[j]
            coefs[i * num_variables_per_segment + j * polynomial_order + 1] = (end_point[j] - start_point[j]) / duration

    return coefs

def plot_trajectory(keyframes, times, coefs, polynomial_order, num_segments):
    num_variables_per_segment = polynomial_order * 4
    t_total = np.linspace(times[0], times[-1], 500)
    trajectory = np.zeros((len(t_total), 3))

    for i, t in enumerate(t_total):
        segment_index = np.searchsorted(times, t, side='right') - 1
        segment_index = min(segment_index, num_segments - 1)
        t_segment = t - times[segment_index]
        segment_coefs = coefs[segment_index * num_variables_per_segment:(segment_index + 1) * num_variables_per_segment]

        # Extract and evaluate the polynomial for each segment
        trajectory[i, 0] = np.polyval(segment_coefs[0:polynomial_order][::-1], t_segment)  # x
        trajectory[i, 1] = np.polyval(segment_coefs[polynomial_order:2*polynomial_order][::-1], t_segment)  # y
        trajectory[i, 2] = np.polyval(segment_coefs[2*polynomial_order:3*polynomial_order][::-1], t_segment)  # z

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], label='Trajectory')
    keyframes = np.array(keyframes)
    ax.scatter(keyframes[:, 0], keyframes[:, 1], keyframes[:, 2], color='red', label='Keyframes')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    plt.title('3D Trajectory with Keyframes')
    plt.legend()
    plt.show()

# Define keyframes, times, and other parameters for testing
keyframes = [[0, 0, 0], [1, 1, 1], [2, 0, 2]]  # Example keyframes
times = [0, 1, 5]  # Example times
polynomial_order = 7  # Example polynomial order

# Generate a test trajectory with linear interpolation
coefs = test_trajectory(keyframes, times, polynomial_order, len(keyframes) - 1)

# Plot the trajectory
plot_trajectory(keyframes, times, coefs, polynomial_order, len(keyframes) - 1)
