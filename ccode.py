import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from qpsolvers import solve_qp



def generate_trajectory_qp(keyframes, times, polynomial_order, mu_r, mu_psi):
    num_segments = len(keyframes) - 1
    num_variables_per_segment = polynomial_order * 4  # x, y, z, ψ
    total_variables = num_variables_per_segment * num_segments

    # Hessian matrix H for the quadratic cost
    H = np.zeros((total_variables, total_variables))
    for i in range(num_segments):
        # Snap (4th derivative) for x, y, z
        for j in range(3):
            snap_indices = i * num_variables_per_segment + j * polynomial_order + np.arange(4, polynomial_order)
            H[np.ix_(snap_indices, snap_indices)] = mu_r

        # Yaw acceleration (2nd derivative)
        yaw_indices = i * num_variables_per_segment + 3 * polynomial_order + np.arange(2, polynomial_order)
        H[np.ix_(yaw_indices, yaw_indices)] = mu_psi

    # Linear cost vector f
    f = np.zeros(total_variables)

    # Constraints matrices A and b
    num_keyframe_constraints = len(keyframes) * 4
    num_continuity_constraints = (num_segments - 1) * 4 * 2  # For each segment: velocity and acceleration
    num_constraints = num_keyframe_constraints + num_continuity_constraints
    A = np.zeros((num_constraints, total_variables))
    b = np.zeros(num_constraints)

    # Position constraints at keyframes
    for i in range(num_segments):
        t = times[i+1] - times[i]
        for j in range(4):  # x, y, z, ψ
            A[i * 4 + j, i * num_variables_per_segment + j * polynomial_order : i * num_variables_per_segment + j * polynomial_order + polynomial_order] = np.array([t**k for k in range(polynomial_order)])
            b[i * 4 + j] = keyframes[i+1][j]

    # Continuity constraints between segments
    constraint_idx = num_keyframe_constraints
    for i in range(1, num_segments):
        for j in range(4):  # x, y, z, ψ
            # Ensure indices are within bounds
            if constraint_idx < num_constraints:
                # Velocity continuity
                A[constraint_idx, (i - 1) * num_variables_per_segment + j * polynomial_order + 1] = 1  # End of segment i-1
                A[constraint_idx, i * num_variables_per_segment + j * polynomial_order + 1] = -1  # Start of segment i
                constraint_idx += 1

                # Acceleration continuity
                A[constraint_idx, (i - 1) * num_variables_per_segment + j * polynomial_order + 2] = 2  # End of segment i-1
                A[constraint_idx, i * num_variables_per_segment + j * polynomial_order + 2] = -2  # Start of segment i
                constraint_idx += 1

    epsilon = 1e-8  # Small positive value
    np.fill_diagonal(H, np.diag(H) + epsilon)
    # Solve QP problem
    c = solve_qp(H, f, A, b, solver="cvxopt")

    if c is not None:
        return c
    else:
        raise ValueError("QP optimization failed")


def plot_trajectory_qp(keyframes, times, c, polynomial_order, num_segments):
    # Similar plotting function as before
    # ...
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


# Example usage
# keyframes = [[0, 0, 0, 0],[1,1,0,0],[2,2,6,0],[3,3,4,0], [4 , 4 , 5 , 0], [5, 5, 5, 0], [6, 6, 6, 0]]  # Example keyframes with yaw angles
keyframes = [
    [0, 0, 0, 0],    # Starting position at time 0
    [1, 1, 1, 1],    # First keyframe at time 1
    [2, 0, 2, 2],    # Second keyframe at time 2
    [3, 1, 3, 3],    # Third keyframe at time 3
    [4, 0, 4, 4],    # Fourth keyframe at time 4
    [5, 1, 5, 5],    # Fifth keyframe at time 5
    [6, 0, 6, 6]     # Final position at time 6
]
times = [0, 1, 2,3,4,5,6]  # Times corresponding to each keyframe
polynomial_order = 5  # Quintic polynomials
mu_r = 1  # Weight for snap
mu_psi = 1  # Weight for yaw acceleration

coefs = generate_trajectory_qp(keyframes, times, polynomial_order, mu_r, mu_psi)
plot_trajectory_qp(keyframes, times, coefs, polynomial_order, len(keyframes) - 1)
