# # import numpy as np
# # from qpsolvers import solve_qp

# # M = np.array([[1.0, 2.0, 0.0], [-8.0, 3.0, 2.0], [0.0, 1.0, 1.0]])
# # P = M.T @ M  # this is a positive definite matrix
# # q = np.array([3.0, 2.0, 3.0]) @ M
# # G = np.array([[1.0, 2.0, 1.0], [2.0, 0.0, 1.0], [-1.0, 2.0, -1.0]])
# # h = np.array([3.0, 2.0, -2.0])
# # A = np.array([1.0, 1.0, 1.0])
# # b = np.array([1.0])

# # x = solve_qp(P, q, G, h, A, b, solver="proxqp")
# # print(f"QP solution: x = {x}")

# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from qpsolvers import solve_qp

# # Trajectory generation functions (as defined previously)
# def qp_xyz(x0, xd0, xdd0, xf, xdf, t_f):
#     H = np.array([
#         [0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 2 * 24**2 * t_f, 24 * 120 * t_f**2],
#         [0, 0, 0, 0, 24 * 120 * t_f**2, 2 * (1/3) * 120**2 * t_f**3]
#     ])
#     f = np.zeros(6)
#     Aeq = np.array([
#         [1, 0, 0, 0, 0, 0],
#         [0, 1, 0, 0, 0, 0],
#         [0, 0, 2, 0, 0, 0],
#         [1, t_f, t_f**2, t_f**3, t_f**4, t_f**5],
#         [0, 1, 2*t_f, 3*t_f**2, 4*t_f**3, 5*t_f**4]
#     ])
#     beq = np.array([x0, xd0, xdd0, xf, xdf])
#     return solve_qp(H, f, None, None, Aeq, beq,solver="proxqp")
#     # return solve_qp(H, f, None, None, Aeq, beq)

# def qp_yaw(yaw0, yawd0, yawf, yawdf, t_f):
#     H = np.array([
#         [0, 0, 0, 0],
#         [0, 0, 0, 0],
#         [0, 0, 2 * 4 * t_f, 12 * t_f**2],
#         [0, 0, 12 * t_f**2, 2 * 12 * t_f**3]
#     ])
#     f = np.zeros(4)
#     Aeq = np.array([
#         [1, 0, 0, 0],
#         [0, 1, 0, 0],
#         [1, t_f, t_f**2, t_f**3],
#         [0, 1, 2*t_f, 3*t_f**2]
#     ])
#     beq = np.array([yaw0, yawd0, yawf, yawdf])
#     return solve_qp(H, f, None, None, Aeq, beq,solver="proxqp")
#     # return solve_qp(H, f, None, None, Aeq, beq)

# # Example initial conditions and final conditions
# x0, xd0, xdd0, xf, xdf = 0, 0, 0, 2, 0
# y0, yd0, ydd0, yf, ydf = 0, 0, 0, 2, 0
# z0, zd0, zdd0, zf, zdf = 0, 0, 0, 2, 0
# yaw0, yawd0, yawf, yawdf = 0, 0, 0, 0
# t_f = 2  # Example total time

# # Calculate coefficients
# coefs_x = qp_xyz(x0, xd0, xdd0, xf, xdf, t_f)
# coefs_y = qp_xyz(y0, yd0, ydd0, yf, ydf, t_f)
# coefs_z = qp_xyz(z0, zd0, zdd0, zf, zdf, t_f)
# coefs_yaw = qp_yaw(yaw0, yawd0, yawf, yawdf, t_f)

# # Simulate and plot
# t = np.linspace(0, t_f, 100)
# x_traj = np.polyval(coefs_x, t)
# y_traj = np.polyval(coefs_y, t)
# z_traj = np.polyval(coefs_z, t)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot(x_traj, y_traj, z_traj, label='Trajectory')
# ax.scatter([x0, xf], [y0, yf], [z0, zf], color='red', label='Start/End Points')
# ax.set_xlabel('X axis')
# ax.set_ylabel('Y axis')
# ax.set_zlabel('Z axis')
# plt.title('3D Trajectory')
# plt.legend()
# plt.show()


# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.optimize import minimize
# from mpl_toolkits.mplot3d import Axes3D

# def generate_trajectory(keyframes, times, polynomial_order, mu_r, mu_psi):
#     num_segments = len(keyframes) - 1
#     num_variables_per_segment = polynomial_order * 4  # For x, y, z, ψ

#     # Define the objective function for snap minimization
#     def objective_function(c):
#         cost = 0
#         for i in range(num_segments):
#             segment_coefs = c[i * num_variables_per_segment:(i + 1) * num_variables_per_segment]
#             snap = segment_coefs[-4:]  # Last 4 coefficients of position polynomials (snap)
#             yaw_accel = segment_coefs[-1]  # Last coefficient of yaw polynomial (yaw acceleration)
#             cost += mu_r * np.sum(snap**2) + mu_psi * yaw_accel**2
#         return cost

#     # Define constraints for passing through keyframes
#     def constraints_function(c):
#         constraints = []
#         for i in range(num_segments):
#             segment_coefs = c[i * num_variables_per_segment:(i + 1) * num_variables_per_segment]
#             # Position constraints at the start and end of each segment
#             for j in range(3):  # x, y, z
#                 start_pos = np.polyval(segment_coefs[j * polynomial_order:(j + 1) * polynomial_order][::-1], 0)
#                 end_pos = np.polyval(segment_coefs[j * polynomial_order:(j + 1) * polynomial_order][::-1], times[i + 1] - times[i])
#                 constraints.extend([start_pos - keyframes[i][j], end_pos - keyframes[i + 1][j]])
#             # Yaw angle constraints (simplified)
#             start_yaw = segment_coefs[3 * polynomial_order]
#             end_yaw = np.polyval(segment_coefs[3 * polynomial_order:4 * polynomial_order][::-1], times[i + 1] - times[i])
#             constraints.extend([start_yaw - keyframes[i][3], end_yaw - keyframes[i + 1][3]])
#         return np.array(constraints)

#     # Initial guess (linear interpolation between keyframes)
#     initial_guess = np.zeros(num_segments * num_variables_per_segment)

#     # Solve QP problem using optimization
#     result = minimize(objective_function, initial_guess, constraints={'type': 'eq', 'fun': constraints_function})

#     if result.success:
#         coefs = result.x
#     else:
#         raise ValueError("Optimization failed")

#     return coefs

# def plot_trajectory(keyframes, times, coefs, polynomial_order, num_segments):
#     num_variables_per_segment = polynomial_order * 4
#     t_total = np.linspace(times[0], times[-1], 500)
#     trajectory = np.zeros((len(t_total), 3))

#     for i, t in enumerate(t_total):
#         segment_index = np.searchsorted(times, t, side='right') - 1
#         segment_index = min(segment_index, num_segments - 1)
#         t_segment = t - times[segment_index]
#         segment_coefs = coefs[segment_index * num_variables_per_segment:(segment_index + 1) * num_variables_per_segment]

#         # Extract and evaluate the polynomial for each segment
#         trajectory[i, 0] = np.polyval(segment_coefs[0:polynomial_order][::-1], t_segment)  # x
#         trajectory[i, 1] = np.polyval(segment_coefs[polynomial_order:2*polynomial_order][::-1], t_segment)  # y
#         trajectory[i, 2] = np.polyval(segment_coefs[2*polynomial_order:3*polynomial_order][::-1], t_segment)  # z

#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], label='Trajectory')
#     keyframes = np.array(keyframes)
#     ax.scatter(keyframes[:, 0], keyframes[:, 1], keyframes[:, 2], color='red', label='Keyframes')
#     ax.set_xlabel('X axis')
#     ax.set_ylabel('Y axis')
#     ax.set_zlabel('Z axis')
#     plt.title('3D Trajectory with Keyframes')
#     plt.legend()
#     plt.show()

# # Define keyframes, times, and other parameters
# keyframes = [[0, 0, 0, 0], [1, 1, 1, np.pi/2], [2, 0, 2, np.pi]]  # Example keyframes with yaw angles
# times = [0, 1, 2]  # Example times
# polynomial_order = 5  # Using quintic polynomials for each segment
# mu_r = 1  # Weight for snap
# mu_psi = 1  # Weight for yaw angle derivative

# # Generate trajectory
# coefs = generate_trajectory(keyframes, times, polynomial_order, mu_r, mu_psi)

# # Plot the trajectory
# plot_trajectory(keyframes, times, coefs, polynomial_order, len(keyframes) - 1)


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D

def generate_trajectory(keyframes, times, polynomial_order, mu_r, mu_psi):
    num_segments = len(keyframes) - 1
    num_variables_per_segment = polynomial_order * 4  # For x, y, z, ψ

    # Define the objective function for snap minimization
    def objective_function(c):
        cost = 0
        for i in range(num_segments):
            segment_coefs = c[i * num_variables_per_segment:(i + 1) * num_variables_per_segment]
            # Snap (4th derivative of position) coefficients
            snap = segment_coefs[3*polynomial_order-4:3*polynomial_order]
            # Yaw acceleration (2nd derivative) coefficient
            yaw_accel = segment_coefs[-2]
            cost += mu_r * np.sum(snap**2) + mu_psi * yaw_accel**2
        return cost

    # Define constraints for passing through keyframes
    def constraints_function(c):
        constraints = []
        for i in range(num_segments):
            segment_coefs = c[i * num_variables_per_segment:(i + 1) * num_variables_per_segment]
            # Position constraints at the start and end of each segment
            for j in range(3):  # x, y, z
                start_pos = np.polyval(segment_coefs[j * polynomial_order:(j + 1) * polynomial_order][::-1], 0)
                end_pos = np.polyval(segment_coefs[j * polynomial_order:(j + 1) * polynomial_order][::-1], times[i + 1] - times[i])
                constraints.extend([start_pos - keyframes[i][j], end_pos - keyframes[i + 1][j]])
            # Yaw angle constraints
            start_yaw = segment_coefs[3 * polynomial_order]
            end_yaw = np.polyval(segment_coefs[3 * polynomial_order:4 * polynomial_order][::-1], times[i + 1] - times[i])
            constraints.extend([start_yaw - keyframes[i][3], end_yaw - keyframes[i + 1][3]])
        return np.array(constraints)

    # Initial guess (linear interpolation between keyframes)
    initial_guess = np.zeros(num_segments * num_variables_per_segment)

    # Solve QP problem using optimization
    result = minimize(objective_function, initial_guess, constraints={'type': 'eq', 'fun': constraints_function})

    if result.success:
        coefs = result.x
    else:
        raise ValueError("Optimization failed")

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

# Define keyframes, times, and other parameters
keyframes = [[0, 0, 0, 0], [1 , 10 , 2 , np.pi/4], [4, 2, 8, np.pi/2], [5, 0, 2, np.pi]]  # Example keyframes with yaw angles
times = [0, 1, 2,3]  # Example times
polynomial_order = 5  # Using quintic polynomials for each segment
mu_r = 1  # Weight for snap
mu_psi = 1  # Weight for yaw angle derivative

# Generate trajectory
coefs = generate_trajectory(keyframes, times, polynomial_order, mu_r, mu_psi)

# Plot the trajectory
plot_trajectory(keyframes, times, coefs, polynomial_order, len(keyframes) - 1)


#######################################
import numpy as np
from qpsolvers import solve_qp

def generate_trajectory_qp(keyframes, times, polynomial_order, mu_r, mu_psi):
    # num_segments = len(keyframes) - 1
    # num_variables_per_segment = polynomial_order * 4  # For x, y, z, ψ
    # total_variables = num_variables_per_segment * num_segments

    # # Define Hessian matrix H for the quadratic cost
    # H = np.zeros((total_variables, total_variables))
    # for i in range(num_segments):
    #     for j in range(3):
    #         # Snap coefficients (4th derivative) for x, y, z
    #         snap_indices = np.arange(j * polynomial_order + 3, (j + 1) * polynomial_order)
    #         H[snap_indices + i * num_variables_per_segment, snap_indices + i * num_variables_per_segment] = mu_r
    #     # Yaw acceleration coefficient (2nd derivative)
    #     yaw_acc_index = 3 * polynomial_order + 1
    #     H[yaw_acc_index + i * num_variables_per_segment, yaw_acc_index + i * num_variables_per_segment] = mu_psi

    # # Linear cost vector f
    # f = np.zeros(total_variables)

    # # Define the constraints matrices A and b
    # # For simplicity, we only add constraints for the keyframes here
    # A = np.zeros((len(keyframes) * 4, total_variables))  # Adjust the size based on actual constraints
    # b = np.zeros(len(keyframes) * 4)

    # for i, keyframe in enumerate(keyframes):
    #     for j in range(4):  # x, y, z, ψ
    #         A[i * 4 + j, i * num_variables_per_segment + j * polynomial_order] = 1
    #         b[i * 4 + j] = keyframe[j]
    ########
    num_segments = len(keyframes) - 1
    num_variables_per_segment = polynomial_order * 4  # For x, y, z, ψ
    total_variables = num_variables_per_segment * num_segments

    # Hessian matrix H and linear cost vector f setup remains the same
    H = np.zeros((total_variables, total_variables))
    for i in range(num_segments):
        for j in range(3):
            # Snap coefficients (4th derivative) for x, y, z
            snap_indices = np.arange(j * polynomial_order + 3, (j + 1) * polynomial_order)
            H[snap_indices + i * num_variables_per_segment, snap_indices + i * num_variables_per_segment] = mu_r
        # Yaw acceleration coefficient (2nd derivative)
        yaw_acc_index = 3 * polynomial_order + 1
        H[yaw_acc_index + i * num_variables_per_segment, yaw_acc_index + i * num_variables_per_segment] = mu_psi

    # Linear cost vector f
    f = np.zeros(total_variables)

    # Adjust the constraints matrices A and b
    num_constraints = len(keyframes) * 4 + (num_segments - 1) * 4  # Additional continuity constraints
    A = np.zeros((num_constraints, total_variables))
    b = np.zeros(num_constraints)
    # num_constraints = len(keyframes) * 4 * 3 + (num_segments - 1) * 4 * 2
    # A = np.zeros((num_constraints, total_variables))
    # b = np.zeros(num_constraints)

    constraint_idx = 0
    for i in range(num_segments):
    # Keyframe constraints
        # for i, keyframe in enumerate(keyframes):
        #     for j in range(4):  # x, y, z, ψ
        #         if i < num_segments:  # For all but the last segment
        #             A[i * 4 + j, i * num_variables_per_segment + j * polynomial_order] = 1
        #             b[i * 4 + j] = keyframe[j]
        #         if i > 0:  # Continuity constraints (except for the first segment)
        #             A[(len(keyframes) * 4) + (i - 1) * 4 + j, (i - 1) * num_variables_per_segment + j * polynomial_order] = -1
        #             A[(len(keyframes) * 4) + (i - 1) * 4 + j, i * num_variables_per_segment + j * polynomial_order] = 1
        #             b[(len(keyframes) * 4) + (i - 1) * 4 + j] = 0
        if i < num_segments - 1:
            next_segment_idx = i + 1
            for j in range(4):  # x, y, z, ψ
                # Velocity continuity
                A[constraint_idx, (i * num_variables_per_segment) + (j * polynomial_order + 1)] = 1
                A[constraint_idx, (next_segment_idx * num_variables_per_segment) + (j * polynomial_order + 1)] = -1
                constraint_idx += 1

                # Acceleration continuity
                A[constraint_idx, (i * num_variables_per_segment) + (j * polynomial_order + 2)] = 1
                A[constraint_idx, (next_segment_idx * num_variables_per_segment) + (j * polynomial_order + 2)] = -1
                constraint_idx += 1

    # Solve QP problem
    c = solve_qp(H, f, A, b,solver="proxqp")

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
keyframes = [[0, 0, 0, 0], [1, 1, 2, np.pi/2], [4, 2, 8, np.pi/2], [5, 0, 2, np.pi]]  # Keyframes with yaw angles
times = [0, 1, 2, 3]  # Times corresponding to each keyframe
polynomial_order = 5  # Quintic polynomials
mu_r = 1  # Weight for snap
mu_psi = 1  # Weight for yaw acceleration

c = generate_trajectory_qp(keyframes, times, polynomial_order, mu_r, mu_psi)
plot_trajectory_qp(keyframes, times, c, polynomial_order, len(keyframes) - 1)
