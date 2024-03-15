import torch
from qpth.qp import QPFunction
import matplotlib.pyplot as plt
import numpy as np

class MinimumSnapTrajectoryPlanner:
    def __init__(self, n_order: int, v0: torch.Tensor, a0: torch.Tensor, ve: torch.Tensor, ae: torch.Tensor):
        self.n_order = n_order
        self.v0 = v0
        self.a0 = a0
        self.ve = ve
        self.ae = ae
    
    def arrange_time_stamps(self, waypts: torch.Tensor, total_time: float) -> torch.Tensor:
        """
        Arrange time stamps based on distances between waypoints and total time.
        """
        x = waypts[:, 1:] - waypts[:, :-1]
        dist = torch.sqrt(torch.sum(x ** 2, dim=0))
        k = total_time / torch.sum(dist)
        ts = torch.cat([torch.tensor([0]), torch.cumsum(dist * k, dim=0)])
        return ts
    
    def calc_tvec(self, t: float, n_order: int, r: int) -> torch.Tensor:
        """
        Calculate the time vector for a given time 't', order of polynomial 'n_order', and derivative 'r'.
        """
        tvec = torch.zeros(n_order + 1)
        for i in range(r + 1, n_order + 2):
            product = torch.prod(torch.arange(i - r, i, dtype=torch.float32))
            tvec[i - 1] = product * (t ** (i - r - 1))
        return tvec

    # ... (other methods like computeQ, minimum_snap_single_axis_simple remain the same)
    def computeQ(n, r, t1, t2):
        T = torch.zeros((n - r) * 2 + 1)
        # print(T)
        for i in range((n - r) * 2 + 1):
            T[i] = t2 ** (i + 1) - t1 ** (i + 1)  # Align with MATLAB's 1-based indexing
            # print(T[i])

        Q = torch.zeros(n + 1, n + 1) # Matrix size n+1 to match MATLAB's zeros(n+1,n+1)
        # print(Q)
        for i in range(r+1, n + 2):  # Adjusted range for 0-based indexing
            for j in range(i, n + 2):  # Adjusted range for 0-based indexing
                k1 = i - r -1
                # print(k1)
                k2 = j - r -1
                k = k1 + k2 + 1
                # print(k)
                # Adjusting product calculations to match MATLAB's inclusive range
                prod_k1 = torch.prod(torch.tensor(range(k1 + 1, k1 + r+1 )))
                # print(prod_k1)
                prod_k2 = torch.prod(torch.tensor(range(k2 + 1, k2 + r+1))) 
                # print(prod_k1 * prod_k2 / k * T[k - 1])
                Q[i-1, j-1] = prod_k1 * prod_k2 / k * T[k - 1]  # Corrected T index for 0-based indexing
                Q[j-1, i-1] = Q[i-1, j-1]

        return Q
    def minimum_snap_single_axis_simple(waypts, ts, n_order, v0, a0, ve, ae):
        # with torch.no_grad():
        p0 = waypts[0]
        pe = waypts[-1]
        n_poly = len(waypts) -1
        # print(n_poly)
        n_coef = n_order + 1
        # corridor_r = 0.1 

        # Compute Q
        Q_all = torch.block_diag(*[computeQ(n_order, 3, ts[i], ts[i + 1]) for i in range(n_poly)])
        # print(Q_all)
        b_all = torch.zeros(Q_all.shape[0], dtype=torch.float64)

        # print(b_all.shape)

        # Setup equality constraints
        Aeq = torch.zeros(4 * n_poly + 2, n_coef * n_poly, dtype=torch.float64)
        # print(Aeq.shape)
        beq = torch.zeros(4 * n_poly + 2, dtype=torch.float64)
        # print(beq.shape)
        Aeq[0:3, :n_coef] = torch.stack([
            calc_tvec(ts[0], n_order, 0),
            calc_tvec(ts[0], n_order, 1),
            calc_tvec(ts[0], n_order, 2)])
        # print(Aeq)
        Aeq[3:6, -n_coef:] = torch.stack([
            calc_tvec(ts[-1], n_order, 0),
            calc_tvec(ts[-1], n_order, 1),
            calc_tvec(ts[-1], n_order, 2)])
        # print(Aeq.shape)
        beq[0:6] = torch.tensor([p0, v0, a0, pe, ve, ae])

        # Mid p constraints
        neq = 6
        for i in range(1, n_poly):
            Aeq[neq, i * n_coef:(i + 1) * n_coef] = calc_tvec(ts[i], n_order, 0)
            beq[neq] = waypts[i]
            neq += 1
        # print(Aeq)
        # Continuous constraints
        for i in range(1, n_poly):
            tvec_p = calc_tvec(ts[i], n_order, 0)
            tvec_v = calc_tvec(ts[i], n_order, 1)
            tvec_a = calc_tvec(ts[i], n_order, 2)
            Aeq[neq:neq + 3, (i - 1) * n_coef:(i + 1) * n_coef] = torch.stack([
                torch.cat([tvec_p, -tvec_p]),
                torch.cat([tvec_v, -tvec_v]),
                torch.cat([tvec_a, -tvec_a])])
            neq += 3
        # print( Q_all.size(0))
            
        G_dummy = torch.zeros(1, Q_all.size(0), Q_all.size(0), dtype=torch.float64)  # 3D tensor with the correct shape
        h_dummy = torch.ones(1, Q_all.size(0) ,dtype=torch.float64)
            
        Q_all = Q_all.double()
        b_all = b_all.double()
        Aeq = Aeq.double()
        beq = beq.double()
        G_dummy = G_dummy.double()
        h_dummy = h_dummy.double()
        

        # # Add a small positive value to diagonal of Q for numerical stability
        Q_all += torch.eye(Q_all.size(0), dtype=torch.float64) * 1e-6

        # Solve the quadratic program
        qpsolver = QPFunction(verbose=1)
        p = qpsolver(Q_all, b_all, G_dummy, h_dummy, Aeq, beq)
        # solver_options = {'maxIter': 10, 'solver': qpth.qp.QPSolvers.CVXPY}  # Increase iterations here
        # p = QPFunction(verbose=1, **solver_options)(Q_all, b_all, G_dummy, h_dummy, Aeq, beq)
        polys = p.view(n_coef, n_poly)
        return polys
        '''Q_np = Q_all.squeeze(0).numpy()  # Removing the batch dimension
        p_np = b_all.squeeze(0).numpy()
        G_np = G_dummy.squeeze(0).numpy()
        h_np = h_dummy.squeeze(0).numpy()
        A_np = Aeq.squeeze(0).numpy()
        b_np = beq.squeeze(0).squeeze(-1).numpy()  # Removing the redundant dimension

        # Define the cvxpy problem
        x = cp.Variable(Q_np.shape[1])
        objective = cp.Minimize((1/2) * cp.quad_form(x, Q_np) + p_np.T @ x)
        constraints = [G_np @ x <= h_np, A_np @ x == b_np]
        prob = cp.Problem(objective, constraints)

        # Solve the problem
        result = prob.solve()

        # Check if the problem is solved successfully
        if prob.status in ["infeasible", "unbounded"]:
            print("Problem is infeasible or unbounded")
        else:
            print("Optimal value:", prob.value)

        # Convert the solution back to PyTorch tensor
        x_solution = torch.tensor(x.value, dtype=torch.float64)

        # Reshape the solution to match the number of coefficients and polynomials
        polys = x_solution.view(n_coef, n_poly)
        return polys'''

    def poly_val(poly, t, r):
        val = 0
        n = len(poly) - 1
        if r <= 0:
            for i in range(n + 1):
                val += poly[i] * t**i
        else:
            for i in range(r, n + 1):
                a = poly[i] * np.prod(range(i-r+1, i+1)) * t**(i-r)
                val += a
        return val


    def polys_vals(polys, ts, tt, r):
        N = tt.size(0)
        vals = torch.zeros(N)
        # print(vals)
        idx = 0
        for i in range(N):
            t = tt[i]
            if t < ts[idx]:
                vals[i] = 0
            else:
                while idx < len(ts) - 1 and t > ts[idx + 1] + 0.0001:
                    idx += 1
                vals[i] = poly_val(polys[:, idx], t, r)
        return vals

    def plot_trajectory(self, waypts: torch.Tensor, polys_x: torch.Tensor, polys_y: torch.Tensor, ts: torch.Tensor):
        """
        Plot the trajectory based on waypoints and polynomial coefficients.
        """
        plt.plot(waypts[0], waypts[1], '*r')
        plt.plot(waypts[0], waypts[1], 'b--')
        plt.title('Minimum Snap Trajectory')
        colors = ['g', 'r', 'c']
        for i in range(polys_x.shape[1]):
            tt = torch.arange(ts[i], ts[i+1], 0.01)
            xx = self.polys_vals(polys_x, ts, tt, 0)
            yy = self.polys_vals(polys_y, ts, tt, 0)
            plt.plot(xx.detach().numpy(), yy.detach().numpy(), colors[i % 3])
        plt.show()

    # ... (other methods like polys_vals remain the same)

if __name__ == "__main__":
    # Define waypoints, initial and final velocities, and accelerations
    waypts = torch.tensor([[0, 0], [1, 2], [2, -1], [4, 8], [5, 2]], dtype=torch.float64).t()
    v0, a0, ve, ae = torch.tensor([0, 0]), torch.tensor([0, 0]), torch.tensor([0, 0]), torch.tensor([0, 0])
    total_time = 5
    n_order = 5

    # Create an instance of the planner
    planner = MinimumSnapTrajectoryPlanner(n_order, v0, a0, ve, ae)
    ts = planner.arrange_time_stamps(waypts, total_time)
    
    # Plan the trajectory
    polys_x = planner.minimum_snap_single_axis_simple(waypts[0], ts, planner.n_order, planner.v0[0], planner.a0[0], planner.ve[0], planner.ae[0])
    polys_y = planner.minimum_snap_single_axis_simple(waypts[1], ts, planner.n_order, planner.v0[1], planner.a0[1], planner.ve[1], planner.ae[1])
    
    # Plot the trajectory
    planner.plot_trajectory(waypts, polys_x, polys_y, ts)
