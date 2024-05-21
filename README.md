# MODEL PREDICTIVE CONTROL

Model predictive control (MPC), also known as receding horizon control or moving horizon control, is an advanced method of process control that has been widely adopted in various industries. It is a form of optimal control that uses an explicit process model to predict the future response of a system over a specified time horizon.

The basic MPC optimization problem can be formulated as:

$$
\min_{\xi, \mathbf{u}, \theta, \mathbf{v}, \rho} \sum_{k=1}^{N} e_{k}^T Q e_{k} + u_{k}^T R u_{k} + \Delta u_{k}^T S \Delta u_{k} - \gamma \rho_{k}
$$

$$
\text{s.t.} \quad \xi_0 = \xi_\text{init},
$$

$$
\xi_{k+1} = f(\xi_k, u_k),
$$

$$
\theta_{k+1} = \theta_k + \rho_k I_s,
$$

$$
0 \leq \theta_k \leq L,
$$

$$
0 \leq \rho_k \leq \bar{\rho},
$$

$$
\underline{\xi} \leq \xi_k \leq \bar{\xi},
$$

$$
\underline{u} \leq u_k \leq \bar{u},
$$

$$
\underline{b} \leq \Delta x_\text{pos} \leq \bar{b}
$$

Here, the objective is to minimize the sum of tracking errors, control efforts, and change in control efforts, subject to constraints on the initial state, system dynamics, state and input limits, and other problem-specific constraints.


The basic principle behind MPC is to optimize the current control action by solving a finite horizon optimal control problem at each time step. The optimization problem involves minimizing an objective function that considers both the desired setpoint and control effort, subject to constraints on the system inputs, outputs, and states.

The key steps involved in MPC are as follows:

1. **Process model**: A mathematical model of the process is required, which can be derived from first principles or identified using system identification techniques.
2. **Predictions**: At each time step, the current state of the process is obtained, and the model is used to predict the future behavior of the system over a specified prediction horizon.
3. **Optimization**: An optimization problem is formulated and solved to determine the sequence of future control actions that minimize the objective function while satisfying the constraints. The objective function typically includes terms for tracking the desired setpoint and minimizing control effort.
4. **Receding horizon strategy**: Only the first control action from the optimized sequence is applied to the process. At the next time step, the optimization problem is solved again with updated measurements, and the process repeats.

# MODEL PREDICTIVE PATH INTEGRAL

Model predictive path integral control (MPPI) is a sampling-based algorithm that builds upon the principles of model predictive control (MPC). It is particularly useful for controlling complex, nonlinear systems with high-dimensional state and action spaces, where traditional MPC methods may become computationally intractable.

The key idea behind MPPI is to use a sampling-based approach to approximate the optimal control sequence, rather than explicitly solving an optimization problem as in traditional MPC. MPPI combines concepts from optimal control theory, path integral methods, and reinforcement learning.

Here's how MPPI relates to and extends MPC:

- **Model-based control**: Like MPC, MPPI relies on a model of the system dynamics to predict the future behavior of the system. However, MPPI does not require an explicit optimization problem formulation.
- **Sampling-based approach**: Instead of solving an optimization problem, MPPI generates a set of candidate control sequences by sampling from a probability distribution. These control sequences are then weighted based on their associated costs or rewards, and the optimal control is approximated as a weighted average of the sampled sequences.
- **Path integral formulation**: MPPI uses a path integral formulation to compute the weights for the sampled control sequences. This formulation is inspired by techniques from stochastic optimal control and path integral control, which provide a principled way to update the control distribution based on the cost or reward function.

The main advantages of MPPI compared to traditional MPC include:

- **Handling high-dimensional and nonlinear systems**: MPPI can handle complex, nonlinear systems with high-dimensional state and action spaces, where traditional MPC methods may struggle due to the computational complexity of solving the optimization problem.
- **No explicit constraints**: MPPI does not require explicit formulation of constraints on states and inputs, as the cost or reward function can implicitly encode desired constraints.
- **Robustness**: MPPI can cope with model uncertainties and disturbances by continuously re-evaluating the control distribution based on the latest state information.



