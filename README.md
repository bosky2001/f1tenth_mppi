# MODEL PREDICTIVE CONTROL

Model predictive control (MPC), also known as receding horizon control or moving horizon control, is an advanced method of process control that has been widely adopted in various industries. It is a form of optimal control that uses an explicit process model to predict the future response of a system over a specified time horizon.

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
- **Scalability**: The sampling-based approach in MPPI scales better to high-dimensional systems compared to optimization-based MPC methods.
- **Robustness**: MPPI can cope with model uncertainties and disturbances by continuously re-evaluating the control distribution based on the latest state information.

MPPI has been successfully applied in various domains, including robotics, and autonomous vehicles, where it has demonstrated superior performance in controlling complex, nonlinear systems.

# CODE WALKTHROUGH
This repo contains MPPI written in JAX by [Google Research](https://github.com/google-research/google-research/blob/c9f05e51f37cacc291f58799a1f732743625078b/jax_mpc/jax_mpc/mppi.py). JAX is particularly suited for monte-carlo style MPC, as rollouts can be efficiently parallelized using jax.vmap().

# Model Predictive Path Integral (MPPI) Control for F1TENTH Autonomous Racing

This is a detailed tutorial on how to use and walk through the provided code, which is an implementation of the Model Predictive Path Integral (MPPI) control algorithm for motion planning and control in autonomous systems, specifically for the F1TENTH autonomous racing platform.

## Understanding MPPI

Before diving into the code, let's first understand the MPPI algorithm and its components:

### Model Predictive Control (MPC)
- MPC is an advanced control technique that uses a system model to predict the future behavior of the system and optimize the control inputs over a receding horizon.
- MPPI is a variant of MPC that uses a sampling-based approach to approximate the optimal control sequence.

### Path Integral Control
- Path integral control is a stochastic optimal control technique that uses sampling to compute the optimal control input distribution.
- MPPI borrows the concept of weighting the sampled control sequences based on their associated costs or rewards.

### Reinforcement Learning
- MPPI can be viewed as a reinforcement learning algorithm, where the cost or reward function encodes the desired behavior.
- The algorithm learns to generate control sequences that maximize the cumulative reward or minimize the cumulative cost.

## Code Walkthrough

### 1. `mppi.py`
This file contains the implementation of the MPPI algorithm. Here's a breakdown of the essential components:

- `MPPI` class: Implements the MPPI algorithm with methods for state initialization, state update, and optimal control action computation.
- `init_state`: Initializes the MPPI state, including the optimal control sequence and covariance matrices (if adaptive covariance is used).
- `update`: Core method that performs the MPPI iteration by generating candidate control sequences, evaluating associated costs/rewards, and computing the optimal control sequence.
- `get_action`: Retrieves the first control action from the optimal control sequence.
- `returns`: Helper method that computes the cumulative returns (sum of rewards) for a given reward sequence.
- `weights`: Helper method that computes the weights for the sampled control sequences based on their associated returns.
- `rollout`: Helper method that simulates the system dynamics using the provided control sequence and environment model.

### 2. `mppi_env.py`
This file contains the implementation of the environment model used by the MPPI algorithm. It includes the following components:

- `MPPIEnv` class: Represents the environment model for the F1TENTH autonomous racing platform.
- `step`: Simulates the system dynamics using either the kinematic single-track (KS) model or the dynamic single-track (ST) model, taking the current state and control inputs as input and returning the next state, rewards, and dynamics residuals.
- `reward_fn`: Computes the reward based on the current state and a reference trajectory.
- `calc_ref_trajectory`: Calculates the reference trajectory for a given state, course, and speed profile.
- `get_reference_traj`: Generates the reference trajectory for the current state based on the waypoints and target speed.

### 3. `main.py`
This file contains the main ROS node implementation and serves as the entry point for the MPPI planner. Here's what it does:

- `MPPIPlanner` class: Inherits from the `Node` class provided by ROS2 and represents the MPPI planner node.
- `__init__`: Initializes the ROS node, sets up publishers and subscribers, loads the waypoints, and creates instances of the `MPPIEnv` and `MPPI` classes.
- `load_waypoints`: Loads the waypoints from a provided file path.
- `init_state`: Initializes the MPPI state.
- `pose_callback`: Callback function executed whenever a new pose message is received from the robot's odometry. It performs the following tasks:
 - Updates the MPPI state using the current state and sampled control sequences.
 - Computes the reference trajectory based on the current state and waypoints.
 - Determines the optimal control action from the MPPI state.
 - Publishes the control commands (steering angle and velocity) to the robot.
 - Visualizes the reference trajectory, optimal trajectory, and sampled trajectories.
- Visualization methods (`viz_ref_points`, `viz_rej_traj`, `viz_opt_traj`, `viz_sampled_traj`, `pub_sampled_traj`): Provided to visualize the reference waypoints, reference trajectory, optimal trajectory, and sampled trajectories.

## Running the Code

To use this code, you need to have a working ROS2 environment set up and the required dependencies installed (e.g., `jax`, `jaxlib`, `numpy`, `rclpy`). Here are the steps to run the code:

1. Start a ROS2 environment by running `ros2 run` in a terminal.
2. In another terminal, navigate to the directory containing the `main.py` file.
3. Run the `main.py` script using `python3 main.py`.
4. The MPPI planner node should start, and you should see output messages indicating its initialization.
5. Once the robot's odometry data starts streaming (either from a simulation or a real robot), the MPPI planner will start computing and publishing control commands.
6. You can visualize the reference waypoints, reference trajectory, optimal trajectory, and sampled trajectories using tools like RViz or other visualization tools that subscribe to the corresponding ROS topics.

Note that the code assumes the availability of a robot's odometry data and a set of predefined waypoints. You may need to modify the code or provide the required inputs (e.g., waypoints file path, odometry topic) based on your specific setup and requirements.

Additionally, the code includes various configuration parameters, such as the number of MPPI iterations, samples, prediction horizon, and other algorithm-specific parameters. You may need to adjust these parameters based on your system dynamics, performance requirements, and desired behavior.

Overall, this code provides a solid implementation of the MPPI algorithm for motion planning and control in autonomous systems, specifically tailored for the F1TENTH autonomous racing platform. However, it may require additional integration and customization to work with your specific setup and requirements.
Additionally, the code includes various configuration parameters, such as the number of MPPI iterations, samples, prediction horizon, and other algorithm-specific parameters. You may need to adjust these parameters based on your system dynamics, performance requirements, and desired behavior.
Overall, this code provides a solid implementation of the MPPI algorithm for motion planning and control in autonomous systems, specifically tailored for the F1TENTH autonomous racing platform. However, it may require additional integration and customization to work with your specific setup and requirements.
