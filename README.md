# MODEL PREDICTIVE CONTROL
Model predictive control (MPC), also known as receding horizon control or moving horizon control, is an advanced method of process control that has been widely adopted in various industries, including chemical processing, oil refining, and automotive applications. It is a form of optimal control that uses an explicit process model to predict the future response of a system over a specified time horizon.
The basic principle behind MPC is to optimize the current control action by solving a finite horizon optimal control problem at each sampling instant. The optimization problem involves minimizing an objective function that considers both the desired setpoint and control effort, subject to constraints on the system inputs, outputs, and states.
The key steps involved in MPC are as follows:

Process model: A mathematical model of the process is required, which can be derived from first principles or identified using system identification techniques.
Predictions: At each sampling instant, the current state of the process is obtained, and the model is used to predict the future behavior of the system over a specified prediction horizon.
Optimization: An optimization problem is formulated and solved to determine the sequence of future control actions that minimize the objective function while satisfying the constraints. The objective function typically includes terms for tracking the desired setpoint and minimizing control effort.
Receding horizon strategy: Only the first control action from the optimized sequence is applied to the process. At the next sampling instant, the optimization problem is solved again with updated measurements, and the process repeats.

The main advantages of MPC include:

Constraint handling: MPC can explicitly handle constraints on inputs, outputs, and states, which is essential for safe and optimal operation of many processes.
Multivariable control: MPC can handle multiple inputs and outputs simultaneously, making it suitable for complex systems with interactions and coupling.
Prediction of future behavior: By considering the predicted future behavior of the system, MPC can anticipate and account for upcoming disturbances or setpoint changes, resulting in improved performance.
Flexibility: MPC can incorporate various objectives, constraints, and models, making it adaptable to different control problems.

MPC has been successfully applied in various industries, including chemical processing, oil refining, automotive systems, aerospace, and renewable energy systems, among others. It is particularly useful in systems with significant dynamics, constraints, and multivariable interactions.

# MODEL PREDICTIVE PATH INTEGRAL
Model predictive path integral control (MPPI) is a reinforcement learning algorithm that builds upon the principles of model predictive control (MPC). It is particularly useful for controlling complex, nonlinear systems with high-dimensional state and action spaces, where traditional MPC methods may become computationally intractable.
The key idea behind MPPI is to use a sampling-based approach to approximate the optimal control sequence, rather than explicitly solving an optimization problem as in traditional MPC. MPPI combines concepts from optimal control theory, path integral methods, and reinforcement learning.
Here's how MPPI relates to and extends MPC:

Model-based control: Like MPC, MPPI relies on a model of the system dynamics to predict the future behavior of the system. However, MPPI does not require an explicit optimization problem formulation.
Sampling-based approach: Instead of solving an optimization problem, MPPI generates a set of candidate control sequences by sampling from a probability distribution. These control sequences are then weighted based on their associated costs or rewards, and the optimal control is approximated as a weighted average of the sampled sequences.
Path integral formulation: MPPI uses a path integral formulation to compute the weights for the sampled control sequences. This formulation is inspired by techniques from stochastic optimal control and path integral control, which provide a principled way to update the control distribution based on the cost or reward function.
Reinforcement learning: MPPI can be viewed as a reinforcement learning algorithm, where the cost or reward function encodes the desired behavior, and the algorithm learns to generate control sequences that maximize the cumulative reward or minimize the cumulative cost.

The main advantages of MPPI compared to traditional MPC include:

Handling high-dimensional and nonlinear systems: MPPI can handle complex, nonlinear systems with high-dimensional state and action spaces, where traditional MPC methods may struggle due to the computational complexity of solving the optimization problem.
No explicit constraints: MPPI does not require explicit formulation of constraints on states and inputs, as the cost or reward function can implicitly encode desired constraints.
Scalability: The sampling-based approach in MPPI scales better to high-dimensional systems compared to optimization-based MPC methods.
Robustness: MPPI can cope with model uncertainties and disturbances by continuously re-evaluating the control distribution based on the latest state information.

MPPI has been successfully applied in various domains, including robotics, autonomous vehicles, and aerospace systems, where it has demonstrated superior performance in controlling complex, nonlinear systems.

# CODE WALKTHROUGH
This repo contains MPPI written in JAX by [Google Research](https://github.com/google-research/google-research/blob/c9f05e51f37cacc291f58799a1f732743625078b/jax_mpc/jax_mpc/mppi.py). JAX is particularly suited for monte-carlo style MPC, as rollouts can be efficiently parallelized using jax.vmap().

The code is an implementation of the Model Predictive Path Integral (MPPI) control algorithm for motion planning and control in autonomous systems, specifically for the F1TENTH autonomous racing platform.
Before diving into the code, let's first understand the MPPI algorithm and its components:

Model Predictive Control (MPC): MPC is an advanced control technique that uses a system model to predict the future behavior of the system and optimize the control inputs over a receding horizon. MPPI is a variant of MPC that uses a sampling-based approach to approximate the optimal control sequence.
Path Integral Control: Path integral control is a stochastic optimal control technique that uses sampling to compute the optimal control input distribution. MPPI borrows the concept of weighting the sampled control sequences based on their associated costs or rewards.
Reinforcement Learning: MPPI can be viewed as a reinforcement learning algorithm, where the cost or reward function encodes the desired behavior, and the algorithm learns to generate control sequences that maximize the cumulative reward or minimize the cumulative cost.

Now, let's go through the code step by step:
1. mppi.py
This file contains the implementation of the MPPI algorithm. Here's a breakdown of the essential components:

MPPI class: This class implements the MPPI algorithm. It has methods for initializing the state, updating the state based on the current environment and control inputs, and computing the optimal control action.
init_state: This method initializes the MPPI state, which includes the optimal control sequence and covariance matrices (if adaptive covariance is used).
update: This is the core method that performs the MPPI iteration. It generates a set of candidate control sequences by sampling from a probability distribution, evaluates the associated costs or rewards using the environment model, and computes the optimal control sequence as a weighted average of the sampled sequences.
get_action: This method retrieves the first control action from the optimal control sequence.
returns: This helper method computes the cumulative returns (sum of rewards) for a given reward sequence.
weights: This helper method computes the weights for the sampled control sequences based on their associated returns.
rollout: This helper method simulates the system dynamics using the provided control sequence and environment model.

2. mppi_env.py
This file contains the implementation of the environment model used by the MPPI algorithm. It includes the following components:

MPPIEnv class: This class represents the environment model for the F1TENTH autonomous racing platform.
step: This method simulates the system dynamics using either the kinematic single-track (KS) model or the dynamic single-track (ST) model. It takes the current state and control inputs as input and returns the next state, along with additional information like rewards and dynamics residuals.
reward_fn: This method computes the reward based on the current state and a reference trajectory.
calc_ref_trajectory: This method calculates the reference trajectory for a given state, course, and speed profile.
get_reference_traj: This method generates the reference trajectory for the current state based on the waypoints and target speed.

3. main.py
This file contains the main ROS node implementation and serves as the entry point for the MPPI planner. Here's what it does:

MPPIPlanner class: This class inherits from the Node class provided by ROS2 and represents the MPPI planner node.
__init__: This method initializes the ROS node, sets up publishers and subscribers, loads the waypoints, and creates instances of the MPPIEnv and MPPI classes.
load_waypoints: This method loads the waypoints from a provided file path.
init_state: This method initializes the MPPI state.
pose_callback: This is the callback function that gets executed whenever a new pose message is received from the robot's odometry. It performs the following tasks:

Updates the MPPI state using the current state and sampled control sequences.
Computes the reference trajectory based on the current state and waypoints.
Determines the optimal control action from the MPPI state.
Publishes the control commands (steering angle and velocity) to the robot.
Visualizes the reference trajectory, optimal trajectory, and sampled trajectories.


Various visualization methods (viz_ref_points, viz_rej_traj, viz_opt_traj, viz_sampled_traj, pub_sampled_traj) are provided to visualize the reference waypoints, reference trajectory, optimal trajectory, and sampled trajectories.

To use this code, you need to have a working ROS2 environment set up and the required dependencies installed (e.g., jax, jaxlib, numpy, rclpy). Here are the steps to run the code:

Start a ROS2 environment by running ros2 run in a terminal.
In another terminal, navigate to the directory containing the main.py file.
Run the main.py script using python3 main.py.
The MPPI planner node should start, and you should see output messages indicating its initialization.
Once the robot's odometry data starts streaming (either from a simulation or a real robot), the MPPI planner will start computing and publishing control commands.
You can visualize the reference waypoints, reference trajectory, optimal trajectory, and sampled trajectories using tools like RViz or other visualization tools that subscribe to the corresponding ROS topics.

Note that the code assumes the availability of a robot's odometry data and a set of predefined waypoints. You may need to modify the code or provide the required inputs (e.g., waypoints file path, odometry topic) based on your specific setup and requirements.
Additionally, the code includes various configuration parameters, such as the number of MPPI iterations, samples, prediction horizon, and other algorithm-specific parameters. You may need to adjust these parameters based on your system dynamics, performance requirements, and desired behavior.
Overall, this code provides a solid implementation of the MPPI algorithm for motion planning and control in autonomous systems, specifically tailored for the F1TENTH autonomous racing platform. However, it may require additional integration and customization to work with your specific setup and requirements.
