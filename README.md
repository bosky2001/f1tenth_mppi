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
