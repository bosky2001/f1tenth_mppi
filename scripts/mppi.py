import jax
import jax.numpy as jnp
from functools import partial


class MPPI():
    """An MPPI based planner."""
    def __init__(self, config, jRNG, temperature=0.01,
                damping=0.001, a_noise=0.5, scan=False):
        self.config = config
        self.jRNG = jRNG
        self.n_iterations = config.n_iterations
        self.n_steps = config.n_steps
        self.n_samples = config.n_samples
        self.temperature = temperature
        self.damping = damping
        self.a_std = a_noise
        self.scan = scan  # whether to use jax.lax.scan instead of python loop

        self.adaptive_covariance = config.adaptive_covariance
        self.a_shape = config.control_dim
        self.accum_matrix = jnp.triu(jnp.ones((self.n_steps, self.n_steps)))

    def init_state(self, env):
        # uses random as a hack to support vmap
        # we should find a non-hack approach to initializing the state
        self.env = env
        self.dim_a = jnp.prod(self.env.a_shape)  # np.int32
        a_opt = 0.0*jax.random.uniform(self.jRNG.new_key(), shape=(self.n_steps,
                                                self.dim_a))  # [n_steps, dim_a]
        # a_cov: [n_steps, dim_a, dim_a]
        if self.adaptive_covariance:
            # note: should probably store factorized cov,
            # e.g. cholesky, for faster sampling
            a_cov = (self.a_std**2)*jnp.tile(jnp.eye(self.dim_a), (self.n_steps, 1, 1))
        else:
            a_cov = None
        
        self.a_opt = a_opt
        self.a_cov = a_cov
 



    @partial(jax.jit, static_argnums=(0, 1))
    def get_a_opt(self, env, a_opt, reference, s, da):
        r = jax.vmap(env.reward_fn, in_axes=(0, None))(
                s, reference
            ) # [n_samples, n_steps]
            
        R = jax.vmap(self.returns)(r) # [n_samples, n_steps], pylint: disable=invalid-name
        w = jax.vmap(self.weights, 1, 1)(R)  # [n_samples, n_steps]
        da_opt = jax.vmap(jnp.average, (1, None, 1))(da, 0, w)  # [n_steps, dim_a]
        a_opt = jnp.clip(a_opt + da_opt, -1.0, 1.0)  # [n_steps, dim_a]
        return a_opt

    @partial(jax.jit, static_argnums=(0))
    def get_samples(self, env_state, a_opt, rng):
        da = jax.random.truncated_normal(
                rng,
                -jnp.ones_like(a_opt) - a_opt,
                jnp.ones_like(a_opt) - a_opt,
                shape=(self.n_samples, self.n_steps, 2)
            )
        a = jnp.clip(jnp.expand_dims(a_opt, axis=0) + da, -1.0, 1.0)
        # print('up', env_state.shape)


        _, s = jax.vmap(self.rollout, in_axes=(0, None))(
            a, env_state
        )
        return s, da

    @partial(jax.jit, static_argnums=(0, 1))
    def iter_step(self, env, env_state, rng, reference_traj):

        self.a_opt = jnp.concatenate([self.a_opt[1:, :],
                                jnp.expand_dims(jnp.zeros((self.dim_a,)),
                                                axis=0)])  # [n_steps, dim_a]

        a_opt = self.a_opt
        a_cov = self.a_cov
        
        s, da= self.get_samples(env_state, a_opt, rng)
        # print(s.shape)

        a_opt = self.get_a_opt(env, a_opt, reference_traj, s, da)


        
        _, s_opt = self.rollout(a_opt, env_state)
            
        return (a_opt, a_cov, s, s_opt)

    # @partial(jax.jit, static_argnums=(0, 1))
    def update(self, env, env_state, rng):
        # mpc_state: ([n_steps, dim_a], [n_steps, dim_a, dim_a])
        # env: {.step(s, a), .reward(s)}
        # env_state: [env_shape] np.float32
        # rng: rng key for mpc sampling

        
        # a_opt, a_cov = mpc_state

        for _ in range(self.n_iterations):
            # (a_opt, a_cov, s, s_opt), _ = iteration_step((a_opt, a_cov, rng), None)
            # self.a_opt, self.a_cov, s= self.iteration_step(env, self.a_opt, self.a_cov, rng, env_state, env.reference)
            (a_opt, a_cov, s, s_opt) = self.iter_step(env, env_state,rng, env.reference)

            # predicted_states.append(s)

        # self.a_opt = a_opt
        # self.a_cov = a_cov
        return (a_opt, a_cov), s, s_opt


    def get_action(self, mpc_state, a_shape):
        a_opt, _ = mpc_state
        return jnp.reshape(a_opt[0, :], a_shape)


    @partial(jax.jit, static_argnums=(0))
    def returns(self, r):
        # r: [n_steps]
        return jnp.dot(self.accum_matrix, r)  # R: [n_steps]


    @partial(jax.jit, static_argnums=(0))
    def weights(self, R):  # pylint: disable=invalid-name
        # R: [n_samples]
        # R_stdzd = (R - jnp.min(R)) / ((jnp.max(R) - jnp.min(R)) + self.damping)
        # R_stdzd = R - jnp.max(R) # [n_samples] np.float32
        R_stdzd = (R - jnp.max(R)) / ((jnp.max(R) - jnp.min(R)) + self.damping)  # pylint: disable=invalid-name
        w = jnp.exp(R_stdzd / self.temperature)  # [n_samples] np.float32
        w = w/jnp.sum(w)  # [n_samples] np.float32
        return w
    

    @partial(jax.jit, static_argnums=(0))
    def rollout(self, actions, env_state):
        # actions: [n_steps, dim_a]
        # env: {.step(s, a), .reward(s)}
        # env_state: np.float32

        def rollout_step(env_state, actions):
            actions = jnp.reshape(actions, self.env.a_shape)
            (env_state, env_var, mb_dyna) = self.env.step(env_state, actions)
            reward = self.env.reward(env_state)
            return env_state, (env_state, reward)

        scan_output = []
        for t in range(self.n_steps):
            env_state, output = rollout_step(env_state, actions[t, :])
            scan_output.append(output)
        states, reward = jax.tree_util.tree_map(lambda *x: jnp.stack(x), *scan_output)
   
        return reward, states
   