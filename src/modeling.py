import numpy as np
from scipy.special import softmax
from scipy.optimize import differential_evolution
from multiprocessing import Pool

class Environment:
    """
    Environment class
    """
    def __init__(self, states: list[int], actions: list[int]):
        """
        Args:
            states: list of states
            actions: list of actions
        """
        self.states = states
        self.actions = actions
        self.n_states = len(states)
        self.n_actions = len(actions)
        self.reward_function = {}
        for state in states:
            self.reward_function[state] = np.nan


class ProbabilisticReversalEnvironment(Environment):
    """
    Probabilistic Reversal Environment class
    """
    def __init__(self, states: list[int], actions: list[int], n_trials_per_episode: int, n_episodes: int, p_reward_correct: float, p_reward_incorrect: float):
        """
        Args:
            states: list of states
            actions: list of actions
            n_trials_per_episode: number of trials per episode
            n_episodes: number of episodes
            p_reward_correct: probability of reward for the correct action
            p_reward_incorrect: probability of reward for the incorrect action
        """
        super().__init__(states, actions)
        self.reset_reward_function()
        self.n_trials_per_episode = n_trials_per_episode
        self.n_episodes = n_episodes
        self.n_trials = self.n_trials_per_episode * self.n_episodes
        self.reward_probabilities = np.array([p_reward_incorrect, p_reward_correct])

    def reset_reward_function(self):
        """
        Randomly reset the reward function
        """
        for state in self.states:
            self.reward_function[state] = np.random.choice(self.actions)

    def get_reward(self, state: int, action: int) -> int:
        """
        Get the reward for the given state and action

        Args:
            state: state
            action: action

        Returns:
            reward: reward
        """
        if self.reward_function[state] == action:
            return int(np.random.rand() < self.reward_probabilities[1])
        else:
            return int(np.random.rand() < self.reward_probabilities[0])

    def generate_trials(self) -> np.ndarray:
        """
        Generate the trials for the given number of episodes and trials per episode

        Returns:
            data: np.ndarray of shape (n_trials, 4)
            data[:, 0]: episode index
            data[:, 1]: trial index
            data[:, 2]: state
            data[:, 3]: correct action
        """
        episodes = np.arange(self.n_episodes).repeat(self.n_trials_per_episode)
        trials = np.arange(self.n_trials_per_episode).repeat(self.n_episodes).reshape(self.n_trials_per_episode, self.n_episodes).T.flatten()
        states = np.random.randint(0, self.n_states, self.n_trials)
        correct_actions = np.zeros_like(states)
        for episode in range(self.n_episodes):
            self.reset_reward_function()
            correct_actions[episode * self.n_trials_per_episode:(episode + 1) * self.n_trials_per_episode] = [self.reward_function[states[i]] for i in range(episode * self.n_trials_per_episode, (episode + 1) * self.n_trials_per_episode)]
        data = np.array([episodes, trials, states, correct_actions]).T.astype(int)
        return data


class RLModel:
    """
    RLModel class
    """
    def __init__(self, env: Environment, param_names: list[str], param_bounds: list[list[float, float]]):
        """
        Args:
            env: Environment class instance
            param_names: list of parameter names for the RL model
            param_bounds: list of parameter bounds for the RL model
        """
        self.env = env
        self.n_states = env.n_states
        self.n_actions = env.n_actions
        self.param_names = param_names
        self.param_bounds = {}
        for param_name, param_bound in zip(param_names, param_bounds):
            self.param_bounds[param_name] = param_bound
        self.params = {}
        self.q_table = np.zeros((self.n_states, self.n_actions))

    def parse_params(self, params: list[float]):
        """
        Parse the parameters

        Args:
            params: list of parameters
        """
        for param_i, param_name in enumerate(self.param_names):
            self.params[param_name] = params[param_i]

    def sample_params(self, method: str="uniform") -> list[float]:
        """
        Sample the parameters from the parameter bounds using the specified method. 

        Args:
            method: method to sample the parameters. Supported methods include "uniform".

        Returns:
            params: list of parameters
        """
        if method == "uniform":
            return [np.random.uniform(self.param_bounds[param_name][0], self.param_bounds[param_name][1]) for param_name in self.param_names]
        else:
            raise ValueError(f"Method {method} not supported")

    def get_q_value(self, state: int, action: int) -> float:
        """
        Get the Q-value for the given state and action

        Args:
            state: state
            action: action

        Returns:
            q_value: Q-value table
        """
        return self.q_table[state, action]
    
    def update_q_value(self, state: int, action: int, reward: int):
        """
        Update the Q-value for the given state-action pair using the Q-learning update rule.

        Args:
            state: state
            action: action
            reward: reward
        """
        self.q_table[state, action] += self.params["alpha"] * (reward - self.q_table[state, action])

    def softmax_policy(self, state: int) -> np.ndarray:
        """
        Get the softmax policy for the given state

        Args:
            state: state

        Returns:
            policy: softmax policy
        """
        q_values = self.q_table[state]
        return softmax(self.params["beta"] * q_values)
    
    def sample_action(self, state: int) -> int:
        """
        Sample the action from the softmax policy for the given state

        Args:
            state: state

        Returns:
            action: action
        """
        policy = self.softmax_policy(state)
        return np.random.choice(self.n_actions, p=policy)

    def individual_simulate(self, params: list[float], env_data: np.ndarray) -> np.ndarray:
        """
        Simulate the RL model for the given parameters and environment data

        Args:
            params: list of parameters
            env_data: environment data

        Returns:
            data: np.ndarray of shape (n_trials, 6)
            data[:, 0]: episode index
            data[:, 1]: trial index
            data[:, 2]: state
            data[:, 3]: correct action
            data[:, 4]: action
            data[:, 5]: reward
        """
        self.parse_params(params)
        n_trials = env_data.shape[0]
        data = np.zeros((n_trials, 6))
        data[:, 0] = env_data[:, 0].copy()
        data[:, 1] = env_data[:, 1].copy()
        data[:, 2] = env_data[:, 2].copy()
        for t in range(n_trials):
            state = data[t, 2]
            action = self.sample_action(state)
            correct_action = self.env.reward_function[state]
            reward = self.env.get_reward(state, action)
            self.update_q_value(state, action, reward)
            data[t, 3:] = [correct_action, action, reward]
        return data

    def parallel_simulate(self, args) -> tuple[int, int, np.ndarray]:
        """
        Simulate the RL model for the given parameters and environment data in parallel

        Args:
            args: tuple of (participant_i, iteration, params, env_data)

        Returns:
            participant_i: participant index
            iteration: iteration index
            data: np.ndarray of shape (n_trials, 6)
        """
        participant_i, iteration, params, env_data = args
        data = self.individual_simulate(params, env_data)
        return participant_i, iteration, data

    def simulate(self, n_participants: int, n_iters: int) -> np.ndarray :
        """
        Simulate the RL model for the given number of participants and iterations

        Args:
            n_participants: number of participants
            n_iters: number of iterations

        Returns:
            data: np.ndarray of shape (n_participants * n_iters, n_trials, 6)
        """
        inputs = []
        for participant_i in range(n_participants):
            params = self.sample_params()
            for iteration in range(n_iters):
                env_data = self.env.generate_trials()
                inputs.append((participant_i, iteration, params, env_data))

        with Pool() as pool:
            results = pool.map(self.parallel_simulate, inputs)

        data = np.zeros((n_participants, n_iters, env_data.shape[0], 6))
        for participant_i, iteration, this_data in results:
            data[participant_i][iteration] = this_data

        return data.reshape(n_participants * n_iters, env_data.shape[0], 6)

    def nllh(self, params: list[float], data: np.ndarray) -> float:
        """
        Compute the negative log-likelihood for the given parameters and data

        Args:
            params: list of parameters
            data: environment data

        Returns:
            nllh: negative log-likelihood
        """
        self.parse_params(params)
        n_trials = data.shape[0]
        llh = 0
        for t in range(n_trials):
            state, action, reward = int(data[t, 2]), int(data[t, 4]), int(data[t, 5])
            policy = self.softmax_policy(state)
            llh += np.log(policy[action]) 
            self.update_q_value(state, action, reward)
        return -llh


class StickyRLModel(RLModel):
    """
    StickyRLModel class
    """
    def __init__(self, env: Environment, param_names: list[str], param_bounds: list[list[float, float]]):
        """
        Initialize the StickyRLModel

        Args:
            env: environment
            param_names: list of parameter names
            param_bounds: list of parameter bounds
        """
        super().__init__(env, param_names, param_bounds)

    def softmax_policy(self, state: int, stick_side: np.ndarray) -> np.ndarray:
        """
        Get the softmax policy for the given state and stick side

        Args:
            state: state
            stick_side: stick side

        Returns:
            policy: softmax policy
        """
        q_values = self.q_table[state]
        return softmax(self.params["beta"] * (q_values + self.params["stickiness"] * stick_side))
    
    def sample_action(self, state: int, stick_side: np.ndarray) -> int:
        """
        Sample the action from the softmax policy for the given state and stick side

        Args:
            state: state
            stick_side: stick side

        Returns:
            action: action
        """
        policy = self.softmax_policy(state, stick_side)
        return np.random.choice(self.n_actions, p=policy)

    def individual_simulate(self, params: list[float], env_data: np.ndarray) -> np.ndarray:
        """
        Simulate the RL model for the given parameters and environment data

        Args:
            params: list of parameters
            env_data: environment data

        Returns:
            data: np.ndarray of shape (n_trials, 6)
            data[:, 0]: episode index
            data[:, 1]: trial index
            data[:, 2]: state
            data[:, 3]: correct action
            data[:, 4]: action
            data[:, 5]: reward
        """
        self.parse_params(params)
        n_trials = env_data.shape[0]
        data = np.zeros((n_trials, 6))
        data[:, :4] = env_data[:, :4].copy()

        stick_side = np.zeros(self.n_actions)
        for t in range(n_trials):
            state = int(data[t, 2])
            action = self.sample_action(state, stick_side)
            correct_action = int(data[t, 3])
            reward = self.env.get_reward(state, action)
            self.update_q_value(state, action, reward)
            stick_side = np.zeros(self.n_actions)
            stick_side[action] = 1
            data[t, 3:] = [correct_action, action, reward]
        return data

    def nllh(self, params: list[float], data: np.ndarray) -> float:
        """
        Compute the negative log-likelihood for the given parameters and data

        Args:
            params: list of parameters
            data: environment data

        Returns:
            nllh: negative log-likelihood
        """
        self.parse_params(params)
        n_trials = data.shape[0]
        llh = 0
        stick_side = np.zeros(self.n_actions)
        for t in range(n_trials):
            state, action, reward = int(data[t, 2]), int(data[t, 4]), int(data[t, 5])
            policy = self.softmax_policy(state, stick_side)
            llh += np.log(policy[action]) 
            self.update_q_value(state, action, reward)
            stick_side = np.zeros(self.n_actions)
            stick_side[action] = 1
        return -llh


class Optimizer:
    def optimize(self, agent: RLModel, data: np.ndarray, method: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Optimize the given RL model for the given data using the specified method

        Args:
            agent: RL model
            data: environment data
            method: optimization method. Supported methods include "mle".

        Returns:
            best_params: best parameters
            best_nllh: best negative log-likelihood
        """
        if method.lower() == "mle":
            bounds = [agent.param_bounds[param_name] for param_name in agent.param_names]
            result = differential_evolution(agent.nllh, bounds=bounds, args=(data,))
            best_params = result.x
            best_nllh = agent.nllh(best_params, data)
            return best_params, best_nllh
        else:
            raise ValueError(f"Method {self.method} not supported")

    def parallel_optimizer(self, args) -> tuple[int, int, np.ndarray, float]:
        """
        Optimize the given RL model for the given data using the specified method in parallel

        Args:
            args: tuple of (model_i, participant_i, data, agent, optimization_method)

        Returns:
            model_i: model index
            participant_i: participant index
            best_params: best parameters
            best_nllh: best negative log-likelihood
        """
        model_i, participant_i, data, agent, optimization_method = args
        best_params, best_nllh = self.optimize(agent, data, optimization_method)
        return model_i, participant_i, best_params, best_nllh

    def fit(self, env: Environment, data: np.ndarray, model_names: list[str], param_names: list[list[str]], param_bounds: list[list[list[float, float]]], optimization_method: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Fit the given RL model for the given data using the specified method

        Args:
            env: environment
            data: data
            model_names: list of model names
            param_names: list of parameter names (list of lists; each list contains the parameter names for a model)
            param_bounds: list of parameter bounds (list of lists; each list contains the parameter bounds for a model)
            optimization_method: optimization method. Supported methods include "mle".

        Returns:
            best_params_all: best parameters for all models and participants
            best_nllh_all: best negative log-likelihood for all models and participants
        """
        n_participants = data.shape[0]
        inputs = []
        for model_i, model_name in enumerate(model_names):
            agent = globals()[model_name](env, param_names[model_i], param_bounds[model_i])
            for participant_i in range(n_participants):
                inputs.append((model_i, participant_i, data[participant_i], agent, optimization_method))

        print(f"Fitting {len(model_names)} models with {n_participants} participants using {optimization_method} method")

        with Pool() as pool:
            n_processes = pool._processes
            print(f"Using at most {n_processes} parallel processes")
            results = pool.map(self.parallel_optimizer, inputs)
        
        # parse results
        max_n_params = max([len(params) for params in param_names])
        best_params_all = np.full((len(model_names), n_participants, max_n_params), np.nan)
        best_nllh_all = np.zeros((len(model_names), n_participants))
        for model_i, participant_i, best_params, best_nllh in results:
            best_params_all[model_i][participant_i][:len(best_params)] = best_params
            best_nllh_all[model_i][participant_i] = best_nllh

        return best_params_all, best_nllh_all

    def compute_fit_metric(self, nllh: np.ndarray, n_params: list[int], data: np.ndarray, metric: str) -> np.ndarray:
        """
        Compute the fit metric for the given negative log-likelihood, number of parameters, and data

        Args:
            nllh: negative log-likelihood
            n_params: number of parameters
            data: data
            metric: fit metric. Supported metrics include "aic" and "bic".

        Returns:
            fit_metric: fit metric
        """
        n_trials = data.shape[1]
        if metric.lower() == "aic":
            return 2 * nllh + 2 * n_params[:,np.newaxis]
        elif metric.lower() == "bic":
            return 2 * nllh + np.log(n_trials) * n_params[:,np.newaxis]
        else:
            raise ValueError(f"Metric {metric} not supported")
