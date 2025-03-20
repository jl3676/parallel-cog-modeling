import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import src.helpers as helpers
import src.modeling as modeling

def plot_switch_learning_curve(data: np.ndarray, n_trials_to_plot: int=5):
    """
    Plot the learning curve around switches

    Args:
        data: data
        n_trials_to_plot: number of trials to plot on each side of the switch
    """
    switch_data_all = np.full((data.shape[0], 2 * n_trials_to_plot + 1), np.nan)
    for participant_idx in range(data.shape[0]):
        participant_data = data[participant_idx]
        correct_actions = helpers.get_correct_actions(participant_data)
        rewards = helpers.get_rewards(participant_data)
        switch_trials = np.where(np.diff(correct_actions) != 0)[0] + 1
        switch_data = np.full((len(switch_trials), 2 * n_trials_to_plot + 1), np.nan)
        for switch_idx, switch_trial in enumerate(switch_trials):
            switch_data[switch_idx, :] = rewards[switch_trial - n_trials_to_plot:switch_trial + n_trials_to_plot + 1]
        switch_data = switch_data.mean(axis=0)
        switch_data_all[participant_idx] = switch_data

    mean_switch_data = switch_data_all.mean(axis=0)
    sem_switch_data = stats.sem(switch_data_all, axis=0)
    plt.plot(mean_switch_data)
    plt.fill_between(range(mean_switch_data.shape[0]), mean_switch_data - sem_switch_data, mean_switch_data + sem_switch_data, alpha=0.2)
    plt.xticks(range(0, 2 * n_trials_to_plot + 1, n_trials_to_plot), range(-n_trials_to_plot, n_trials_to_plot + 1, n_trials_to_plot))
    plt.ylim(0, 1)
    plt.xlabel("Trial from switch")
    plt.ylabel("Reward")
    plt.title(f"Learning curve around switches (n={data.shape[0]})")
    plt.show()

def plot_fit_metric(data: np.ndarray, optimizer: modeling.Optimizer, best_nllh_all: np.ndarray, model_names: list[str], param_names: list[list[str]], fit_metrics: list[str]):
    """
    Plot the fit metric

    Args:
        data: data
        optimizer: optimizer
        best_nllh_all: best negative log-likelihood for all models and participants
        model_names: model names
        param_names: parameter names
        fit_metrics: fit metrics. Supported metrics include "AIC" and "BIC".
    """
    n_params = np.array([len(param_names[0]), len(param_names[1])])
    colors = ['cornflowerblue', 'hotpink']

    # Plot results
    plt.figure(figsize=(10, 6))
    for metric_i, metric in enumerate(fit_metrics):
        plt.subplot(1, 2, metric_i + 1)
        scores = optimizer.compute_fit_metric(best_nllh_all, n_params, data, metric)
        mean_scores = np.mean(scores, axis=1)
        sem_scores = np.std(scores, axis=1) / np.sqrt(scores.shape[1])
        plt.bar(model_names, mean_scores, color=colors, yerr=sem_scores, capsize=10, alpha=0.3)

        xs = []
        for m in range(len(model_names)):
            x = m + np.random.normal(0, 0.05, size=scores[m].shape)
            xs.append(x)
            plt.scatter(x, scores[m], linewidth=0, alpha=0.9, color=colors[m])

        for i in range(len(xs[0])):
            plt.plot([xs[0][i], xs[1][i]], [scores[0][i], scores[1][i]], color='k', alpha=0.1)

        # do paired wilcoxon signed rank test
        for m1 in range(len(model_names)):
            for m2 in range(m1 + 1, len(model_names)):
                _, p = stats.wilcoxon(scores[m1], scores[m2])
                if p < 0.05:
                    print(f"{model_names[m1]} and {model_names[m2]} fit significantly different by {metric} (p={p:.3f}).")

        plt.xlabel('Model')
        plt.ylabel('Score')
        plt.title(f'{metric}')
    plt.tight_layout()
    plt.show()