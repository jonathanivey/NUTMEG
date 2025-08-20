__all__ = ["NUTMEG"]

from typing import Any, Iterator, List, Optional, Tuple, Union

import attr
import numpy as np
import pandas as pd
import scipy.stats as sps
from numpy.typing import NDArray
from scipy.special import digamma
from tqdm.auto import tqdm, trange

from NUTMEG.base import BaseClassificationAggregator


def normalize(x: NDArray[np.float64], smoothing: float) -> NDArray[np.float64]:
    """Normalizes the rows of the matrix using the smoothing parameter.

    Args:
        x (np.ndarray): The array to normalize.
        smoothing (float): The smoothing parameter.

    Returns:
        np.ndarray: Normalized array
    """
    norm = (x + smoothing).sum(axis=1)
    return np.divide(
        x + smoothing,
        norm[:, np.newaxis],
        out=np.zeros_like(x),
        where=~np.isclose(norm[:, np.newaxis], np.zeros_like(norm[:, np.newaxis])),
    )


def variational_normalize(
    x: NDArray[np.float64], hparams: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Normalizes the rows of the matrix using the NUTMEG priors.

    Args:
        x (np.ndarray): The array to normalize.
        hparams (np.ndarray): The prior parameters.

    Returns:
        np.ndarray: Normalized array
    """
    norm = (x + hparams).sum(axis=1)
    norm = np.exp(digamma(norm))
    return np.divide(
        np.exp(digamma(x + hparams)),
        norm[:, np.newaxis],
        out=np.zeros_like(x),
        where=~np.isclose(norm[:, np.newaxis], np.zeros_like(norm[:, np.newaxis])),
    )


def decode_distribution(gold_label_marginals: NDArray[np.float64]) -> pd.DataFrame:
    """Decodes the distribution from marginals.

    Args:
        gold_label_marginals (pd.DataFrame): Gold label marginals.

    Returns:
        pd.DataFrame: Decoded distribution
    """

    # return gold_label_marginals.div(gold_label_marginals.sum(axis=1), axis=0)
    return gold_label_marginals / gold_label_marginals.sum(axis=2, keepdims=True)

        
        


@attr.s
class NUTMEG(BaseClassificationAggregator):

    n_restarts: int = attr.ib(default=10)
    """The number of optimization runs of the algorithms.
    The final parameters are those that gave the best log likelihood.
    If one run takes too long, this parameter can be set to 1."""

    n_iter: int = attr.ib(default=50)
    """The maximum number of EM iterations for each optimization run."""

    method: str = attr.ib(default="vb")
    """The method which is used for the M-step. Either 'vb' or 'em'.
    'vb' means optimization with Variational Bayes using priors.
    'em' means standard Expectation-Maximization algorithm."""

    smoothing: float = attr.ib(default=0.1)
    """The smoothing parameter for the normalization."""

    default_noise: float = attr.ib(default=0.5)
    """The default noise parameter for the initialization."""

    alpha: float = attr.ib(default=0.5)
    """The prior parameter for the Beta distribution on $\theta_j$."""

    beta: float = attr.ib(default=0.5)
    """The prior parameter for the Beta distribution on $\theta_j$."""

    random_state: int = attr.ib(default=0)
    """The state of the random number generator."""

    verbose: int = attr.ib(default=0)
    """Specifies if the progress will be printed or not:
    0 — no progress bar, 1 — only for restarts, 2 — for both restarts and optimization."""

    spamming_: NDArray[np.float64] = attr.ib(init=False)
    """The posterior distribution of workers' spamming states."""

    thetas_: NDArray[np.float64] = attr.ib(init=False)
    """The posterior distribution of workers' spamming labels."""

    theta_priors_: Optional[NDArray[np.float64]] = attr.ib(init=False)
    r"""The prior parameters for the Beta distribution on $\theta_j$."""

    strategy_priors_: Optional[NDArray[np.float64]] = attr.ib(init=False)
    r"""The prior parameters for the Diriclet distribution on $\xi_j$."""

    smoothing_: float = attr.ib(init=False)
    """The smoothing parameter."""

    probas_: Optional[NDArray[np.float64]] = attr.ib(init=False)
    """The probability distributions of task labels.
    The `pandas.DataFrame` data is indexed by `task` so that `result.loc[task, label]` is the probability that
    the `task` true label is equal to `label`. Each probability is in the range from 0 to 1,
    all task probabilities must sum up to 1."""

    labels_: Optional[NDArray[np.float64]] = attr.ib(init=False)

    def fit(self, data: pd.DataFrame, return_unobserved=True) -> "NUTMEG":
        """Fits the model to the training data.

        Args:
            data (DataFrame): The training dataset of workers' labeling results
                which is represented as the `pandas.DataFrame` data containing `task`, `worker`, and `label` columns.

        Returns:
            NUTMEG: The fitted NUTMEG model.
        """

        workers, worker_names = pd.factorize(data["worker"])
        labels, label_names = pd.factorize(data["label"].astype(str))
        tasks, task_names = pd.factorize(data["task"])
        subpops, subpop_names = pd.factorize(data["subpopulation"])

        # calculate the proportion of each subpopulation for each task
        counts = np.zeros((len(task_names), len(subpop_names)), dtype=float)
        np.add.at(counts, (tasks, subpops), 1)
        proportions = counts / counts.sum(axis=1, keepdims=True)

        n_workers = len(worker_names)
        n_labels = len(label_names)

        self.smoothing_ = 0.01 / n_labels

        annotation = data.copy(deep=True)

        best_log_marginal_likelihood = -np.inf

        def restarts_progress() -> Iterator[int]:
            if self.verbose > 0:
                yield from trange(self.n_restarts, desc="Restarts")
            else:
                yield from range(self.n_restarts)

        for _ in restarts_progress():
            self._initialize(n_workers, n_labels)
            (
                log_marginal_likelihood,
                gold_label_marginals,
                strategy_expected_counts,
                knowing_expected_counts,
            ) = self._e_step(
                annotation,
                task_names,
                worker_names,
                label_names,
                subpop_names,
                tasks,
                workers,
                labels,
                subpops,
                proportions,
            )

            def iteration_progress() -> Tuple[Iterator[int], Optional["tqdm[int]"]]:
                if self.verbose > 1:
                    trange_ = trange(self.n_iter, desc="Iterations")
                    return iter(trange_), trange_
                else:
                    return iter(range(self.n_iter)), None

            iterator, pbar = iteration_progress()

            for _ in iterator:
                if self.method == "vb":
                    self._variational_m_step(
                        knowing_expected_counts, strategy_expected_counts
                    )
                else:
                    self._m_step(knowing_expected_counts, strategy_expected_counts)
                (
                    log_marginal_likelihood,
                    gold_label_marginals,
                    strategy_expected_counts,
                    knowing_expected_counts,
                ) = self._e_step(
                    annotation,
                    task_names,
                    worker_names,
                    label_names,
                    subpop_names,
                    tasks,
                    workers,
                    labels,
                    subpops,
                    proportions,
                )
                if self.verbose > 1:
                    assert isinstance(pbar, tqdm)
                    pbar.set_postfix(
                        {"log_marginal_likelihood": round(log_marginal_likelihood, 5)}
                    )
            if log_marginal_likelihood > best_log_marginal_likelihood:
                best_log_marginal_likelihood = log_marginal_likelihood
                best_thetas = self.thetas_.copy()
                best_spamming = self.spamming_.copy()

        self.thetas_ = best_thetas
        self.spamming_ = best_spamming
        _, gold_label_marginals, _, _ = self._e_step(
            annotation, task_names, worker_names, label_names, subpop_names, tasks, workers, labels, subpops, proportions,
        )

        # calculate marginals for subpopulations that have not labeled an item
        gold_label_marginals = self.predict_unobserved_instances(data, task_names, label_names, subpop_names, gold_label_marginals, return_unobserved)

        # we need a way to indicate which index of our output corresponds to each label
        self.label_key = label_names.values
        self.probas_ = decode_distribution(gold_label_marginals)
        self.labels_ = self.label_key[np.argmax(gold_label_marginals, axis=2)]
        
        if not return_unobserved:
            self.labels_[np.all(np.isnan(gold_label_marginals), axis=2)] = np.nan

        return self

    def fit_predict(self, data: pd.DataFrame, return_unobserved=True) -> "pd.Series[Any]":
        """
        Fits the model to the training data and returns the aggregated results.

        Args:
            data (DataFrame): The training dataset of workers' labeling results
                which is represented as the `pandas.DataFrame` data containing `task`, `worker`, and `label` columns.

        Returns:
            Series: Task labels. The `pandas.Series` data is indexed by `task`
                so that `labels.loc[task]` is the most likely true label of tasks.
        """
        self.fit(data, return_unobserved)
        assert self.labels_ is not None, "no labels_"
        return self.labels_

    def fit_predict_proba(self, data: pd.DataFrame, return_unobserved=True) -> pd.DataFrame:
        """
        Fits the model to the training data and returns probability distributions of labels for each task.

        Args:
            data (DataFrame): The training dataset of workers' labeling results
                which is represented as the `pandas.DataFrame` data containing `task`, `worker`, and `label` columns.

        Returns:
            DataFrame: Probability distributions of task labels.
                The `pandas.DataFrame` data is indexed by `task` so that `result.loc[task, label]` is the probability that the `task` true label is equal to `label`.
                Each probability is in the range from 0 to 1, all task probabilities must sum up to 1.
        """
        self.fit(data, return_unobserved)
        assert self.probas_ is not None, "no probas_"
        return self.probas_

    def _initialize(self, n_workers: int, n_labels: int) -> None:
        """Initializes the NUTMEG parameters.

        Args:
            n_workers (int): The number of workers.
            n_labels (int): The number of labels.

        Returns:
            None
        """

        self.spamming_ = sps.uniform(1, 1 + self.default_noise).rvs(
            size=(n_workers, 2),
            random_state=self.random_state,
        )
        self.thetas_ = sps.uniform(1, 1 + self.default_noise).rvs(
            size=(n_workers, n_labels), random_state=self.random_state
        )

        self.spamming_ = self.spamming_ / self.spamming_.sum(axis=1, keepdims=True)
        self.thetas_ = self.thetas_ / self.thetas_.sum(axis=1, keepdims=True)

        if self.method == "vb":
            self.theta_priors_ = np.empty((n_workers, 2))
            self.theta_priors_[:, 0] = self.alpha
            self.theta_priors_[:, 1] = self.beta

            self.strategy_priors_ = np.ones((n_workers, n_labels)) * 10.0

    def _e_step(
        self,
        annotation: pd.DataFrame,
        task_names: Union[List[Any], "pd.Index[Any]"],
        worker_names: Union[List[Any], "pd.Index[Any]"],
        label_names: Union[List[Any], "pd.Index[Any]"],
        subpop_names: Union[List[Any], "pd.Index[Any]"],
        tasks: NDArray[np.int64],
        workers: NDArray[np.int64],
        labels: NDArray[np.int64],
        subpops: NDArray[np.int64],
        proportions: NDArray[np.float64],
    ) -> Tuple[float, pd.DataFrame, NDArray[np.float64], pd.DataFrame]:
        """Performs E-step of the NUTMEG algorithm.

        Args:
            annotation (DataFrame): The workers' labeling results. The `pandas.DataFrame` data contains `task`, `worker`, and `label` columns.
            task_names (List[Any]): The task names.
            worker_names (List[Any]): The workers' names.
            label_names (List[Any]): The label names.
            tasks (np.ndarray): The task IDs in the annotation.
            workers (np.ndarray): The workers' IDs in the annotation.
            labels (np.ndarray): The label IDs in the annotation.

        Returns:
            Tuple[float, pd.DataFrame, pd.DataFrame, pd.DataFrame]: The log marginal likelihood, gold label marginals,
                strategy expected counts, and knowing expected counts.
        """

        gold_label_marginals = np.zeros((len(task_names), len(subpop_names), len(label_names)))

        knowing_expected_counts = pd.DataFrame(
            np.zeros((len(worker_names), 2)),
            index=worker_names,
            columns=["knowing_expected_count_0", "knowing_expected_count_1"],
        )

        for label_idx, label in enumerate(label_names):
            annotation["gold_marginal"] = self.spamming_[workers, 0] * self.thetas_[
                workers, labels
            ] + self.spamming_[workers, 1] * (label_idx == labels)

            for subpop_idx, subpop in enumerate(subpop_names):
                
                # if a subpopulation is not present, temporarily set its gold marginals to nan
                gold_label_marginals[:, subpop_idx, label_idx] = annotation[annotation['subpopulation']==subpop].groupby("task").prod(
                    numeric_only=True)["gold_marginal"].reindex(annotation["task"].unique(), fill_value=1e-8) / len(label_names)
                    # numeric_only=True)["gold_marginal"].reindex(annotation["task"].unique(), fill_value=np.nan) / len(label_names)

        # for all tasks where a subpopulation's annotation is unavailable

        
        # instance_marginals = gold_label_marginals.sum(axis=1)
        instance_marginals = (gold_label_marginals.sum(axis=2) * proportions).sum(axis=1)
        log_marginal_likelihood = np.log(instance_marginals + 1e-8).sum()

        annotation["strategy_marginal"] = 0.0


        for label in range(len(label_names)):
            annotation["strategy_marginal"] += gold_label_marginals[tasks, subpops, label] / (
                self.spamming_[workers, 0] *  self.thetas_[workers, labels]
                + self.spamming_[workers, 1] * (labels == label)
            )

        annotation["strategy_marginal"] = (
            annotation["strategy_marginal"]
            * self.spamming_[workers, 0]
            * self.thetas_[workers, labels]
        )

        # annotation.set_index("task", inplace=True)
        # annotation["instance_marginal"] = instance_marginals
        # annotation.reset_index(inplace=True)

        annotation["instance_marginal"] = instance_marginals[tasks]

        annotation["strategy_marginal"] = (
            annotation["strategy_marginal"] / annotation["instance_marginal"]
        )

        strategy_expected_counts = (
            annotation.groupby(["worker", "label"])
            .sum(numeric_only=True)["strategy_marginal"]
            .unstack()
            .fillna(0.0)
        )

        knowing_expected_counts["knowing_expected_count_0"] = annotation.groupby(
            "worker"
        ).sum(numeric_only=True)["strategy_marginal"]

        # annotation["knowing_expected_counts"] = (
        #     gold_label_marginals.values[tasks, labels].ravel()
        #     * self.spamming_[workers, 1]
        #     / (
        #         self.spamming_[workers, 0] * self.thetas_[workers, labels]
        #         + self.spamming_[workers, 1]
        #     )
        # ) / instance_marginals.values[tasks]
        annotation["knowing_expected_counts"] = (
            gold_label_marginals[tasks, subpops, labels]
            * self.spamming_[workers, 1]
            / (
                self.spamming_[workers, 0] * self.thetas_[workers, labels]
                + self.spamming_[workers, 1]
            )
        ) / instance_marginals[tasks]

        knowing_expected_counts["knowing_expected_count_1"] = annotation.groupby(
            "worker"
        ).sum(numeric_only=True)["knowing_expected_counts"]

        return (
            log_marginal_likelihood,
            gold_label_marginals,
            strategy_expected_counts,
            knowing_expected_counts,
        )

    def _m_step(
        self,
        knowing_expected_counts: pd.DataFrame,
        strategy_expected_counts: pd.DataFrame,
    ) -> None:
        """
        Performs M-step of the NUTMEG algorithm.

        Args:
            knowing_expected_counts (DataFrame): The knowing expected counts.
            strategy_expected_counts (DataFrame): The strategy expected counts.

        Returns:
            None
        """
        self.spamming_ = normalize(knowing_expected_counts.values, self.smoothing_)
        self.thetas_ = normalize(strategy_expected_counts.values, self.smoothing_)

    def _variational_m_step(
        self,
        knowing_expected_counts: pd.DataFrame,
        strategy_expected_counts: pd.DataFrame,
    ) -> None:
        """
        Performs variational M-step of the NUTMEG algorithm.

        Args:
            knowing_expected_counts (DataFrame): The knowing expected counts.
            strategy_expected_counts (DataFrame): The strategy expected counts.

        Returns:
            None
        """
        assert self.theta_priors_ is not None
        self.spamming_ = variational_normalize(
            knowing_expected_counts.values, self.theta_priors_
        )
        assert self .strategy_priors_ is not None
        self.thetas_ = variational_normalize(
            strategy_expected_counts.values, self.strategy_priors_
        )

    def predict_unobserved_instances(self,
        data: pd.DataFrame,
        task_names: Union[List[Any], "pd.Index[Any]"],
        label_names: Union[List[Any], "pd.Index[Any]"],
        subpop_names: Union[List[Any], "pd.Index[Any]"],
        gold_label_marginals: NDArray[np.float64],
        return_unobserved=True
        ) -> NDArray[np.float64]:

        """
        Calculates predictions for unobserved subpopulations based on Bayes rule. Assumes independence of subpopulation labels

            Args:
                data (DataFrame): The training dataset of workers' labeling results
                    which is represented as the `pandas.DataFrame` data containing `task`, `worker`, and `label` columns.
                task_names (List[Any]): The task names.
                label_names (List[Any]): The label names.
                gold__label_marginals

            Returns:
                gold_label_marginals
        """

        # start by predicting labels
        prelim_labels = np.argmax(gold_label_marginals, axis=2)

        # observed_labeled_instances[i][j] where i indicates the subpopulation observed and j indicates the label predicted
        observed_prediction_totals = [[None] * len(label_names)] * len(subpop_names)
        observed_prediction_sizes = [[None] * len(label_names)] * len(subpop_names)

        for i in range (len(subpop_names)):

            # only want instances where the subpopulation is observed
            subpop_presence = np.array(data.groupby('task')['subpopulation'].unique().apply(lambda x: i in x))

            for j in range(len(label_names)):
                
                # instances where the subpopulation gave the label j
                observed_prediction_totals[i][j] = gold_label_marginals[(prelim_labels[:, i] == j) * subpop_presence].sum(axis=0)
                observed_prediction_sizes[i][j] = ((prelim_labels[:, i] == j) * subpop_presence).sum()


        # determine which subpopulations were observed
        obs_subpops = data.groupby('task')['subpopulation'].unique().apply(set)

        # find all tasks that have unobserved subpopulations
        unobs_subpops = set(subpop_names) - obs_subpops
        unobs_subpops = unobs_subpops[unobs_subpops.apply(bool)]


        # for every task with unobserved subpopulations
        for task, unobs in unobs_subpops.items():

            task_idx = np.where(task_names==task)

            obs_total = np.zeros(observed_prediction_totals[0][0].shape)
            obs_size = 0

            # for each subpopulation that was observed
            for obs_subpop in obs_subpops[task]:
                
                obs_subpop_idx = np.where(subpop_names==obs_subpop)[0][0]

                # identify what label the observed subgroup is predicted to have assigned
                obs_subpop_label_idx = prelim_labels[task_idx, obs_subpop_idx][0][0]

                
                # create a list of all observed subpopulations and what they labeled
                obs_total += observed_prediction_totals[obs_subpop_idx][obs_subpop_label_idx]
                obs_size += observed_prediction_sizes[obs_subpop_idx][obs_subpop_label_idx]

            # calculate the expected gold marginal based on observations
            obs_prediction = obs_total / obs_size

            # for every unobserved subpopulation
            for unobs_subpop in unobs:

                unobs_subpop_idx = np.where(subpop_names==unobs_subpop)[0][0]
                
                if return_unobserved:
                    # set gold marginals to the values from our prediction
                    gold_label_marginals[task_idx, unobs_subpop_idx] = obs_prediction[unobs_subpop_idx]
                else:
                    gold_label_marginals[task_idx, unobs_subpop_idx] = np.nan

        return gold_label_marginals
