"""
Implements Bayesian Optimization Hyperband (BOHB) for PIE.

Sources:
Hyperband: https://arxiv.org/pdf/1603.06560.pdf
BOHB: https://arxiv.org/pdf/1801.01596.pdf
Bayesian dropout: https://arxiv.org/ftp/arxiv/papers/1802/1802.05400.pdf
Threshold: https://ijai.iaescore.com/index.php/IJAI/article/view/24324
"""

# Standard library
import os
import json
from typing import Callable
import uuid
import math
import traceback
import random
import functools

# Third-party
from bayes_opt import BayesianOptimization, UtilityFunction

# Local
from pie import utils
from pie.settings import check_settings, merge_task_defaults
from pie.settings import Settings

utility = UtilityFunction(kind="ucb", kappa=2.5, xi=0.0)

# We can't use -inf as a threshold, because the GP can't handle it.
INFINITY = -100

print = functools.partial(print, flush=True)


class PieSpace:
    """
    We use the bayes_opt library to generate configurations in the hyperparameter space.
    Bayes_opt only support floats, so we need to convert them so Pie understands them.
    Bayes_opt however works with an inclusive range: asking for range (0, 1) can genuinely generate 0 or 1.
    We would have needed to convert them anyway, because Pie can have nested dictionary hyperparameters (e.g. lm_schedule).
    """

    @staticmethod
    def to_choice(value, options) -> str:
        return options[math.floor(value)]

    @staticmethod
    def to_int(value) -> int:
        return math.floor(value)

    @staticmethod
    def to_bool(value) -> bool:
        # Round (0,1) instead of flooring (0,2), because of the inclusive range.
        return bool(round(value))

    @staticmethod
    def bool_range():
        return (0, 1)

    @staticmethod
    def choice_range(n):
        n_with_error = n - 0.000001
        return (0, n_with_error)

    @staticmethod
    def get_bounds() -> dict:
        """
        Defines the bounds of the hyperparameter space for bayes_opt.
        """
        # Note: for choice and int, add 1 to the upper bound
        return {
            "include_lm": PieSpace.bool_range(),
            "lm_shared_softmax": PieSpace.bool_range(),
            "lm_patience": (1, 5),  # int
            "lm_factor": (0, 1),  # float
            "lm_weight": (0, 1),  # float
            "batch_size": (10, 100),  # int
            "pretrain_embeddings": PieSpace.bool_range(),
            "freeze_embeddings": PieSpace.bool_range(),
            "dropout": (0, 1),  # float
            "word_dropout": (0, 1),  # float
            "optimizer": PieSpace.choice_range(11),  # choice
            "clip_norm": (0, 10),  # float
            "lr": (0, 0.1),  # float
            "lr_factor": (0, 1),  # float
            "min_lr": (0, 0.0001),  # float
            "lr_patience": (0, 5),  # int
            "wemb_dim": (50, 500),  # int
            "cemb_dim": (50, 500),  # int
            "cemb_type": PieSpace.choice_range(2),  # choice
            "custom_cemb_cell": PieSpace.bool_range(),
            "cemb_layers": (1, 5),  # int
            "scorer": PieSpace.choice_range(3),  # choice
            "linear_layers": (1, 4),  # int
            "hidden_size": (10, 500),  # int
            "num_layers": (1, 4),  # int
            "cell": PieSpace.choice_range(2),  # choice
            "init_rnn": PieSpace.choice_range(3),  # choice
        }

    @staticmethod
    def optimizers() -> list[str]:
        return [
            "Adadelta",
            "Adagrad",
            "Adam",
            "AdamW",
            "Adamax",
            "ASGD",
            "SGD",
            "RAdam",
            "Rprop",
            "RMSprop",
            "NAdam",
        ]

    @staticmethod
    def to_pie_config(t):
        pie_config = {}
        pie_config["include_lm"] = PieSpace.to_bool(t["include_lm"])
        pie_config["lm_shared_softmax"] = PieSpace.to_bool(t["lm_shared_softmax"])
        pie_config["lm_schedule"] = {}
        pie_config["lm_schedule"]["patience"] = PieSpace.to_int(t["lm_patience"])
        pie_config["lm_schedule"]["factor"] = t["lm_factor"]
        pie_config["lm_schedule"]["weight"] = t["lm_weight"]
        pie_config["batch_size"] = PieSpace.to_int(t["batch_size"])
        pie_config["pretrain_embeddings"] = PieSpace.to_bool(t["pretrain_embeddings"])
        pie_config["freeze_embeddings"] = PieSpace.to_bool(t["freeze_embeddings"])
        pie_config["dropout"] = t["dropout"]
        pie_config["word_dropout"] = t["word_dropout"]
        pie_config["optimizer"] = PieSpace.to_choice(
            t["optimizer"], PieSpace.optimizers()
        )
        pie_config["clip_norm"] = t["clip_norm"]
        pie_config["lr"] = t["lr"]
        pie_config["lr_factor"] = t["lr_factor"]
        pie_config["min_lr"] = t["min_lr"]
        pie_config["lr_patience"] = PieSpace.to_int(t["lr_patience"])
        pie_config["wemb_dim"] = PieSpace.to_int(t["wemb_dim"])
        pie_config["cemb_dim"] = PieSpace.to_int(t["cemb_dim"])
        pie_config["cemb_type"] = PieSpace.to_choice(t["cemb_type"], ["rnn", "cnn"])
        pie_config["custom_cemb_cell"] = PieSpace.to_bool(t["custom_cemb_cell"])
        pie_config["cemb_layers"] = PieSpace.to_int(t["cemb_layers"])
        pie_config["scorer"] = PieSpace.to_choice(
            t["scorer"], ["general", "dot", "bahdanau"]
        )
        pie_config["linear_layers"] = PieSpace.to_int(t["linear_layers"])
        pie_config["hidden_size"] = PieSpace.to_int(t["hidden_size"])
        pie_config["num_layers"] = PieSpace.to_int(t["num_layers"])
        pie_config["cell"] = PieSpace.to_choice(t["cell"], ["LSTM", "GRU"])
        pie_config["init_rnn"] = PieSpace.to_choice(
            t["init_rnn"], ["default", "xavier_uniform", "orthogonal"]
        )
        return pie_config


class BOHB:
    """
    gp_opt_only:
    I.e. for each bracket perform the first halving while always registering results,
    making the GP non-random. In other havlings, only choose the top k=1 config and exploit it.

    enable_bayes_dropout:
    Only ever optimize a subset of the hyperparameters, and fill in the rest with the best candidate.

    warm_start:
    The first candidate is a hardcoded known to be good candidate, and the rest are generated by the GP.
    """

    def __init__(
        self,
        train_fn,
        settings,
        max_indiv_budget: int,
        eta,
        enable_bayes_dropout,
        gp_opt_only,
        warm_start,
        kwargs,
    ):
        self.eta = eta
        self.num_brackets = math.floor(math.log(max_indiv_budget, eta))
        # convert individual budget to total bracket budget
        self.max_indiv_budget = max_indiv_budget
        self.total_budget = (self.num_brackets + 1) * max_indiv_budget
        self.best_candidate: Candidate = None  # type: ignore
        self.candidates: list[Candidate] = []
        self.history: dict[str, Candidate] = {}
        self.random_bayes = BayesianOptimization(
            f=None,
            pbounds=PieSpace.get_bounds(),
            verbose=2,
        )
        self.bayes = BayesianOptimization(
            f=None,
            pbounds=PieSpace.get_bounds(),
            allow_duplicate_points=True,
        )
        self.loss_threshold = INFINITY
        self.settings = settings
        self.train_fn = train_fn
        self.kwargs = kwargs
        self.enable_bayes_dropout = enable_bayes_dropout
        self.gp_opt_only = gp_opt_only
        self.warm_start = warm_start
        # Number of dimensions kept in dropout bayes
        self.dropout_dims = 9

    def run(self):
        # the hyperband convention is to go reverse.
        for bracket in reversed(range(self.num_brackets + 1)):
            self.bracket = bracket
            # Reset candidates for each bracket.
            self.candidates = []
            # Number of candidates decreases with bracket number.
            num_candidates = math.ceil(
                int(self.total_budget / self.max_indiv_budget / (bracket + 1))
                * self.eta**bracket
            )
            # Minimum budget for each candidate increases with bracket number.
            min_indiv_budget = math.ceil(
                self.max_indiv_budget * (self.eta ** (-bracket))
            )

            # run
            print(f"+++ Bracket {bracket} with {num_candidates} candidates +++")
            self.run_bracket(num_candidates, min_indiv_budget)
            print(f"+++ Bracket {bracket} finished +++")
            # New best?
            # Candidates are sorted at this point.
            if (
                self.best_candidate is None
                or self.candidates[0].loss > self.best_candidate.loss
            ):
                # delete the old best
                if self.best_candidate is not None:
                    self.best_candidate.delete()
                self.best_candidate = self.candidates[0]
                print(
                    f"+++ New best found in bracket {bracket}: {self.best_candidate.loss} +++"
                )
            else:
                # wasnt the best so delete it
                print(
                    f"+++ Top candidate of bracket {bracket} was not the best: {self.candidates[0].loss}"
                )
                self.candidates[0].delete()

            if self.gp_opt_only:
                break

        # final best
        self.best_candidate.save_metadata()
        print(
            f"+++ Final best: {self.best_candidate.loss} [id: {self.best_candidate.pie_config['config_uuid']}] +++"
        )

    def run_bracket(self, num_bracket_candidates, min_indiv_budget):
        # number of halvings is equal to the bracket number.
        for halving in range(self.bracket + 1):
            self.halving = halving
            # Each halving has a decreasing number of candidates.
            num_halving_candidates = (
                math.floor(num_bracket_candidates * (self.eta ** (-halving)))
                if halving == 0
                else len(self.candidates)
            )
            # How much budget each candidate gets in this halving.
            indiv_halving_budget = math.ceil(min_indiv_budget * (self.eta**halving))

            self.losses: list[float] = []

            self.run_halving(num_halving_candidates, indiv_halving_budget)

            next_num_halving_candidates = math.floor(num_halving_candidates / self.eta)
            self.top_k(next_num_halving_candidates)

    def run_halving(self, num_halving_candidates, indiv_halving_budget):
        # 1st iter: we initialize the configurations
        # first bracket: random
        # other brackets: modelled by the GP
        for c_i in range(num_halving_candidates):
            print(
                f"+++ Bracket {self.bracket} (h{self.halving + 1}/{self.bracket + 1}) (c{c_i + 1}/{num_halving_candidates}) (b{indiv_halving_budget})+++"
            )

            # Generate a new candidate in the first halving each bracket.
            c = self.new_candidate() if self.halving == 0 else self.candidates[c_i]

            # Run
            c.run(indiv_halving_budget)

            # Register the loss in the GP, except for the first halving of the first bracket.
            if not self.is_very_first_halving():
                self.register(c)

        # At the end of the very first halving of the very first bracket, we register the random configs.
        if self.is_very_first_halving():
            # Only register the configs AFTER we ran them all.
            # Intermediate registration changes the GP, so it won't be random.
            for c in self.candidates:
                self.register(c)

    def new_candidate(self):
        if self.warm_start:
            c = self.warm_candidate()
            self.candidates.append(c)
            if self.enable_bayes_dropout:
                # use the warm candidate as the best candidate
                self.best_candidate = c
            self.warm_start = False
            print(f"+++ Using warm candidate +++")
            return c

        config = {}
        if not self.history:
            print(f"+++ Random candidate +++")
            config = self.random_bayes.suggest(utility)
        else:
            config = (
                self.dropout_bayes()
                if self.enable_bayes_dropout
                else self.bayes.suggest(utility)
            )
        c = Candidate(
            bayes_config=config,
            settings=self.settings,
            train_fn=self.train_fn,
            kwargs=self.kwargs,
        )
        self.candidates.append(c)
        return c

    def warm_candidate(self):
        # hardcoded numeric config compatible with the bayes space
        config = {
            "include_lm": int(True),
            "lm_shared_softmax": int(True),
            "lm_patience": 2,
            "lm_factor": 0.5,
            "lm_weight": 0.2,
            "batch_size": 25,
            "pretrain_embeddings": int(False),
            "freeze_embeddings": int(False),
            "dropout": 0.25,
            "word_dropout": 0,
            "optimizer": 2.5,  # "Adam",
            "clip_norm": 5,
            "lr": 0.001,
            "lr_factor": 0.75,
            "min_lr": 0.000001,
            "lr_patience": 2,
            "wemb_dim": 0,
            "cemb_dim": 300,
            "cemb_type": 0.5,  # "rnn"
            "custom_cemb_cell": int(False),
            "cemb_layers": 2,
            "scorer": 0.5,  # "general",
            "linear_layers": 1,
            "hidden_size": 150,
            "num_layers": 1,
            "cell": 1.5,  # "GRU",
            "init_rnn": 0.5,  # "default",
        }
        return Candidate(
            bayes_config=config,
            settings=self.settings,
            train_fn=self.train_fn,
            kwargs=self.kwargs,
        )

    def dropout_bayes(self) -> dict:
        if random.random() < 0.1:
            print(f"+++ Random candidate +++")
            return self.random_bayes.suggest(utility)
        # choose self.dropout_dims random dimensions
        bounds = PieSpace.get_bounds()
        dims: dict = {
            k: bounds[k] for k in random.sample(bounds.keys(), self.dropout_dims)
        }
        bayes = BayesianOptimization(
            f=None,
            pbounds=dims,
            allow_duplicate_points=True,
        )
        # dedupe history, taking only the best loss
        histories: dict[dict, list[float]] = {}
        for v in self.history.values():
            # only register the dimensions that are in the dropout
            dims_history = {k: v.bayes_config[k] for k in dims}
            tup_history = tuple(sorted(dims_history.items()))
            if tup_history not in histories:
                histories[tup_history] = []
            histories[tup_history].append(v.loss)

        print(f"+++ {len(histories)} unique configs out of {len(self.history)} +++")

        # reduce the list[float] to the top loss
        for k, v in histories.items():
            histories[k] = max(v)

        # register history
        for k, v in histories.items():
            bayes.register(
                params=dict(k),
                target=v,
            )

        dropout_suggestion = bayes.suggest(utility)
        # fill in the missing dimensions using self.best_candidate
        print(f"+++ Dropout suggestion: {dropout_suggestion} +++")
        print(f"+++ Best candidate: {self.best_candidate.bayes_config} +++")
        result = (
            self.best_candidate.bayes_config
            if self.best_candidate
            else self.random_bayes.suggest(utility)
        )
        for k in dims:
            result[k] = dropout_suggestion[k]

        print(f"+++ Merged config: {result} +++")
        return result

    def register(self, candidate: "Candidate"):
        print(f"+++ Registering score {candidate.loss} +++")
        self.history[candidate.id] = candidate
        if not self.enable_bayes_dropout:
            self.bayes.register(
                params=candidate.bayes_config,
                target=candidate.loss,
            )

    # Used to determine whether to register the random configs in the GP, while still in the first halving.
    # Registering them means GP suggestions won't be random anymore.
    def is_very_first_halving(self):
        # Do register them when GP-optimizing.
        if self.gp_opt_only:
            return False

        return self.halving == 0 and self.bracket == self.num_brackets

    def avg_loss(self) -> float:
        losses = [c.loss for c in self.candidates]
        return sum(losses) / len(losses)

    def top_k(self, k) -> None:
        if self.gp_opt_only:
            k = 1
        self.loss_threshold = self.avg_loss()
        to_be_deleted = []

        # Do nothing if there are not enough candidates.
        if k >= len(self.candidates):
            return
        elif k <= 0:
            k = 1

        # Sort.
        sorted_candidates = [
            c for c in sorted(self.candidates, reverse=True, key=lambda i: i.loss)
        ]

        # Delete
        # Below top k will be deleted.
        to_be_deleted.extend(sorted_candidates[k:])
        # Below threshold will be deleted even in the top k.
        below_threshold = [
            c for c in sorted_candidates[:k] if c.loss < self.loss_threshold
        ]
        print(f"+++ {len(below_threshold)} candidates below threshold +++")
        to_be_deleted.extend(below_threshold)
        for d in to_be_deleted:
            d.delete()

        # Keep above threshold in top k.
        self.candidates = [
            c for c in sorted_candidates[:k] if c.loss >= self.loss_threshold
        ]

        print(
            f"+++ Keeping {len(self.candidates)} of top {k}: {[c.loss for c in self.candidates]} +++"
        )


class Candidate:
    def __init__(self, bayes_config: dict, settings: dict, train_fn: Callable, kwargs):
        self.bayes_config = bayes_config
        self.settings = settings
        self.train_fn = train_fn
        self.loss = INFINITY
        self.pie_config = {}
        self.existing_model_path = ""
        self.epochs_trained = 0
        self.kwargs = kwargs
        self.id = str(uuid.uuid4())

    def run(self, epochs) -> None:
        try:
            self.set_pie_config(epochs)
            self.existing_model_path, pie_loss = self.train_fn(
                check_settings(merge_task_defaults(self.pie_config)), **self.kwargs
            )

            # minus because we want to maximize
            self.loss = -(pie_loss["lemma"] + pie_loss["pos"])
            if math.isnan(self.loss):  # Yes, this does occur for some reason
                self.loss = INFINITY

            self.pie_config["existing_model"] = self.existing_model_path
        except Exception as e:
            print(f"+++ Exception in single_run: {e} +++\n{traceback.format_exc()}")
            self.loss = INFINITY
            self.delete()

    def set_pie_config(self, epochs) -> None:
        self.pie_config = PieSpace.to_pie_config(self.bayes_config)
        self.pie_config = Settings(
            utils.recursive_merge(dict(self.settings), self.pie_config, overwrite=True)
        )
        self.pie_config["epochs"] = epochs
        self.pie_config["existing_model"] = self.existing_model_path
        self.epochs_trained += epochs

    def delete(self) -> None:
        self.save_metadata()
        if os.path.exists(self.existing_model_path):
            os.remove(self.existing_model_path)
        else:
            print(f"+++ The file {self.existing_model_path} does not exist +++")

    def save_metadata(self) -> None:
        """
        Write the used parameters, resulting loss value and uuid of this model to disk.
        """
        self.pie_config["loss"] = self.loss
        self.pie_config["config_uuid"] = self.id
        self.pie_config["epochs"] = self.epochs_trained
        json_path = os.path.join(
            self.pie_config["modelpath"], f"hyperparameters-{self.id}.json"
        )
        with open(json_path, "w+") as f:
            json.dump(self.pie_config, f)
