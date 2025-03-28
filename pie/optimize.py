from pie.hyperband import BOHB


def run_optimize(train_fn, settings, **kwargs):
    bohb = BOHB(
        train_fn,
        settings,
        81,
        3,
        enable_bayes_dropout=True,
        gp_opt_only=True,
        warm_start=True,
        kwargs=kwargs,
    )
    bohb.run()
