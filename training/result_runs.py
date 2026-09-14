"""Parse and filter the run names emitted by ``train_curriculum.py``."""

import re
from dataclasses import dataclass


RUN_NAME_RE = re.compile(
    r"^tl_(?P<order>.+)_(?P<schedule>staged|competence)"
    r"_ep(?P<epochs>\d+)_(?P<lm_init>[^_]+)lm_(?P<modality>[^_]+)"
    r"_seed(?P<seed>\d+)_(?P<run_id>.+)$"
)


@dataclass(frozen=True)
class RunSpec:
    name: str
    order: str
    schedule: str
    epochs: int
    lm_init: str
    modality: str
    seed: int
    run_id: str

    @property
    def condition(self):
        """Fields that define an experimental condition, excluding seed."""
        return self.epochs, self.lm_init, self.modality, self.order, self.schedule


def parse_run_name(name):
    match = RUN_NAME_RE.fullmatch(name)
    if match is None:
        return None
    fields = match.groupdict()
    return RunSpec(
        name=name,
        order=fields["order"],
        schedule=fields["schedule"],
        epochs=int(fields["epochs"]),
        lm_init=fields["lm_init"],
        modality=fields["modality"],
        seed=int(fields["seed"]),
        run_id=fields["run_id"],
    )


def add_run_filter_args(parser):
    parser.add_argument("--epochs", type=int, help="include only this training budget")
    parser.add_argument("--lm_init", help="include only this language-model initialization")
    parser.add_argument("--modality", help="include only this modality, e.g. mm or textonly")
    parser.add_argument(
        "--run_name_regex",
        help="include only complete run names matching this regular expression",
    )


def compile_run_name_filter(parser, expression):
    if expression is None:
        return None
    try:
        return re.compile(expression)
    except re.error as exc:
        parser.error(f"invalid --run_name_regex: {exc}")


def run_matches(spec, args, name_filter=None):
    return (
        (args.epochs is None or spec.epochs == args.epochs)
        and (args.lm_init is None or spec.lm_init == args.lm_init)
        and (args.modality is None or spec.modality == args.modality)
        and (name_filter is None or name_filter.search(spec.name) is not None)
    )


def condition_label(condition):
    epochs, lm_init, modality, order, schedule = condition
    return f"ep{epochs}/{lm_init}lm/{modality}/{order}/{schedule}"


def record_unique_score(scores, origins, condition, task, spec, accuracy, report):
    """Store one score and reject reruns that would otherwise overwrite it."""
    identity = condition, task, spec.seed
    if identity in origins:
        previous = origins[identity]
        raise RuntimeError(
            "Multiple results match the same condition/task/seed:\n"
            f"  {previous}\n"
            f"  {report}\n"
            "Select one experiment with --epochs, --lm_init, --modality, or "
            "--run_name_regex."
        )
    origins[identity] = report
    scores[condition][task][spec.seed] = accuracy
