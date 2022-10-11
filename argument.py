from dataclasses import dataclass, field
from transformers import TrainingArguments


@dataclass
class TrainingCat1NLPArguments(TrainingArguments):
    output_dir: str = field(default="./output/")
    num_train_epochs: int = field(default=10)
    per_device_train_batch_size: int = field(default=16)
    per_device_eval_batch_size: int = field(default=16)
    overwrite_output_dir: bool = field(default=True)
    load_best_model_at_end: bool = field(default=True)
    evaluation_strategy: str = field(default="steps")
    save_strategy: str = field(default="steps")
    metric_for_best_model: str = field(default="accuracy")
    save_total_limit: int = field(default=5)
    lr_scheduler_type: str = field(
        default="cosine_with_restarts",
        metadata={
            "help": "Select evaluation strategy[linear, cosine, cosine_with_restarts, polynomial, constant, constant with warmup]"
        },
    )
    warmup_steps: int = field(default=500)


@dataclass
class TrainCat1NLPModelArguments:
    model_name: str = field(default="klue/bert-base")
    loss_name: str = field(default="focal")
    project_cat1_name: str = field(default="cat1_nlp")
    data_path: str = field(default="./data/")


@dataclass
class TrainingCat2NLPArguments(TrainingArguments):
    output_dir: str = field(default="./output/")
    num_train_epochs: int = field(default=10)
    per_device_train_batch_size: int = field(default=16)
    per_device_eval_batch_size: int = field(default=16)
    overwrite_output_dir: bool = field(default=True)
    load_best_model_at_end: bool = field(default=True)
    evaluation_strategy: str = field(default="steps")
    save_strategy: str = field(default="steps")
    metric_for_best_model: str = field(default="accuracy")
    save_total_limit: int = field(int=5)
    lr_scheduler_type: str = field(
        default="cosine_with_restarts",
        metadata={
            "help": "Select evaluation strategy[linear, cosine, cosine_with_restarts, polynomial, constant, constant with warmup]"
        },
    )
    warmup_steps: int = field(default=500)


@dataclass
class TrainCat2NLPModelArguments:
    model_name: str = field(default="klue/bert-base")
    loss_name: str = field(default="focal")
    project_cat2_name: str = field(default="cat2_nlp")
    data_path: str = field(default="./data/")
