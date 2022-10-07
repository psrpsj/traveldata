from dataclasses import dataclass, field
from transformers import TrainingArguments


@dataclass
class TrainingCat1Arguments(TrainingArguments):
    output_dir: str = field(default="./output/cat1")
    num_train_epochs: int = field(default=5)
    per_device_train_batch_size: int = field(default=8)
    overwrite_output_dir: bool = field(default=True)
    load_best_model_at_end: bool = field(default=True)
    evaluation_strategy: str = field(default="steps")
    save_strategy: str = field(default="steps")
    metric_for_best_model: str = field(default="accuracy")
    lr_scheduler_type: str = field(
        default="cosine_with_restarts",
        metadata={
            "help": "Select evaluation strategy[linear, cosine, cosine_with_restarts, polynomial, constant, constant with warmup]"
        },
    )
    warmup_steps: int = field(default=500)
