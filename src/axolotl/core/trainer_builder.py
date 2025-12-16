"""
Builder for the training args and trainer
"""

import abc
import importlib
import logging
import math
import os
import sys
from abc import abstractmethod
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Optional, Union
import ipdb
import torch
import transformers
from datasets import Dataset
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, DistributedSampler, SequentialSampler
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments
from transformers.trainer_pt_utils import SequentialDistributedSampler
from transformers import AutoTokenizer
from axolotl.monkeypatch.relora import ReLoRACallback, ReLoRAScheduler
from axolotl.utils.callbacks import (
    EvalFirstStepCallback,
    GPUStatsCallback,
    SaveAxolotlConfigtoWandBCallback,
    SaveBetterTransformerModelCallback,
    bench_eval_callback_factory,
    log_prediction_callback_factory,
)
from axolotl.utils.collators import DataCollatorForSeq2Seq
from axolotl.utils.dataloader import MultipackDistributedDataloader
from axolotl.utils.schedulers import get_cosine_schedule_with_quadratic_warmup
from collections import defaultdict
import numpy as np
def analyze_gradient_health_fixed(model, 
                                 threshold_low=1e-8,     # 梯度消失阈值
                                 threshold_high=1e2,     # 梯度爆炸阈值（调低了！）
                                 nan_inf_threshold=0.1,  # NaN/Inf比例阈值
                                 verbose=True,
                                 layer_stats=True,
                                 param_details=False,
                                 per_element_norm=True):  # 新增：按元素计算梯度大小
    """
    修复版梯度健康度分析：避免大参数矩阵范数计算误差
    
    修改点：
    1. 不直接使用整体L2范数，而是使用平均梯度大小
    2. 添加相对梯度检查
    3. 优化大参数处理
    """
    
    print("🔍 模型梯度健康度分析（修复版）")
    print("=" * 60)
    
    stats = {
        'total_params': 0,
        'trainable_params': 0,
        'params_with_grad': 0,
        'params_no_grad': 0,
        'params_healthy': 0,
        'params_vanished': 0,
        'params_exploded': 0,
        'params_nan': 0,
        'params_inf': 0,
        'grad_avg_magnitudes': [],  # 改为平均梯度大小
        'grad_abs_means': [],       # 梯度绝对值均值
        'grad_maxs': [],
        'grad_mins': [],
        'layer_stats': defaultdict(lambda: {
            'total': 0, 'healthy': 0, 'vanished': 0, 
            'exploded': 0, 'nan': 0, 'inf': 0
        }),
        'problematic_layers': [],
        'max_grad_abs_mean': 0.0,
        'min_grad_abs_mean': float('inf'),
        'median_grad_abs_mean': 0.0,
        'mean_grad_abs_mean': 0.0,
    }
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        param_count = param.numel()
        stats['total_params'] += param_count
        stats['trainable_params'] += param_count
        
        # 获取层名
        if '.' in name:
            layer_name = '.'.join(name.split('.')[:-1])
        else:
            layer_name = 'root'
        
        if param.grad is not None:
            stats['params_with_grad'] += param_count
            grad = param.grad
            
            # 检查NaN和Inf
            nan_mask = torch.isnan(grad)
            inf_mask = torch.isinf(grad)
            nan_count = torch.sum(nan_mask).item()
            inf_count = torch.sum(inf_mask).item()
            has_nan = nan_count > 0
            has_inf = inf_count > 0
            
            # 移除NaN/Inf
            if has_nan or has_inf:
                # 创建有效掩码
                valid_mask = ~(nan_mask | inf_mask)
                valid_grad = grad[valid_mask]
                if valid_grad.numel() == 0:
                    # 全是NaN/Inf
                    stats['params_nan'] += nan_count
                    stats['params_inf'] += inf_count
                    stats['layer_stats'][layer_name]['nan'] += nan_count
                    stats['layer_stats'][layer_name]['inf'] += inf_count
                    continue
            else:
                valid_grad = grad
            
            # 关键修改：不直接用L2范数，而是用平均梯度大小
            if per_element_norm:
                # 方法1：计算平均梯度大小（对每个元素独立评估）
                grad_abs_mean = torch.mean(torch.abs(valid_grad)).item()
                grad_max = torch.max(torch.abs(valid_grad)).item()
                grad_min = torch.min(torch.abs(valid_grad)).item()
            else:
                # 方法2：如果必须用范数，用相对范数
                grad_norm = torch.norm(valid_grad, p=2).item()
                # 除以sqrt(param_count)得到每个元素的平均贡献
                grad_abs_mean = grad_norm / math.sqrt(param_count)
                grad_max = torch.max(torch.abs(valid_grad)).item()
                grad_min = torch.min(torch.abs(valid_grad)).item()
            
            # 记录统计
            stats['grad_avg_magnitudes'].append(grad_abs_mean)
            stats['grad_maxs'].append(grad_max)
            stats['grad_mins'].append(grad_min)
            
            # 判断梯度状态（基于每个元素的平均梯度大小）
            if has_nan or has_inf:
                status = "💀"
                if has_nan:
                    stats['params_nan'] += nan_count
                    stats['layer_stats'][layer_name]['nan'] += nan_count
                if has_inf:
                    stats['params_inf'] += inf_count
                    stats['layer_stats'][layer_name]['inf'] += inf_count
            elif grad_abs_mean < threshold_low:
                status = "❄️"  # 梯度消失
                stats['params_vanished'] += param_count
                stats['layer_stats'][layer_name]['vanished'] += param_count
            elif grad_abs_mean > threshold_high:
                status = "💥"  # 梯度爆炸
                stats['params_exploded'] += param_count
                stats['layer_stats'][layer_name]['exploded'] += param_count
            else:
                status = "✅"  # 梯度正常
                stats['params_healthy'] += param_count
                stats['layer_stats'][layer_name]['healthy'] += param_count
            
            # 更新统计
            stats['max_grad_abs_mean'] = max(stats['max_grad_abs_mean'], grad_abs_mean)
            stats['min_grad_abs_mean'] = min(stats['min_grad_abs_mean'], grad_abs_mean)
            
            stats['layer_stats'][layer_name]['total'] += param_count
            
            if param_details and verbose:
                print(f"{status} {name:40} | "
                      f"形状: {str(param.shape):<15} | "
                      f"平均梯度大小: {grad_abs_mean:.2e} | "
                      f"[{grad_min:.2e}, {grad_max:.2e}] | "
                      f"NaN: {nan_count}/{param_count} | "
                      f"Inf: {inf_count}/{param_count}")
        else:
            stats['params_no_grad'] += param_count
            stats['layer_stats'][layer_name]['total'] += param_count
            if param_details and verbose:
                print(f"❌ {name:40} | 形状: {str(param.shape):<15} | 梯度: None")
    
    # 计算统计
    if stats['grad_avg_magnitudes']:
        stats['mean_grad_abs_mean'] = np.mean(stats['grad_avg_magnitudes'])
        stats['median_grad_abs_mean'] = np.median(stats['grad_avg_magnitudes'])
    
    # 计算比例
    if stats['trainable_params'] > 0:
        stats['healthy_ratio'] = stats['params_healthy'] / stats['trainable_params']
        stats['vanished_ratio'] = stats['params_vanished'] / stats['trainable_params']
        stats['exploded_ratio'] = stats['params_exploded'] / stats['trainable_params']
        stats['nan_ratio'] = stats['params_nan'] / stats['trainable_params']
        stats['inf_ratio'] = stats['params_inf'] / stats['trainable_params']
        stats['has_grad_ratio'] = stats['params_with_grad'] / stats['trainable_params']
    else:
        stats['healthy_ratio'] = 0.0
        stats['vanished_ratio'] = 0.0
        stats['exploded_ratio'] = 0.0
        stats['nan_ratio'] = 0.0
        stats['inf_ratio'] = 0.0
        stats['has_grad_ratio'] = 0.0
    
    # 打印汇总
    if verbose:
        print(f"\n📊 梯度分析汇总（修复版）")
        print("=" * 60)
        print(f"总参数数: {stats['total_params']:,}")
        print(f"可训练参数: {stats['trainable_params']:,}")
        print(f"有梯度参数: {stats['params_with_grad']:,} ({stats['has_grad_ratio']:.1%})")
        print(f"梯度正常参数: {stats['params_healthy']:,} ({stats['healthy_ratio']:.1%})")
        print(f"梯度消失参数: {stats['params_vanished']:,} ({stats['vanished_ratio']:.1%})")
        print(f"梯度爆炸参数: {stats['params_exploded']:,} ({stats['exploded_ratio']:.1%})")
        print(f"NaN梯度参数: {stats['params_nan']:,} ({stats['nan_ratio']:.1%})")
        print(f"Inf梯度参数: {stats['params_inf']:,} ({stats['inf_ratio']:.1%})")
        
        if stats['grad_avg_magnitudes']:
            print(f"\n📈 梯度大小统计（每个参数的平均梯度）")
            print(f"最大平均梯度: {stats['max_grad_abs_mean']:.2e}")
            print(f"最小平均梯度: {stats['min_grad_abs_mean']:.2e}")
            print(f"平均梯度大小: {stats['mean_grad_abs_mean']:.2e}")
            print(f"中位梯度大小: {stats['median_grad_abs_mean']:.2e}")
            print(f"正常范围: [{threshold_low:.0e}, {threshold_high:.0e}]")
    
    return stats


try:
    import torch._dynamo  # pylint: disable=ungrouped-imports
except ImportError:
    pass

LOG = logging.getLogger("axolotl.core.trainer_builder")


@dataclass
class AxolotlTrainingArguments(TrainingArguments):
    """
    Extend the base TrainingArguments for axolotl helpers
    """

    lr_quadratic_warmup: bool = field(
        default=False,
        metadata={"help": "Use quadratic warmup for cosine scheduling."},
    )
    sample_packing: bool = field(
        default=False,
        metadata={"help": "Use sample packing for efficient training."},
    )
    eval_sample_packing: Optional[bool] = field(
        default=None,
        metadata={"help": "Use sample packing for efficient evals."},
    )
    sample_packing_efficiency: float = field(
        default=1.0,
        metadata={"help": "Sample packing efficiency for calculating batch length."},
    )
    max_seq_length: int = field(
        default=2048,
        metadata={"help": "The maximum sequence length the model can handle"},
    )
    sample_packing_seq_len_multiplier: int = field(
        default=1,
        metadata={"help": "the multiplier for the max len for packed sequences"},
    )
    relora_steps: Optional[int] = field(
        default=None,
        metadata={"help": "how often to reset for ReLoRA"},
    )
    relora_warmup_steps: Optional[int] = field(
        default=None,
        metadata={"help": "how many warmup steps to take after reset for ReLoRA"},
    )
    bench_split: Optional[str] = field(
        default="eval", metadata={"help": "The benchmark split to run on"}
    )
    bench_dataset: Optional[str] = field(
        default="pharaouk/dharma-1/dharma_1_mini.json",
        metadata={
            "help": "Benchmark dataset to use: options are `mmlu-zs`, `mmlu-fs`, or the full path to the dataset file"
        },
    )
    do_bench_eval: Optional[bool] = field(
        default=False, metadata={"help": "Whether to run the Benchmark evaluation."}
    )
    max_bench_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "If set, only evaluates on `max_bench_samples` of the benchmark dataset."
        },
    )
    bench_source_max_len: int = field(
        default=2048, metadata={"help": "Maximum source sequence length for bench."}
    )


class AxolotlTrainer(Trainer):
    """
    Extend the base Trainer for axolotl helpers
    """

    args = None  # type: AxolotlTrainingArguments
    
    def __init__(self, *args, num_epochs=1, bench_data_collator=None, **kwargs):
        self.num_epochs = num_epochs
        self.bench_data_collator = bench_data_collator
        tokenizer_cls = getattr(transformers, 'LlamaTokenizer')
        self.tokenizer = tokenizer_cls.from_pretrained(
            '/mnt/zyk/Medusa/vicuna-7b-v1.5',
            trust_remote_code=None or False,
            use_fast=True,
    )

        super().__init__(*args, **kwargs)

    def create_scheduler(
        self, num_training_steps: int, optimizer: torch.optim.Optimizer = None
    ):
        """
        Setup the scheduler. The optimizer of the trainer must have been set up either before this method is called or
        passed as an argument.

        Args:
            num_training_steps (int): The number of training steps to do.
            optimizer (torch.optim.Optimizer): The training optimizer
        """

        # fmt: off
        if self.lr_scheduler is None:  # type: ignore  # pylint: disable=access-member-before-definition
            # fmt: on
            if (
                self.args.lr_scheduler_type == "cosine"
                and self.args.lr_quadratic_warmup is True
            ):
                self.lr_scheduler = get_cosine_schedule_with_quadratic_warmup(  # pylint: disable=attribute-defined-outside-init
                    optimizer,
                    num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                    num_training_steps=num_training_steps,
                )
            else:
                return super().create_scheduler(num_training_steps, optimizer)
        return self.lr_scheduler

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.args.world_size > 1 and self.args.sample_packing:
            return DistributedSampler(
                self.train_dataset,
                num_replicas=self.args.world_size,
                rank=self.args.process_index,
                seed=self.args.seed,
            )
        return super()._get_train_sampler()

    def _get_eval_sampler(
        self, eval_dataset: Dataset
    ) -> Optional[torch.utils.data.Sampler]:
        if (
            self.args.world_size > 1
            and self.args.sample_packing
            and self.args.eval_sample_packing is not False
        ):
            return SequentialDistributedSampler(
                eval_dataset,
                num_replicas=self.args.world_size,
                rank=self.args.process_index,
                batch_size=self.args.per_device_eval_batch_size,
            )
        return super()._get_eval_sampler(eval_dataset)

    def get_train_dataloader(self) -> Union[DataLoader, MultipackDistributedDataloader]:
        if self.args.sample_packing:
            train_sampler = self._get_train_sampler()
            return self.accelerator.prepare(
                MultipackDistributedDataloader(
                    self.train_dataset,
                    batch_size=self._train_batch_size,
                    seq_max_length=self.args.max_seq_length,
                    collate_fn=self.data_collator,
                    sampler=train_sampler,
                    packing_efficiency_estimate=self.args.sample_packing_efficiency,
                    sample_packing_seq_len_multiplier=self.args.sample_packing_seq_len_multiplier,
                    device_count=int(os.environ.get("WORLD_SIZE", 1)),
                    num_epochs=self.num_epochs,
                )
            )
        return super().get_train_dataloader()

    def get_eval_dataloader(
        self, eval_dataset: Optional[Dataset] = None
    ) -> Union[DataLoader, MultipackDistributedDataloader]:
        if self.args.sample_packing and self.args.eval_sample_packing is not False:
            eval_dataset = (
                eval_dataset if eval_dataset is not None else self.eval_dataset
            )

            eval_sampler = self._get_eval_sampler(eval_dataset)
            return self.accelerator.prepare(
                MultipackDistributedDataloader(
                    eval_dataset,
                    batch_size=self.args.eval_batch_size,
                    seq_max_length=self.args.max_seq_length,
                    collate_fn=self.data_collator,
                    sampler=eval_sampler,
                    packing_efficiency_estimate=self.args.sample_packing_efficiency,
                    sample_packing_seq_len_multiplier=self.args.eval_batch_size,
                    device_count=int(os.environ.get("WORLD_SIZE", 1)),
                    num_epochs=self.num_epochs,
                )
            )
        return super().get_eval_dataloader(eval_dataset)

    def _get_bench_sampler(
        self, bench_dataset: Dataset
    ) -> Optional[torch.utils.data.Sampler]:
        if self.args.world_size <= 1:
            return SequentialSampler(bench_dataset)
        return None

    def get_bench_dataloader(
        self,
        bench_dataset: Dataset,
    ) -> Union[DataLoader, MultipackDistributedDataloader]:
        dataloader_params = {
            "batch_size": self.args.eval_batch_size,
            "collate_fn": self.bench_data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }

        if not isinstance(bench_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_bench_sampler(bench_dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last

        return DataLoader(bench_dataset, **dataloader_params)
        # return self.accelerator.prepare(DataLoader(bench_dataset, **dataloader_params))

    def compute_loss(self, model, inputs,tokenizer,return_outputs=False):
        # use one's weighted cross entropy loss calc
        # if self.args.sample_packing:
        #     labels = inputs.pop("labels")
        #     outputs = model(**inputs)
        #     loss = trainer_weighted_loss(outputs, labels, shift_labels=True)
        #     return (loss, outputs) if return_outputs else loss
        return super().compute_loss(model, inputs,tokenizer,return_outputs=return_outputs)

    def training_step(self, model, inputs) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs,self.tokenizer_1)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            # with amp.scale_loss(loss, self.optimizer) as scaled_loss:
            #     scaled_loss.backward()
            raise NotImplementedError("Apex is no longer supported")
        else:
            self.accelerator.backward(loss)
        analyze_gradient_health_fixed(self.model)
        return loss.detach() / self.args.gradient_accumulation_steps

class OneCycleLRSchedulerTrainer(AxolotlTrainer):
    """
    Trainer subclass that uses the OneCycleLR scheduler
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lr_scheduler = None

    def create_scheduler(
        self,
        num_training_steps: int,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        optimizer = self.optimizer if optimizer is None else optimizer
        num_warmup_steps = self.args.get_warmup_steps(num_training_steps)
        pct_start = num_warmup_steps / num_training_steps

        self.lr_scheduler = OneCycleLR(
            optimizer,
            max_lr=self.args.learning_rate,
            total_steps=num_training_steps,
            pct_start=pct_start,
            div_factor=6,
        )

        return self.lr_scheduler


class ReLoRATrainer(AxolotlTrainer):
    """
    Trainer subclass that uses the OneCycleLR scheduler
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lr_scheduler = None

    def create_scheduler(
        self,
        num_training_steps: int,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        optimizer = self.optimizer if optimizer is None else optimizer
        lr_scheduler = super().create_scheduler(num_training_steps, optimizer)

        if self.args.relora_steps:
            warmup_steps = (
                self.args.relora_warmup_steps if self.args.relora_warmup_steps else 10
            )
            self.lr_scheduler = ReLoRAScheduler(
                optimizer,
                lr_scheduler,
                self.args.relora_steps,
                warmup_steps,
            )
        else:
            self.lr_scheduler = lr_scheduler

        return self.lr_scheduler


class TrainerBuilderBase(abc.ABC):
    """
    Base class for trainer builder
    """

    _train_dataset = None
    _eval_dataset = None

    def __init__(self, cfg, model, tokenizer):
        self.cfg = cfg
        self.model = model
        self.tokenizer = tokenizer

    @property
    def train_dataset(self):
        return self._train_dataset

    @train_dataset.setter
    def train_dataset(self, dataset):
        self._train_dataset = dataset

    @property
    def eval_dataset(self):
        return self._eval_dataset

    @eval_dataset.setter
    def eval_dataset(self, dataset):
        self._eval_dataset = dataset

    @abstractmethod
    def build(self, total_num_steps):
        pass

    @abstractmethod
    def get_callbacks(self):
        pass

    @abstractmethod
    def get_post_trainer_create_callbacks(self, trainer):
        """
        Callbacks added after the trainer is created, usually b/c these need access to the trainer
        """


class HFCausalTrainerBuilder(TrainerBuilderBase):
    """
    Build the HuggingFace training args/trainer for Causal models
    """

    def hook_pre_create_training_args(self, training_arguments_kwargs):
        # TODO
        return training_arguments_kwargs

    def hook_post_create_training_args(self, training_arguments):
        # TODO
        return training_arguments

    def hook_pre_create_trainer(self, trainer_kwargs, trainer_cls):
        # TODO
        return trainer_kwargs, trainer_cls

    def hook_post_create_trainer(self, trainer):
        # TODO
        return trainer

    def get_callbacks(self):
        callbacks = []
        callbacks.append(GPUStatsCallback(self.cfg))
        callbacks.append(EvalFirstStepCallback)

        if self.cfg.relora_steps:
            callbacks.append(ReLoRACallback(self.cfg))

        if (
            hasattr(self.model, "use_bettertransformer")
            and self.model.use_bettertransformer is True
        ):
            callbacks.append(SaveBetterTransformerModelCallback)

        if self.cfg.use_wandb:
            callbacks.append(
                SaveAxolotlConfigtoWandBCallback(self.cfg.axolotl_config_path)
            )

        return callbacks

    def get_post_trainer_create_callbacks(self, trainer):
        callbacks = []
        if self.cfg.use_wandb and self.cfg.eval_table_size > 0:
            LogPredictionCallback = log_prediction_callback_factory(
                trainer, self.tokenizer
            )
            callbacks.append(LogPredictionCallback(self.cfg))

        if self.cfg.do_bench_eval:
            callbacks.append(bench_eval_callback_factory(trainer, self.tokenizer))

        if self.cfg.early_stopping_patience:
            early_stop_cb = EarlyStoppingCallback(
                self.cfg.early_stopping_patience,
            )
            callbacks.append(early_stop_cb)

        return callbacks

    def _get_trainer_cls(self):
        if self.cfg.lr_scheduler == "one_cycle" and (
            self.cfg.fsdp or self.cfg.adapter == "qlora"
        ):
            return OneCycleLRSchedulerTrainer
        if self.cfg.relora_steps:
            return ReLoRATrainer
        return AxolotlTrainer

    def build(self, total_num_steps):
        warmup_steps = (
            self.cfg.warmup_steps
            if self.cfg.warmup_steps is not None
            else min(int(0.03 * total_num_steps), 100)
        )
        logging_steps = (
            self.cfg.logging_steps
            if self.cfg.logging_steps is not None
            else max(min(int(0.005 * total_num_steps), 10), 1)
        )

        training_arguments_kwargs = {}
        if self.cfg.bf16 == "full":
            training_arguments_kwargs["bf16_full_eval"] = True
        else:
            training_arguments_kwargs["bf16"] = self.cfg.bf16
        training_arguments_kwargs["fp16"] = (
            self.cfg.fp16 and not self.cfg.bf16
        ) or False
        training_arguments_kwargs["tf32"] = self.cfg.tf32
        training_arguments_kwargs["warmup_steps"] = warmup_steps
        training_arguments_kwargs["logging_steps"] = logging_steps

        if self.cfg.seed:
            training_arguments_kwargs["seed"] = self.cfg.seed

        if self.cfg.gradient_checkpointing:
            training_arguments_kwargs[
                "gradient_checkpointing"
            ] = self.cfg.gradient_checkpointing
        if self.cfg.fsdp:
            training_arguments_kwargs["fsdp"] = self.cfg.fsdp
            if self.cfg.fsdp_config:
                training_arguments_kwargs["fsdp_config"] = dict(self.cfg.fsdp_config)

        # deepspeed
        if self.cfg.deepspeed:
            training_arguments_kwargs["deepspeed"] = self.cfg.deepspeed

        if self.cfg.lr_quadratic_warmup is not None:
            training_arguments_kwargs[
                "lr_quadratic_warmup"
            ] = self.cfg.lr_quadratic_warmup

        if self.cfg.adam_beta1:
            training_arguments_kwargs["adam_beta1"] = self.cfg.adam_beta1
        if self.cfg.adam_beta2:
            training_arguments_kwargs["adam_beta2"] = self.cfg.adam_beta2
        if self.cfg.adam_epsilon:
            training_arguments_kwargs["adam_epsilon"] = self.cfg.adam_epsilon
        if self.cfg.max_grad_norm:
            training_arguments_kwargs["max_grad_norm"] = self.cfg.max_grad_norm

        if self.cfg.hub_model_id:
            training_arguments_kwargs["hub_model_id"] = self.cfg.hub_model_id
            training_arguments_kwargs["push_to_hub"] = True
            training_arguments_kwargs["hub_private_repo"] = True

            if self.cfg.hub_strategy:
                training_arguments_kwargs["hub_strategy"] = self.cfg.hub_strategy

        if self.cfg.save_safetensors:
            training_arguments_kwargs["save_safetensors"] = self.cfg.save_safetensors

        if self.cfg.sample_packing_eff_est:
            training_arguments_kwargs[
                "sample_packing_efficiency"
            ] = self.cfg.sample_packing_eff_est

        if self.cfg.eval_steps:
            training_arguments_kwargs["evaluation_strategy"] = "steps"
            training_arguments_kwargs["eval_steps"] = self.cfg.eval_steps
        elif self.cfg.evaluation_strategy:
            training_arguments_kwargs[
                "evaluation_strategy"
            ] = self.cfg.evaluation_strategy
        elif self.cfg.val_set_size == 0:
            # no eval set, so don't eval
            training_arguments_kwargs["evaluation_strategy"] = "no"
        else:
            # we have an eval set, but no steps defined, default to use epoch
            training_arguments_kwargs["evaluation_strategy"] = "epoch"

        if self.cfg.save_steps:
            training_arguments_kwargs["save_strategy"] = "steps"
            training_arguments_kwargs["save_steps"] = self.cfg.save_steps
        elif self.cfg.save_strategy:
            training_arguments_kwargs["save_strategy"] = self.cfg.save_strategy
        else:
            # default to saving each epoch if not defined
            training_arguments_kwargs["save_strategy"] = "epoch"

        if self.cfg.do_bench_eval:
            training_arguments_kwargs["do_bench_eval"] = self.cfg.do_bench_eval
            if self.cfg.bench_dataset:
                training_arguments_kwargs["bench_dataset"] = self.cfg.bench_dataset
        if self.cfg.metric_for_best_model:
            training_arguments_kwargs[
                "metric_for_best_model"
            ] = self.cfg.metric_for_best_model
        if self.cfg.greater_is_better:
            training_arguments_kwargs["greater_is_better"] = self.cfg.greater_is_better

        if self.cfg.torch_compile:
            if torch.__version__ < "2.1.0":  # pylint: disable=protected-access
                LOG.warning("torch>=2.1.0 required for torch_compile to work properly")
            elif torch._dynamo:  # pylint: disable=protected-access
                torch._dynamo.config.suppress_errors = (  # pylint: disable=protected-access
                    True
                )
                training_arguments_kwargs["torch_compile"] = self.cfg.torch_compile
                if self.cfg.torch_compile_backend:
                    training_arguments_kwargs[
                        "torch_compile_backend"
                    ] = self.cfg.torch_compile_backend

        # DDP Config
        if self.cfg.ddp_timeout:
            training_arguments_kwargs["ddp_timeout"] = self.cfg.ddp_timeout
        # see https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html
        if self.cfg.ddp_bucket_cap_mb:
            training_arguments_kwargs["ddp_bucket_cap_mb"] = self.cfg.ddp_bucket_cap_mb
        if self.cfg.ddp_broadcast_buffers is not None:
            training_arguments_kwargs[
                "ddp_broadcast_buffers"
            ] = self.cfg.ddp_broadcast_buffers

        # these are all the "standard" kwargs that are def used
        training_arguments_kwargs["max_steps"] = (
            total_num_steps if self.cfg.max_steps else -1
        )
        training_arguments_kwargs["max_seq_length"] = self.cfg.sequence_len
        training_arguments_kwargs[
            "per_device_train_batch_size"
        ] = self.cfg.micro_batch_size
        training_arguments_kwargs[
            "per_device_eval_batch_size"
        ] = self.cfg.eval_batch_size
        training_arguments_kwargs[
            "gradient_accumulation_steps"
        ] = self.cfg.gradient_accumulation_steps
        training_arguments_kwargs[
            "eval_accumulation_steps"
        ] = self.cfg.gradient_accumulation_steps
        training_arguments_kwargs["num_train_epochs"] = self.cfg.num_epochs
        training_arguments_kwargs["learning_rate"] = self.cfg.learning_rate
        training_arguments_kwargs["output_dir"] = self.cfg.output_dir
        training_arguments_kwargs["save_total_limit"] = (
            self.cfg.save_total_limit if self.cfg.save_total_limit else 4
        )
        training_arguments_kwargs["load_best_model_at_end"] = (
            (
                self.cfg.load_best_model_at_end is not False
                or self.cfg.early_stopping_patience
            )
            and self.cfg.val_set_size > 0
            and self.cfg.save_steps
            and self.cfg.eval_steps
            and self.cfg.save_steps % self.cfg.eval_steps == 0
        ) or False
        if self.cfg.ddp_find_unused_parameters is not None:
            training_arguments_kwargs[
                "ddp_find_unused_parameters"
            ] = self.cfg.ddp_find_unused_parameters
        else:    
            training_arguments_kwargs["ddp_find_unused_parameters"] = (
                False if self.cfg.ddp else None
            )
        training_arguments_kwargs["group_by_length"] = self.cfg.group_by_length
        training_arguments_kwargs["report_to"] = "wandb" if self.cfg.use_wandb else None
        training_arguments_kwargs["run_name"] = (
            self.cfg.wandb_run_id if self.cfg.use_wandb else None
        )
        training_arguments_kwargs["optim"] = (
            self.cfg.optimizer if self.cfg.optimizer else "adamw_hf"
        )
        training_arguments_kwargs["lr_scheduler_type"] = (
            self.cfg.lr_scheduler
            if self.cfg.lr_scheduler
            and self.cfg.lr_scheduler not in ("one_cycle", "log_sweep")
            else "cosine"
        )
        training_arguments_kwargs["weight_decay"] = (
            self.cfg.weight_decay if self.cfg.weight_decay is not None else 0.0
        )
        training_arguments_kwargs["sample_packing"] = (
            self.cfg.sample_packing if self.cfg.sample_packing else False
        )
        training_arguments_kwargs["eval_sample_packing"] = (
            self.cfg.sample_packing if self.cfg.sample_packing else False
        )
        training_arguments_kwargs[
            "sample_packing_seq_len_multiplier"
        ] = self.cfg.micro_batch_size
        training_arguments_kwargs["relora_steps"] = self.cfg.relora_steps
        training_arguments_kwargs["relora_warmup_steps"] = self.cfg.relora_warmup_steps
        training_arguments_kwargs = self.hook_pre_create_training_args(
            training_arguments_kwargs
        )
        training_args = (
            AxolotlTrainingArguments(  # pylint: disable=unexpected-keyword-arg
                **training_arguments_kwargs,
            )
        )
        training_args = self.hook_post_create_training_args(training_args)
        trainer_kwargs = {}

        if self.cfg.optimizer == "adamw_anyprecision":
            if Path(self.cfg.torchdistx_path).exists():
                sys.path.append(self.cfg.torchdistx_path)
                importlib.import_module("torchdistx")
        
        data_collator_kwargs = {
            "padding": True,  # True/"longest" is the default
        }
        if self.cfg.pad_to_sequence_len:
            data_collator_kwargs["pad_to_multiple_of"] = 64 * math.ceil(
                self.cfg.sequence_len / 64
            )
        else:
            # A100 is best at 64, while others at 8. Let's use the larger so we don't have to check
            # https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html
            data_collator_kwargs["pad_to_multiple_of"] = 64

        if self.cfg.is_llama_derived_model and self.cfg.landmark_attention:
            from axolotl.monkeypatch.llama_landmark_attn import (
                add_mem_tokens,
                get_mem_id,
                set_model_mem_id,
            )

            set_model_mem_id(self.model, self.tokenizer)

            LOG.info("Adding landmark attention tokens to dataset")

            for dataset in [self.train_dataset, self.eval_dataset]:
                dataset = dataset.map(
                    partial(
                        add_mem_tokens, mem_freq=50, mem_id=get_mem_id(self.tokenizer)
                    ),
                    batched=False,
                    num_proc=32,
                )

        trainer_cls = self._get_trainer_cls()
        trainer_kwargs, trainer_cls = self.hook_pre_create_trainer(
            trainer_kwargs, trainer_cls
        )
        trainer = trainer_cls(
            model=self.model,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            args=training_args,
            data_collator=DataCollatorForSeq2Seq(
                self.tokenizer,
                return_tensors="pt",
                **data_collator_kwargs,
            ),
            bench_data_collator=transformers.DataCollatorForSeq2Seq(
                self.tokenizer,
                return_tensors="pt",
                **data_collator_kwargs,
            ),
            callbacks=self.get_callbacks(),
            num_epochs=self.cfg.num_epochs,
            **trainer_kwargs,
        )
        trainer = self.hook_post_create_trainer(trainer)
        for callback in self.get_post_trainer_create_callbacks(trainer):
            trainer.add_callback(callback)

        return trainer
