from transformers import (
    PretrainedConfig,
    TrainerCallback,
)
import logging
import warnings
from functools import partial
from typing import List, Optional, Tuple, Union
import sys
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss
from axolotl.utils.dict import DictDefault
from axolotl.utils.distributed import is_main_process
import axolotl
from transformers.trainer_pt_utils import LabelSmoother
IGNORE_TOKEN_ID = LabelSmoother.ignore_index
import types
import math
import wandb
import transformers

logger = LOG = logging.getLogger("axolotl.monkeypatch.medusa")

class MedusaConfig(PretrainedConfig):
    """
    Configuration class for Medusa model.

    Args:
        medusa_num_heads (int, optional): Number of heads for the Medusa layer. Default is 2.
        medusa_num_layers (int, optional): Number of Medusa layers. Default is 1.
        base_model_name_or_path (str, optional): The name or path of the base model. Default is "lmsys/vicuna-7b-v1.3".
        num_unfreezed_layers (int, optional): Number of layers to unfreeze. Default is 0.
        **kwargs: Additional keyword arguments to be passed to the parent class constructor.
    """

    def __init__(
        self,
        medusa_num_heads=4,
        medusa_num_layers=1,
        base_model_name_or_path="lmsys/vicuna-7b-v1.3",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.medusa_num_heads = medusa_num_heads
        self.medusa_num_layers = medusa_num_layers
        self.base_model_name_or_path = base_model_name_or_path


class ResBlock(nn.Module):
    """
    A Residual Block module.

    This module performs a linear transformation followed by a SiLU activation,
    and then adds the result to the original input, creating a residual connection.

    Args:
        hidden_size (int): The size of the hidden layers in the block.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        # Initialize as an identity mapping
        nn.init.zeros_(self.linear.weight)
        # Use SiLU activation to keep consistent with the Llama model
        self.act = nn.SiLU()

    def forward(self, x):
        """
        Forward pass of the ResBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output after the residual connection and activation.
        """
        return x + self.act(self.linear(x))

class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, vocab_dim ,lm_head_layer,dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # 线性投影层
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        
        # 输出层
        self.proj = nn.Linear(embed_dim, vocab_dim)
        
        # 复制 lm_head 的权重和偏置（如果存在）
        self.proj.weight.data.copy_(lm_head_layer.weight.data)
        if hasattr(lm_head_layer, 'bias') and lm_head_layer.bias is not None:
            self.proj.bias.data.copy_(lm_head_layer.bias.data)


        self.dropout = nn.Dropout(dropout)
        
        # 缩放因子
        self.scale = self.head_dim ** -0.5

    def forward(self, x, context, mask=None):
        """
        Args:
            x:       Query 序列       [batch_size, seq_len_q, embed_dim]
            context: Key/Value 序列   [batch_size, seq_len_kv, embed_dim]
            mask:    可选的掩码        [batch_size, seq_len_q, seq_len_kv]
        Returns:
            out:     注意力输出        [batch_size, seq_len_q, embed_dim]
        """
        batch_size = x.size(0)
        
        # 1. 线性投影并分头
        q = self.query(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, L_q, D/H]
        k = self.key(context).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, L_kv, D/H]
        v = self.value(context).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, L_kv, D/H]
        
        # 2. 计算注意力分数
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, L_q, L_kv]
        
        # 3. 应用掩码（可选）
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        # 4. 注意力权重和输出
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        out = torch.matmul(attn_weights, v)  # [B, H, L_q, D/H]
        
        # 5. 合并多头并投影
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        out = self.proj(out)
        
        return out

def POS_embedding(current_vec: torch.Tensor, 
                 past_vec: torch.Tensor, 
                 numda: float) -> torch.Tensor:
    result = current_vec + numda * past_vec
    result = result / torch.tensor(1 + numda)
    return result

def add_medusa_heads(
    self,
    medusa_num_heads=4,
    medusa_num_layers=0,
):
    """
    Args:
        self (nn.Module): The base language model to be used.
        medusa_num_heads (int, optional): Number of additional tokens to predict. Defaults to 3.
        medusa_num_layers (int, optional): Number of ResBlock layers for each Medusa head. Defaults to 0.
    """
    hidden_size = self.lm_head.weight.shape[-1]
    vocab_size = self.lm_head.weight.shape[0]
    self.config.medusa_num_layers = medusa_num_layers
    self.config.medusa_num_heads = medusa_num_heads
    self.medusa_num_heads = medusa_num_heads
    # Create a list of Medusa heads
    self.medusa_head = nn.ModuleList(
        [
            nn.Sequential(
                *([ResBlock(hidden_size)] * medusa_num_layers),
                # nn.Linear(hidden_size, vocab_size, bias=False),
            )
            for _ in range(medusa_num_heads)
        ]
    )
    self.cross_attn = nn.ModuleList(
    [CrossAttention(hidden_size,4,vocab_size,self.lm_head) for _ in range(medusa_num_heads)]
    )
    self.proj_layers = nn.ModuleList([
            nn.Linear(vocab_size,hidden_size, bias=False)
            for _ in range(medusa_num_heads)
        ])
    for i in range(medusa_num_heads):
        with torch.no_grad():
            self.proj_layers[i].weight.data = self.lm_head.weight.T
            if self.proj_layers[i].bias is not None:
                nn.init.zeros_(self.proj_layers[i].bias)

    device = next(self.lm_head.parameters()).device
    dtype = next(self.lm_head.parameters()).dtype
    self.cross_attn.to(device).to(dtype)
    self.proj_layers.to(device).to(dtype)

    # Ensure medusa_head's dtype and device align with the base_model
    self.medusa_head.to(self.dtype).to(self.device)
    self.old_forward = self.forward

    # for i in range(medusa_num_heads):
    #     # Initialize the weights of each medusa_head using the base model's weights
    #     self.medusa_head[i][-1].weight.data[:] = self.lm_head.weight.data[:]

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        medusa_return: bool = False,
        medusa_only_heads: bool = False,
    ):
        """Forward pass of the MedusaModel.
        Returns:
            torch.Tensor: A tensor containing predictions from all Medusa heads.
            (Optional) Original predictions from the base model's LM head.
        """
        # LOG.debug("medusa_return: %s", medusa_return)
        if not medusa_return:
            return self.old_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # Pass input through the base model
        if medusa_only_heads:
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    # output_hidden_states=output_hidden_states,
                    output_hidden_states=True,
                    return_dict=return_dict,
                )
            hidden_states = outputs[0]
            out_0 = self.lm_head(hidden_states)
            medusa_logits = [out_0]

            all_layer_outputs = outputs.hidden_states
                # print("Number of layers:", len(all_layer_outputs))  # 打印层数
                # for i, layer_output in enumerate(all_layer_outputs):
                #     print(f"Layer {i} output shape:", layer_output.shape)
            x = 30  
                # 1. 提取后x层的输出
            last_x_layers = all_layer_outputs[-x:]  # 列表，包含x个 [1, 4096, 4096] 张量

                # 2. 对每层取最后一个token的隐藏状态 [:, -1, :]
            last_token_hidden_states = [layer[:, -1, :] for layer in last_x_layers]  # x个 [1, 4096] 张量

                # 3. 堆叠为 [1, x, 4096]
            merged_output = torch.stack(last_token_hidden_states, dim=1)  # [1, x, 4096]

                # 验证形状
                # print("合并后的形状:", merged_output.shape)  # 应输出 torch.Size([1, x, 4096])
                
        else:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            hidden_states = outputs[0]
            medusa_logits = [self.lm_head(hidden_states)]
        # for i in range(self.medusa_num_heads):
        #     medusa_logits.append(self.medusa_head[i](hidden_states))

        embedded = POS_embedding(out_0,out_0,0.8)
        for i in range(self.medusa_num_heads):
            query = self.proj_layers[i](embedded)
            SiLued = self.medusa_head[i](merged_output)
            predicted = self.cross_attn[i](query, SiLued)
            # print("predicted shape:", predicted.shape) #应该输出[1,seq_len,Voacb_size]
            medusa_logits.append(predicted)
            embedded = POS_embedding(predicted,embedded,0.8)
        # print("medusa_logits shape:", torch.stack(medusa_logits, dim=0).shape)#应该输出[medusa_num_heads+1,1,seq_len,Vocab_size]
        # if self.training:  # 仅在训练时检查
        #     for name, param in self.named_parameters():
        #         if param.requires_grad and "medusa" in name.lower():  # 只检查Medusa相关参数
        #             if param.grad is None:
        #                 print(f"[梯度检查] ❌ 参数无梯度: {name}")
        #             else:
        #                 grad_norm = param.grad.norm().item()
        #                 print(f"[梯度检查] ✅ {name}: 梯度范数={grad_norm:.6f}")
        return torch.stack(medusa_logits, dim=0)
    
    self.forward = types.MethodType(forward, self)

def replace_compute_loss(
    medusa_heads_coefficient,
    medusa_decay_coefficient, 
    medusa_scheduler="constant",
    medusa_logging=False,
    medusa_only_heads=False,
    medusa_distillation_regularization=0.0,
    medusa_self_distillation=False,
):
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute the training loss for the model.

        Args:
            model (torch.nn.Module): The model for which to compute the loss.
            inputs (dict): The input data, including input IDs, attention mask, and labels.
            return_outputs (bool): Whether to return model outputs along with the loss.

        Returns:
            Union[float, Tuple[float, torch.Tensor]]: The computed loss, optionally with model outputs.
        """
        if medusa_self_distillation:
            from peft.tuners.tuners_utils import BaseTunerLayer
            with torch.inference_mode():
                # Get the output of the original model for distillation
                for module in model.modules():
                    if isinstance(module, (BaseTunerLayer)):
                        module.enable_adapters(False)
                
                original_logits = model(
                    **inputs,
                    medusa_return=False,
                ).logits

                for module in model.modules():
                    if isinstance(module, (BaseTunerLayer)):
                        module.enable_adapters(True)

        logits = model(
            **inputs,
            medusa_return=True,
            medusa_only_heads=medusa_only_heads,
        )
        labels = inputs["labels"]
        # Shift so that tokens < n predict n
        loss = 0
        loss_fct = CrossEntropyLoss()
        log = {}
        medusa = logits.shape[0]
        for i in range(medusa):
            medusa_logits = logits[i, :, : -(1 + i)].contiguous()
            medusa_labels = labels[..., 1 + i :].contiguous()
            medusa_logits = medusa_logits.view(-1, logits.shape[-1])
            medusa_labels = medusa_labels.view(-1)
            medusa_labels = medusa_labels.to(medusa_logits.device)
            if i == 0:
                if medusa_self_distillation:
                    original_logits = original_logits[:, :-1].contiguous().view(-1, original_logits.shape[-1])
                    mask = medusa_labels.ne(IGNORE_TOKEN_ID)
                    soft_labels = F.softmax(original_logits[mask], dim=-1)
                    loss_i = F.kl_div(
                        F.log_softmax(medusa_logits[mask], dim=-1),
                        soft_labels,
                        reduction="sum",
                    ) / medusa_logits.shape[0]
                elif medusa_distillation_regularization > 0:
                    # use soft labels
                    mask = medusa_labels.ne(IGNORE_TOKEN_ID)
                    soft_labels = F.softmax(medusa_logits[mask], dim=-1) * medusa_distillation_regularization + \
                        F.one_hot(medusa_labels[mask], num_classes=medusa_logits.shape[-1]) * (1 - medusa_distillation_regularization)
                    loss_i = F.kl_div(
                        F.log_softmax(medusa_logits[mask], dim=-1),
                        soft_labels,
                        reduction="sum",
                    ) / medusa_logits.shape[0]
                else:
                    loss_i = loss_fct(medusa_logits, medusa_labels)
            else:
                loss_i = loss_fct(medusa_logits, medusa_labels)
            # Compute the coefficient for medusa losses
            if medusa_scheduler == "sine":
                medusa_scheduler_coefficient = math.sin(
                    self.state.global_step / self.state.max_steps * math.pi / 2
                )
            elif medusa_scheduler == "linear":
                medusa_scheduler_coefficient = (
                    self.state.global_step / self.state.max_steps
                )
            elif medusa_scheduler == "constant":
                medusa_scheduler_coefficient = 1
            elif medusa_scheduler.startswith("sine"):
                ratio = float(medusa_scheduler.split("_")[1])
                if self.state.global_step / self.state.max_steps < ratio:
                    medusa_scheduler_coefficient = math.sin(
                        self.state.global_step / self.state.max_steps / ratio * math.pi / 2
                    )
                else:
                    medusa_scheduler_coefficient = 1
            else:
                raise ValueError(
                    f"Invalid medusa_scheduler: {medusa_scheduler}. "
                    "Must be one of 'sine', 'linear', or 'constant'."
                )
            # Add decay coefficient to the loss
            if i == 0:
                if not medusa_only_heads:
                    loss += loss_i
            else:
                loss += loss_i * medusa_decay_coefficient ** i * medusa_heads_coefficient * medusa_scheduler_coefficient
            not_ignore = medusa_labels.ne(IGNORE_TOKEN_ID)
            medusa_labels = medusa_labels[not_ignore]

            # Add top-k accuracy
            for k in range(1, 10):
                _, topk = medusa_logits.topk(k, dim=-1)
                topk = topk[not_ignore]
                correct = topk.eq(medusa_labels.unsqueeze(-1)).any(-1)
                log[f"medusa{i}_top{k}"] = correct.float().mean().item()

            log[f"medusa{i}_loss"] = loss_i.item()
            log["medusa_scheduler_coefficient"] = medusa_scheduler_coefficient
        # self.log(log)
        # Add prefix to the log
        if model.training:
            prefix = "train"
        else:
            prefix = "eval"
        log = {f"{prefix}/{k}": v for k, v in log.items()}
        if medusa_logging and self.state.is_world_process_zero:
            # Hardcoded for now
            wandb.log({
                **log,
                "train/global_step": self.state.global_step,
            })
        return (loss, logits) if return_outputs else loss
    transformers.trainer.Trainer.compute_loss = compute_loss

def replace_create_optimizer(
    medusa_lr_multiplier,
):
    # Copy from transformers.Trainer.create_optimizer
    from transformers.trainer import is_sagemaker_mp_enabled, Trainer, ShardedDDPOption
    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

        if self.optimizer is None:
            print("啊毒品哈代得到大家")
            decay_parameters = self.get_decay_parameter_names(opt_model)
            print("看着利亚decay_parameters:", decay_parameters)
            print("\n===== 模型参数列表 =====")
            for name, param in opt_model.named_parameters():
                print(f"{name}: shape={tuple(param.shape)}, requires_grad={param.requires_grad}")
            # Separately set lr for medusa_head
            optimizer_grouped_parameters = [
            # 组1：Medusa相关参数（更高学习率）
                {
                    "params": [
                        p for n, p in opt_model.named_parameters()
                        if (p.requires_grad
                            and any(k in n for k in ["medusa_head", "cross_attn", "proj_layers"]))
                    ],
                    "weight_decay": self.args.weight_decay,
                    "lr": self.args.learning_rate * medusa_lr_multiplier,
                },
                # 组2：主干模型参数（需要weight decay）
                {
                    "params": [
                        p for n, p in opt_model.named_parameters()
                        if (p.requires_grad
                            and n in decay_parameters
                            and not any(k in n for k in ["medusa_head", "cross_attn", "proj_layers"]))
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                # 组3：其他无decay参数（如bias、LayerNorm）
                {
                    "params": [
                        p for n, p in opt_model.named_parameters()
                        if (p.requires_grad
                            and n not in decay_parameters
                            and not any(k in n for k in ["medusa_head", "cross_attn", "proj_layers"]))
                    ],
                    "weight_decay": 0.0,
                }
            ]

            print("\n===== 修正后的优化器参数分配检查 =====")
            total_params = set()
            for i, group in enumerate(optimizer_grouped_parameters):
                print(f"参数组 {i}: LR={group.get('lr', 'default')}, WD={group['weight_decay']}")
                print(f"  参数数量: {len(group['params'])}")
                
                # 打印前3个参数示例
                for p in group["params"][:3]:
                    name = [n for n, param in opt_model.named_parameters() if param is p][0]
                    print(f"  - {name}")
                    
                # 检查重复
                for p in group["params"]:
                    if p in total_params:
                        name = [n for n, param in opt_model.named_parameters() if param is p][0]
                        print(f"❌ 参数重复: {name}")
                    total_params.add(p)
            
            print(f"\n总可训练参数: {len(total_params)}")
            print(f"总模型参数: {sum(p.requires_grad for p in opt_model.parameters())}")
            
            # 检查是否所有参数都被分配
            if len(total_params) != sum(p.requires_grad for p in opt_model.parameters()):
                print("❌ 警告：有参数未被分配到任何组！")
                missing_params = [
                    n for n, p in opt_model.named_parameters() 
                    if p.requires_grad and p not in total_params
                ]
                print(f"未分配的参数: {missing_params[:5]}...")  # 打印前5个
            print("✅ 任务完成，程序退出")
            sys.exit(0)
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            if self.sharded_ddp == ShardedDDPOption.SIMPLE:
                self.optimizer = OSS(
                    params=optimizer_grouped_parameters,
                    optim=optimizer_cls,
                    **optimizer_kwargs,
                )
            else:
                self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
                if optimizer_cls.__name__ == "Adam8bit":
                    import bitsandbytes

                    manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                    skipped = 0
                    for module in opt_model.modules():
                        if isinstance(module, nn.Embedding):
                            skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                            logger.info(f"skipped {module}: {skipped/2**20}M params")
                            manager.register_module_override(module, "weight", {"optim_bits": 32})
                            logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                    logger.info(f"skipped: {skipped/2**20}M params")

        if is_sagemaker_mp_enabled():
            self.optimizer = smp.DistributedOptimizer(self.optimizer)

        print("\n===== 优化器参数分配检查 =====")
        total_params = set()
        for i, group in enumerate(optimizer_grouped_parameters):
            print(f"参数组 {i}: LR={group.get('lr', 'default')}, WD={group['weight_decay']}")
            for p in group["params"]:
                param_name = [n for n, param in opt_model.named_parameters() if param is p][0]
                print(f"  - {param_name}")
                if p in total_params:
                    print(f"❌ 参数重复分配: {param_name}")
                total_params.add(p)
        
        print(f"\n总可训练参数: {len(total_params)}")
        print(f"总模型参数: {sum(p.requires_grad for p in opt_model.parameters())}")
        
        return self.optimizer
    transformers.trainer.Trainer.create_optimizer = create_optimizer

    # Fix deepspeed's optimizer
    def deepspeed_init(trainer, num_training_steps, inference=False):
        """
        Init DeepSpeed, after updating the DeepSpeed configuration with any relevant Trainer's args.

        If `resume_from_checkpoint` was passed then an attempt to resume from a previously saved checkpoint will be made.

        Args:
            trainer: Trainer object
            num_training_steps: per single gpu
            resume_from_checkpoint: path to a checkpoint if to resume from after normal DeepSpeedEngine load
            inference: launch in inference mode (no optimizer and no lr scheduler)

        Returns: optimizer, lr_scheduler

        We may use `deepspeed_init` more than once during the life of Trainer, when we do - it's a temp hack based on:
        https://github.com/microsoft/DeepSpeed/issues/1394#issuecomment-937405374 until Deepspeed fixes a bug where it
        can't resume from a checkpoint after it did some stepping https://github.com/microsoft/DeepSpeed/issues/1612

        """
        from deepspeed.utils import logger as ds_logger
        from transformers.integrations.deepspeed import deepspeed_optim_sched

        model = trainer.model
        args = trainer.args

        hf_deepspeed_config = trainer.accelerator.state.deepspeed_plugin.hf_ds_config

        # resume config update - some bits like `model` and `num_training_steps` only become available during train
        hf_deepspeed_config.trainer_config_finalize(args, model, num_training_steps)

        # set the Deepspeed log level consistent with the Trainer
        ds_logger.setLevel(args.get_process_log_level())

        if inference:
            # only Z3 makes sense for the inference
            if not hf_deepspeed_config.is_zero3():
                raise ValueError("ZeRO inference only makes sense with ZeRO Stage 3 - please adjust your config")

            # in case the training config is re-used for inference
            hf_deepspeed_config.del_config_sub_tree("optimizer")
            hf_deepspeed_config.del_config_sub_tree("lr_scheduler")
            optimizer, lr_scheduler = None, None
            model_parameters = None
        else:
            trainer.optimizer = None  # important for when deepspeed_init is used as re-init
            self = trainer
            opt_model = model
            decay_parameters = self.get_decay_parameter_names(opt_model)
            model_parameters = [
                # 组1：仅主干模型参数（严格排除所有自定义模块）
                {
                    "params": [
                        p for n, p in opt_model.named_parameters()
                        if (n in decay_parameters 
                            and p.requires_grad
                            and "medusa_head" not in n
                            and "cross_attn" not in n 
                            and "proj_layers" not in n)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                # 组2：Medusa相关参数（包含所有自定义模块）
                {
                    "params": [
                        p for n, p in opt_model.named_parameters()
                        if (p.requires_grad
                            and ("medusa_head" in n 
                                or "cross_attn" in n 
                                or "proj_layers" in n))
                    ],
                    "weight_decay": self.args.weight_decay,
                    "lr": self.args.learning_rate * medusa_lr_multiplier,
                },
                # 组3：其他无decay参数（排除自定义模块）
                {
                    "params": [
                        p for n, p in opt_model.named_parameters()
                        if (n not in decay_parameters 
                            and p.requires_grad
                            and "medusa_head" not in n
                            and "cross_attn" not in n 
                            and "proj_layers" not in n)
                    ],
                    "weight_decay": 0.0,
                }
            ]
            
            # list(filter(lambda p: p.requires_grad, model.parameters()))
            optimizer, lr_scheduler = deepspeed_optim_sched(
                trainer, hf_deepspeed_config, args, num_training_steps, model_parameters
            )

        # keep for quick debug:
        # from pprint import pprint; pprint(config)

        return optimizer, lr_scheduler
    transformers.integrations.deepspeed.deepspeed_init = deepspeed_init
