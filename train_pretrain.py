import os
import platform # 获取与当前运行平台相关的系统信息；比如操作系统、硬件架构、python解释器等详细信息
import argparse
import time # 获取时间戳、延时执行等
import math # 
import warnings
import pandas as pd
import torch
import torch.distributed as dist
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, DistributedSampler
from contextlib import nullcontext # 作为占位上下文管理器，进入和退出不执行任何操作；

from transformers import AutoTokenizer

from model.model import MiniMindLM
from model.LMConfig import LMConfig
from model.dataset import PretrainDataset

warnings.filterwarnings('ignore')


def Logger(content): # 只有在非分布式训练模式或分布式训练模式下的主进程才会打印日志，防止淹没
    if not ddp or dist.get_rank() == 0:
        print(content)


def get_lr(current_step, total_steps, lr):
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))


def train_epoch(epoch, wandb):
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    start_time = time.time()
    for step, (X, Y, loss_mask) in enumerate(train_loader):
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device) # 因为会将token填充到同一个长度；对于填充部分的损失肯定要盖住；防止干扰

        # 在第0、1000、...step处打印X、Y的值
        # if step == 0:
        #     # print('x'*20, input_ids.shape)
        #     print('x'*20, input_ids, 'x'*20)
        #     # print('+'*15, X.shape)
        #     print('+'*15, X, '+'*15)
        #     # print('-'*10, Y.shape)
        #     print('-'*10, Y, '-'*10)
        #     # print('o'*5, loss_mask.shape)
        #     print('o'*5, loss_mask, 'o'*5)

        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with ctx: # 统一代码逻辑；避免冗余条件判断；安全退出上下文
            res = model(X)
            loss = loss_fct(
                res.logits.view(-1, res.logits.size(-1)), # size(-1)获取张量最后一个维度的大小；TODO debugg 那么综合来看就是（_，类别数）
                Y.view(-1) # TODO 可视化 这几个东西的形状 loss；res.logits.size(-1).shape;
            ).view(Y.size()) # 计算逐位置损失 loss_fct交叉熵损失函数
            loss = (loss * loss_mask).sum() / loss_mask.sum() 
            loss += res.aux_loss # 模型内部辅助损失
            loss = loss / args.accumulation_steps # 按梯度累积步数缩放损失

        scaler.scale(loss).backward() # 在混合精度中避免FP16数值范围不够导致的梯度下溢问题

        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer) # 梯度裁剪要对原来的梯度进行裁剪；要在clip_grad_norm前调用；
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip) # 梯度裁剪；限制所有的参数的梯度范数不超过args.grad_clip

            scaler.step(optimizer) # 更新模型参数，自动处理混合精度中的梯度反缩放；
            scaler.update() # 动态调整梯度缩放因子

            optimizer.zero_grad(set_to_none=True) # 清空梯度，为下一轮梯度累积做准备

        if step % args.log_interval == 0: # 日志输出
            spend_time = time.time() - start_time
            Logger(
                'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.12f} epoch_Time:{}min:'.format(
                    epoch + 1,
                    args.epochs,
                    step,
                    iter_per_epoch,
                    loss.item() * args.accumulation_steps,
                    optimizer.param_groups[-1]['lr'],
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60))

            if (wandb is not None) and (not ddp or dist.get_rank() == 0):
                wandb.log({"loss": loss.item() * args.accumulation_steps,
                           "lr": optimizer.param_groups[-1]['lr'],
                           "epoch_Time": spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60})

        if (step + 1) % args.save_interval == 0 and (not ddp or dist.get_rank() == 0): # 保存模型
            model.eval() # 确保保存模型时参数处于稳定状态；一般不需要，但是这么写会更健壮
            moe_path = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/pretrain_{lm_config.dim}01.pth' # TODO 改了下

            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()

            torch.save(state_dict, ckp)
            model.train()


def init_model(lm_config): # lm_config中模型配置对象，包含模型结构参数如层数、隐藏层维度等
    tokenizer = AutoTokenizer.from_pretrained('./model/minimind_tokenizer') # TODO 加载预训练的分词器；自动识别分词器类型，确保该路径下包含分词器必须的路径；文本转化为token id序列
    model = MiniMindLM(lm_config).to(args.device) # 初始化模型，根据配置创建自定义语言模型
    Logger(f'LLM总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')
    return model, tokenizer


def init_distributed_mode():
    if not ddp: return
    global ddp_local_rank, DEVICE

    dist.init_process_group(backend="nccl") # 初始化进程组；通信后端nccl
    ddp_rank = int(os.environ["RANK"]) # 全局进程编号
    ddp_local_rank = int(os.environ["LOCAL_RANK"]) # 当前节点内的本地GPU编号
    ddp_world_size = int(os.environ["WORLD_SIZE"]) # 总进程数（GPU总数）
    DEVICE = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(DEVICE) # 这两句将当前进程绑定到对应的本地GPU设备；确保每个进程使用独立的GPU，避免资源冲突


# torchrun --nproc_per_node 2 1-pretrain.py
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Pretraining")
    parser.add_argument("--out_dir", type=str, default="out")
    # 若要以最快速度实现zero则epochs设置为1轮；否则应当利用有限的数据训练2~6个epochs。
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32) # 所以X和Y以及loss_mask的形状是[32,511]
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--accumulation_steps", type=int, default=8)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_iters", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--dim', default=512, type=int)
    parser.add_argument('--n_layers', default=8, type=int)
    parser.add_argument('--max_seq_len', default=512, type=int)
    parser.add_argument('--use_moe', default=False, type=bool)
    parser.add_argument("--data_path", type=str, default="./dataset/pretrain_hq.jsonl")
    args = parser.parse_args()

    lm_config = LMConfig(dim=args.dim, n_layers=args.n_layers, max_seq_len=args.max_seq_len, use_moe=args.use_moe)
    args.save_dir = os.path.join(args.out_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    tokens_per_iter = args.batch_size * lm_config.max_seq_len
    torch.manual_seed(1337)
    device_type = "cuda" if "cuda" in args.device else "cpu"

    args.wandb_run_name = f"MiniMind-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"

    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast() # 如果是cpu设备就啥都不做；如果是gpu就后者

    ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
    ddp_local_rank, DEVICE = 0, "cuda:0"

    if ddp:
        init_distributed_mode()
        args.device = torch.device(DEVICE)

    if args.use_wandb and (not ddp or ddp_local_rank == 0):
        import wandb

        wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    else:
        wandb = None

    model, tokenizer = init_model(lm_config)
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=lm_config.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if ddp else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
        num_workers=args.num_workers,
        sampler=train_sampler
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    if ddp:
        model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])

    iter_per_epoch = len(train_loader)
    for epoch in range(args.epochs):
        train_epoch(epoch, wandb)
