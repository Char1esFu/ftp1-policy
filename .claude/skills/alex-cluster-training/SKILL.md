---
name: alex-cluster-training
description: 在 NHR@FAU alex 集群（A40 节点, Slurm）上复现 FTP-1 UniVTAC 单任务 finetune：环境安装、HDF5→zarr 转换、norm stats、多卡 torchrun 训练、wandb 同步，以及已踩过的坑（NCCL P2P 卡死、T3 编码器尺寸、--overwrite 竞态、WORK 未设置）。在 alex 上提交/调试 FTP-1 训练作业、或训练在 DDP 初始化/加载权重/建目录阶段报错时使用。
---

# FTP-1 UniVTAC 训练 @ alex

目标：复现 `ftp1_univtac_finetune` 发布的每任务 checkpoint（论文 App. D：8 GPU，global batch 64，20k steps，lr 5e-5 → 5e-6）。

脚本在仓库根目录的 [slurm/](../../../slurm/)（随仓库提交，clone 后即可用）。`slurm/logs/`（作业日志）和 `slurm/configs/`（norm_stats 生成的数据集配置）是运行产物，被 `slurm/.gitignore` 排除。所有 sbatch 都要在仓库根目录提交。

| 文件 | 作用 |
|---|---|
| `common.sh` | 所有作业共用的环境：路径、uv env、代理、torch.compile 缓存、wandb |
| `setup_env.sh` | 登录节点一次性安装（uv sync、transformers_replace、预训练 ckpt、T3 编码器、tokenizer） |
| `convert_univtac.sbatch` | UniVTAC HDF5 → zarr（每任务 60 episode：0-49 训练，50-59 验证） |
| `norm_stats.sbatch` | 每任务 norm stats + `slurm/configs/dataset_univtac_<short>.json`，固定 train 0-49 / val 50-59 split |
| `train_univtac.sbatch` | 训练主脚本（自动续训，见下） |
| `probe_batch.sbatch` | 单卡跑几十步测显存峰值和步速 |
| `nccl_test.sbatch` | 2 卡 NCCL 连通性诊断（默认 / P2P 关 / 走 IB 三种对比） |
| `wandb_sync.sh` | 上传离线 wandb run（登录节点） |

## 布局（见 common.sh 注释）

- 代码：`~/Projects/ftp1-policy`；home 接近文件数配额，**不要往 home 写大量文件**。
- uv env：anvme workspace `wam-envs` 下 `ftp1/envs/ftp1`（workspace 过期需 `ws_allocate wam-envs 90`）。
- 原始数据：`$WORK/UniVTAC/data/<task>/clean/*.hdf5`，与 ACT / Dream-Tac 共享，**只读，不要改**。
- FTP-1 产物全在 `$FTP1_CACHE=$WORK/ftp1_cache`：zarr、assets（norm stats、T3 编码器）、checkpoints、pretrain_ckpt、wandb。
- 计算节点只能经代理上网：`http(s)_proxy=http://proxy.nhr.fau.de:80`（common.sh 已设置）。

## 工作流

```bash
# 登录节点
cd ~/Projects/ftp1-policy && bash slurm/setup_env.sh
# 以下都必须从 login shell 提交（见 Known Issues: WORK 未设置）
sbatch slurm/convert_univtac.sbatch put_bottle_in_shelf
sbatch slurm/norm_stats.sbatch put_bottle_in_shelf
sbatch slurm/train_univtac.sbatch put_bottle_in_shelf
bash slurm/wandb_sync.sh
```

从本地经 ssh 提交时用 `ssh alex "bash -lc 'cd ~/Projects/ftp1-policy; sbatch ...'"`。

`norm_stats.sbatch` 依赖本仓库对 `scripts/zarr_compute_norm_stats.py` 的修改（`FTP1_PRESERVE_SPLIT=1` 时不删已写好的 `train_val_split.json`）。

### train_univtac.sbatch

- 默认 `--gres=gpu:a40:8`（一个 a40 节点正好 8 卡），每卡 batch = `GLOBAL_BATCH / #GPU`。
- 可覆盖的环境变量：`NUM_TRAIN_STEPS`(20000) `SAVE_INTERVAL`(2000) `GLOBAL_BATCH`(64) `VAL_INTERVAL`(1000) `GRAD_CKPT`(false) `EXP_SUFFIX`。
- 续训：重新提交同一命令即可；有完整 step 目录就 `--resume`，否则从 `$PRETRAIN_CKPT` 全新开始（先 `rm -rf` 实验目录）。
- 每个 checkpoint 约 23 GB（权重 + AdamW），脚本只保留最新的一个和最终 step。

### 先小规模验证再上 8 卡

8 卡排队久，配置有改动时先跑小测试，**保持每卡 batch 8** 使显存有代表性：

```bash
NUM_TRAIN_STEPS=300 SAVE_INTERVAL=150 VAL_INTERVAL=150 GLOBAL_BATCH=16 EXP_SUFFIX=_test2gpu \
  sbatch --gres=gpu:a40:2 --time=01:30:00 slurm/train_univtac.sbatch put_bottle_in_shelf
```

2 卡能过 DDP 初始化 + 若干步，说明多卡通信正常；300 步内会走一遍 validation 和 checkpoint 保存。测完删掉 `$CHECKPOINT_BASE_DIR/ftp1/*_test*`。

### 监控

- 控制台只有 tqdm 进度条（`Training: …| n/20000 … loss=…`），loss 等指标进 wandb。日志含 `\r`，用 `tr '\r' '\n' < slurm/logs/train_<id>.out | grep Training: | tail -1`。
- 看运行中作业的 GPU：`srun --jobid=<id> --overlap nvidia-smi`。
- 开始时间估计：`sbatch --test-only ...`。估计偏悲观（按运行中作业的时限上限算），实际常早数小时；缩短 `--time` 对估计没帮助。

## 参考数值（put_bottle_in_shelf，每卡 batch 8，bf16，torch.compile，无 grad ckpt）

| 配置 | 步速 | 每卡显存 |
|---|---|---|
| 1×A40 | 1.26 s/it | ~36 GB |
| 2×A40（P2P 关） | 1.63 s/it | 38.7 GB / 49 GB |
| 8×A40（P2P 关） | ~2.3 s/it，20k 步约 13 h | 38.7 GB |

- 第 1 步包含 torch.compile，约 15 分钟，期间 GPU 利用率可能是 0%，进度条停在 0，属正常。
- loss 从 ~1.9 很快降到 0.2 附近（前 300 步）。
- validation 里 `jitter_rms_head-track-*` 等为 -1：该任务没有这些维度，正常。

## Known Issues

### 多卡在 DDP 初始化卡死 10 分钟后报 "DDP expects same model across all ranks … rank 1 has inconsistent 0 params"

- **根因**：a40 节点上 GPU 对分属不同 NUMA 节点（`nvidia-smi topo -m` 显示 `SYS`），NCCL 默认 P2P/CUMEM 传输在第一次集合通信就挂住（`last completed NCCL work: -1`），watchdog 600 s 超时。"0 params" 只是超时的副作用，不是模型不一致。
- **修复**：`train_univtac.sbatch` 在 torchrun 前 `export NCCL_P2P_DISABLE=1`（改走 SHM）。`NCCL_P2P_LEVEL=LOC NCCL_SHM_DISABLE=1`（走 IB）也能通。
- Dream-Tac 在 A100（NVLink）上没这个问题，所以不能照搬其“不设 NCCL 变量”的做法。
- 复查：`sbatch slurm/nccl_test.sbatch`，看每个 variant 的 `rank N OK` 和 `exit`（124 = 超时）。

### ViTEncoder state_dict size mismatch（加载 T3 mini.pth）

- 代码里 `T3_PRETRAINED_TACTILE_ENCODER_CHECKPOINTS_BASE_URL`（`ftp1_model_config.py`）指向 `t3_large/encoders/`（宽 1024），但模型的 ViTEncoder 是 768 宽 × 3 层，即 **T3-Medium**。
- **修复**：`$ASSETS_BASE_DIR/checkpoints_t3_encoder_base/mini.pth` 用 `FoundationTactile/models/t3_medium/encoders/mini.pth`（88 MB），`setup_env.sh` 已改。
- 这只是构造时的初始化：随后预训练 ckpt 的 `hpt_tokenizer/GelSightMini_image_224_224_3.safetensors` 会覆盖它（`load_hpt_tokenizers_strict=False`）。`trunk.pth` 上游 404，loader 会跳过，shared chunk 权重在 `model.safetensors` 里。

### FileNotFoundError：checkpoint 目录被删

- `zarr_train_ftp1_pytorch.py` 的 `--overwrite` 在**每个 rank** 上执行 `rmtree` 且没有 rank0 保护，慢 rank 会删掉快 rank 刚建好的目录。
- 作者脚本（`scripts_exp_zarr/univtac_example_real_machine/*.sh`）从不传 `--overwrite`。
- **做法**：不传 `--overwrite`；需要全新开始时在 sbatch 里、torchrun 之前 `rm -rf "$EXP_DIR"`。

### "WORK is not set; submit from a login shell"

- Slurm 作业继承提交时的环境；非登录 ssh（`ssh alex "sbatch …"`）没有 `$WORK`。
- **做法**：`ssh alex "bash -lc '…sbatch…'"`。

### 其他

- 上游 `uv.lock` 固定清华源，alex 上下载大 wheel 超时；本 fork 已提交 PyPI 版。`setup_env.sh` 里的替换保留，从 upstream 合并回清华源的 lock 后会再次生效。
- openpi 通过 gcsfs 拉 PaliGemma tokenizer 会失败，`setup_env.sh` 改用 HTTPS 直接下载。
- torch.compile / triton 缓存放 `$TMPDIR`（home 文件数配额），由 common.sh 设置。
- 6 卡不能整除 global batch 64，脚本会拒绝。
