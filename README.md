# RDPS

RDPS 是一个面向通信信号的研究项目，用于验证 Diffusion Posterior Sampling（DPS）和 Blind-DPS。项目先建立已知信道的可控基准，再研究未知信道联合推理，最后研究自适应 diffusion 推理步数。

## 项目边界

    src/
      rsig/
        config.py          # 对外配置对象
        generator.py       # 单条信号的完整生成流程
        sources.py         # 随机比特和模拟消息信源
        modulation.py      # 星座图、映射、调制与解调
        pulse_shaping.py   # 矩形脉冲和 RRC 发射成形
        channel.py         # identity、flat、FIR 和 selective fading
        impairment.py      # timing、SRO、CFO、phase 和 XO drift
        noise.py           # 无噪声及不同标定方式的 AWGN
        schema.py          # 单条样本结构和 IQ 转换
        seed.py            # 与遍历顺序无关的 seed 派生
        writer.py          # 数据写入和读取能力
      rdps/          # diffusion prior、DPS、Blind-DPS、训练和评估

src/rsig 不得依赖 src/rdps。rsig 是完整、独立的通信信号仿真工具；rdps 只消费其数据并实现恢复算法。当前只生成四种调制和简单已知信道，是完整能力的配置子集，不是另一套数据格式。

## rsig 能力范围

rsig 保留完整 RID2026 风格的 24 种调制：

    OOK, 4ASK, 8ASK,
    BPSK, QPSK, 8PSK, 16PSK, 32PSK,
    16APSK, 32APSK, 64APSK, 128APSK,
    16QAM, 32QAM, 64QAM, 128QAM, 256QAM,
    AM-SSB-WC, AM-SSB-SC, AM-DSB-WC, AM-DSB-SC,
    FM, GMSK, OQPSK

当前 rsig 将 timing、SRO、CFO 和 phase 定义为接收机侧误差，因此完整生成链为：

    信源与调制
    -> 发射脉冲成形
    -> channel
    -> timing offset / symbol-rate offset
    -> carrier-frequency offset / phase offset
    -> noise

这个顺序符合“传播信道后进入非理想接收机”的物理语义。每项损伤都必须可以独立启用或关闭。当前 DPS 数据集关闭 timing、SRO、CFO 和 phase，因此这些模块的先后顺序不会影响当前数据。

## 逆问题定义

数据必须满足可重放的 forward model：

    y_rx = A_theta(x_tx) + noise

其中 x_tx 是实际发射波形，y_rx 是接收波形，theta 包含信道及所有启用的损伤参数。仅启用信道时，theta = h。

固定长度 inverse problem 的多径信道默认使用因果零边界：

    y_clean = np.convolve(x_tx, h, mode="full")[:len(x_tx)]
    y_rx = y_clean + noise

数据生成和 rdps forward operator 必须使用完全相同的边界约定。

## 统一数据集规范

完整 RID2026 风格数据和当前四种调制数据使用同一套 HDF5 schema，只改变配置和维度大小。

    dataset.h5
    ├── modulation                  UTF-8   [M]
    ├── noise_level_db              float32 [S]
    ├── x_tx                        float32 [M, S, E, 2, T]
    ├── h                           float32 [M, S, E, 2, L]
    ├── y_rx                        float32 [M, S, E, 2, T]
    ├── parameters/
    │   ├── samples_per_symbol      uint16  [M, S, E]
    │   ├── rrc_alpha               float32 [M, S, E]
    │   ├── timing_offset           float32 [M, S, E]
    │   ├── symbol_rate_offset      float32 [M, S, E]
    │   ├── phase_offset            float32 [M, S, E]
    │   ├── carrier_offset          float32 [M, S, E]
    │   ├── delay_spread            float32 [M, S, E]
    │   ├── channel_path_count      uint16  [M, S, E]
    │   ├── channel_length          uint16  [M, S, E]
    │   ├── noise_variance          float32 [M, S, E]
    │   ├── noise_std               float32 [M, S, E]
    │   ├── x_tx_scale              float32 [M, S, E]
    │   └── sample_seed             uint64  [M, S, E]
    └── metadata                    JSON

M 为调制种类数，S 为噪声等级或其他实验条件数，E 为每个条件的样本数，T 为波形长度，L 为最大 channel taps 数，2 为 I/Q 两个实数通道。

数组坐标必须严格对应：

    x_tx[m, s, e]
    h[m, s, e]
    y_rx[m, s, e]
    parameters/name[m, s, e]

共同描述 modulation[m]、noise_level_db[s] 条件下的第 e 条样本。

x_tx 是 diffusion prior 的训练目标；h 是该样本完整的已知信道 taps，不足 L 的部分补零；y_rx 是使用保存参数对 x_tx 应用 forward model 后得到的观测。信号布局为 [M, S, E, 2, T]，信道布局为 [M, S, E, 2, L]。

parameters 保存每条样本实际使用且可能变化的仿真参数。仅保存 h 无法表达 timing offset、SRO、phase、CFO 和噪声方差，因此这些字段必须保留。关闭的损伤使用恒等值，例如 offset 为 0、noise variance 为 0。

全局不变设置保存在 metadata JSON 中，至少包括 format/version、generation mode、调制列表、每条件样本数、frame length、最大 taps、channel 类型、boundary、channel normalization、noise calibration、impairment 顺序、IQ 布局、root seed 和 seed 派生规则。

## 明确不保存的内容

- splits：数据划分属于 loader 和训练实验，不属于仿真数据；
- source_bits、source_symbols、source_message：当前目标是 waveform 恢复，不进行 BER/SER 评估；
- 每条样本的 JSON 配置：变化参数进入 parameters，全局参数进入 metadata；
- Python object 或 pickle 数据。

loader 根据 modulation_index、condition_index、example_index 进行确定性、分层划分；训练实验记录 split seed、比例和算法版本。

## 当前数据集配置

当前数据集使用完整 schema，只启用 BPSK、QPSK、16QAM、32QAM 和已知 channel；timing offset、symbol-rate offset、phase offset、carrier-frequency offset、noise 暂时关闭。关闭噪声时 S = 1，noise_level_db[0] = NaN，并满足 y_rx = A_h(x_tx)、noise_variance = 0、noise_std = 0。

## DPS 训练与推理约定

DPS 由“信号先验训练”和“观测条件下的后验采样”两部分组成，不是针对每条观测单独训练一个模型。

### 训练 diffusion signal prior

训练数据只使用 x_tx，输入模型的张量通常为 [batch, 2, T]。训练目标是学习 clean transmitted waveform 的分布。可以比较：

1. 每种调制分别训练一个 prior；
2. 四种调制混合训练一个无条件 prior；
3. 四种调制混合训练一个 modulation-conditioned prior。

这些是实验设置，不是三套 DPS 算法。

### known-channel DPS

推理时读取 y_rx、h 和 parameters；x_tx 只用于计算 NMSE、MSE、EVM 等评估指标，不能泄漏给 sampler。DPS 每一步结合 diffusion prior 更新和 likelihood 数据一致性梯度。只含已知线性信道和 AWGN 时：

    y = A_h(x) + n
    grad_x log p(y | x, h) = A_h^*(y - A_h(x)) / noise_variance

因此 rdps 必须实现与 rsig 完全一致的 A_h 和伴随算子 A_h^*。

### Blind-DPS

Blind-DPS 将 h 也作为未知变量，目标变为 p(x, h | y)。它需要信道先验、归一化和尺度/相位约束。实现顺序是：先完成 known-channel DPS，再做 pilot-assisted 或受约束的交替更新，最后再考虑联合随机采样。

## 研究顺序

1. 固定数据 schema、信号表示、channel operator 和指标；
2. 用 Gaussian 线性逆问题验证 DPS；
3. 用当前四种调制和已知信道验证 known-channel DPS；
4. 与 ZF、MMSE/Wiener 等经典基线比较；
5. 引入 Blind-DPS；
6. 在固定推理步数可靠后研究自适应 diffusion 步数。

## Known-channel DPS 实现

当前 `src/rdps` 已实现第一阶段 known-channel DPS：

    src/rdps/data.py        HDF5 loader 与确定性分层划分
    src/rdps/model.py       带 timestep embedding 的多尺度 1D U-Net
    src/rdps/diffusion.py   DDPM epsilon 训练、祖先采样和合法跳步后验
    src/rdps/operators.py   batch 复数 FIR forward/adjoint
    src/rdps/dps.py         measurement-guided posterior sampling
    src/rdps/evaluation.py  固定 validation 与平衡、可复现的 DPS 评估
    src/rdps/checkpoint.py  配置指纹、随机状态和原子 checkpoint
    src/rdps/progress.py    原地刷新的 epoch/batch 训练进度
    src/rdps/training.py    epoch 训练、Early Stopping 和目录级 resume
    src/rdps/metrics.py     waveform MSE、NMSE 和 measurement residual

模型只以 `x_tx [B,2,T]` 训练无条件 prior。`h` 和 `y_rx` 仅在 DPS 推理时使用，
不会输入 diffusion model。loader 对 `example_index` 做由 `split_seed` 决定的置换，
并把同一组 example split 应用于每个 `(modulation, condition)` 分层；划分算法版本、
seed、比例和实际 example indices 都会写入运行目录。

known-channel operator 严格计算

    np.convolve(x, h, mode="full")[:T]

对应的复数因果零边界卷积，同时提供满足内积关系的伴随算子。实现没有把 I/Q 当成
两个独立通道。无噪声样本的 DPS measurement loss 为

    ||y_rx - A_h(x_0_hat)||^2 / (||y_rx||^2 + eps)

不会除以零噪声方差。`dps.likelihood: gaussian` 为后续 AWGN 数据提供
`||residual||^2 / (2 * noise_variance)`；`auto` 会逐样本判断，因此同一 batch 可以安全
包含零噪声与非零 AWGN 样本。

新训练会创建带 UTC 时间戳的独立目录；resume 则继续使用原目录。训练以 epoch 为单位，
每个 epoch 使用固定 validation seed 和固定 `(t, epsilon)` 条件比较 loss，并在 epoch
结束保存 `last.pt`。`best.pt` 只由 validation loss 决定，Early Stopping 的 patience
状态随 checkpoint 恢复。周期 DPS 只访问 validation split；训练或 Early Stopping 完成
后才加载 `best.pt` 并访问 test split。

训练目录包含实际 `config.yaml`、`split.json`、`validation_selection.json`、
`history.jsonl`、`metrics.json`、checkpoint 以及 validation/test evaluation 子目录。
推理目录包含实际配置、确定性选择坐标、逐样本/per-modulation/macro/micro 指标、
`reconstructions.h5` 和少量时域、频谱、IQ 散点对比图。这里定义

    MSE = mean(|x_tx - x_reconstructed|^2 over I/Q values)
    NMSE = ||x_tx - x_reconstructed||^2 / ||x_tx||^2
    measurement residual = ||y_rx - A_h(x_reconstructed)||^2 / ||y_rx||^2

实现依据 DiffCom 的标准 posterior sampling：先从 epsilon prediction 得到
`x_0_hat`，执行无条件祖先更新，再减去 measurement loss 对当前 `x_t` 的梯度。
当前范围不含 HiFi-DiffCom、Blind-DPS 或自适应 timestep。

评估按 modulation 均分样本配额，再在每个 modulation 内按 condition 均分；stratum
内部由 evaluation seed 确定性抽样。正式汇总以 modulation macro mean 为主，同时保存
普通 sample micro mean。若样本数少于 modulation 数量，程序会明确警告覆盖不完整。

resume 配置使用 `training.resume_output_dir`。checkpoint 会校验数据集 signature、split、
训练 seed、模型、diffusion、batch size、optimizer 超参数、固定 validation 设置和 Early
Stopping 设置；只有 epochs、device、worker、显示与纯评估设置可以调整。旧的 step-based
format-v1 checkpoint 不具备完整 epoch/patience 状态，因此会被明确拒绝，避免伪精确续训。
resume 应只用于中断恢复，并在正式实验开始时设定完整 epoch 上限；不要查看 test 结果后再
提高 epoch 上限，否则会把 test 信息反馈到训练决策中。跨 CPU/CUDA/MPS 恢复虽然允许，
但浮点实现和设备随机数流不同，不能视为逐位一致的训练轨迹。

参考：

- https://github.com/wsxtyrdd/diffcom
- https://arxiv.org/abs/2406.07390

## 运行 known-channel DPS

安装当前包后，可直接使用默认服务器配置：

    conda run -n rfsig python -m pip install -e .
    conda run -n rfsig python train_dps.py --config configs/dps_v1.yaml --device cuda

如果不安装 editable package，可在命令前设置 `PYTHONPATH=src`。从 checkpoint 推理：

    conda run -n rfsig python run_dps.py \
      --checkpoint outputs/train/<run>/checkpoints/best.pt \
      --dataset datasets/dps_v1/dps_v1.h5 \
      --split test --num-samples 40 --sampling-steps 250 \
      --guidance-scale 0.05 --evaluation-seed 30233 \
      --batch-size 4 --device cuda

`device: auto` 的优先级是 CUDA、Apple MPS、CPU。由于模型和 measurement autograd
基于 PyTorch，命令行的 `--device mlx` 是 Apple 设备上的 MPS 别名，并不启用另一套
MLX 模型实现。

本机 smoke 配置每个 epoch 只包含两个 batch：

    conda run -n rfsig python genDS_dps_v1.py \
      --output datasets/smoke/dps_smoke.h5 --examples-per-modulation 4 \
      --frame-length 64 --samples-per-symbol 4 --rrc-span-symbols 5 \
      --guard-samples 16 --channel-length 4 --validation-random-checks 4 \
      --plots-per-modulation 1 --progress-every 4 --overwrite
    conda run -n rfsig python train_dps.py --config configs/dps_smoke.yaml
    conda run -n rfsig python train_dps.py --config configs/dps_smoke.yaml \
      --resume-output outputs/smoke/train/<run> --epochs 2
    conda run -n rfsig python run_dps.py \
      --checkpoint outputs/smoke/train/<run>/checkpoints/best.pt \
      --dataset datasets/smoke/dps_smoke.h5 --split test --num-samples 4 \
      --sampling-steps 4 --guidance-scale 0.05 --evaluation-seed 3017 \
      --batch-size 2 --device cpu \
      --output outputs/smoke/dps

当前 macOS `rfsig` 环境同时加载了多个 OpenMP runtime。本机若遇到重复 runtime 或
`pthread_mutex_init` 错误，可仅对 smoke 命令设置：

    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MPLCONFIGDIR=/tmp/rdps-mpl

服务器环境应先正常安装匹配的 PyTorch/CUDA 依赖，不应默认依赖上述兼容变量。

## 实验复现规则

每个实验必须记录数据集路径、schema 版本、完整配置、root seed、loader split seed、模型 checkpoint、评价指标和输出目录。结果必须能够依据保存的数据、配置和 seed 复现。
