# Diffusion 项目结构设计说明

目标：比较 DDPM、DDIM、SDE、NormFlow、Flow Matching、Reflect Flow、MeanFlow 等模型，**数据/工具/输出** 统一，**前向/调度/网络** 尽量复用。

---

## 一、顶层目录结构

```
Diffusion/
├── data/                    # 数据目录（原始数据、预处理缓存）
├── utils/                   # 通用工具：读数据、日志、可视化
├── output/                  # 所有结果：checkpoint、生成图、曲线、对比表
├── models/                  # 去噪网络（UNet）：只预测噪声 ε，与具体过程解耦
├── diffusion/                 # 各扩散过程：base + 可复用 schedule + 各方法（仅逆向）
│   ├── base/                # 前向/逆向抽象接口（ForwardResult, DiffusionProcess）
│   ├── schedule/            # 可复用加噪（与 ddpm、ddim、sde 同级）：β、α_bar、forward_step
│   ├── ddpm/                # DDPM 逆向，前向用 diffusion.schedule
│   ├── ddim/, sde/, …       # 各方法只做逆向，前向复用 schedule
├── scripts/                 # 入口脚本：train_xx.py / sample_xx.py
├── configs/                 # 配置（可选）
└── docs/                    # 文档与对比结果说明（可选）
```

---

## 二、各目录职责与复用关系

### 1. `data/`
- 存放数据集路径或下载目录（如 `cifar100/`、`celeba/`）。
- 不写业务逻辑，只当“数据根目录”；具体怎么读在 `utils/dataloader.py`。

### 2. `utils/`
- **`dataloader.py`**：统一的数据接口。根据数据集名返回 `DataLoader`，包含 `transform`（如归一化到 [-1,1]）。
- **`logging_utils.py`**：训练日志、loss 曲线、保存 checkpoint 的路径约定。
- **`image_utils.py`**：保存网格图、FID 等（可选）。
- 所有方法都从这里读数据和写日志，保证“同一份数据、同一套评估”。

### 3. `output/`
- 按方法分子目录，例如：
  - `output/ddpm/`, `output/ddim/`, `output/sde/`, `output/flow_matching/` …
- 每个方法下可再分：`checkpoints/`, `samples/`, `curves/`, `metrics/`。
- 脚本里通过 `utils.logging_utils` 或统一配置决定路径，便于做对比实验。

### 4. `methods/base/`（前向/逆向接口）
- **`base.py`**：
  - 定义抽象接口：`ForwardResult`、`DiffusionProcess`（`forward_step` / `reverse_step` / `sample_loop`）；
  - 不绑定具体公式，只约定输入输出，方便各 method 继承或实现同一套接口。
- 时间嵌入在 **`models/time_embed.py`**，由 UNet 等网络使用。
- 各方法（ddpm、ddim、sde、flow_matching 等）在 `methods/<name>/` 中实现具体调度与采样，从 `methods.base` 引用基类。

这样：**加噪** 由 **`methods/schedule/`** 统一提供（可复用），**逆向** 由各 method 自己实现。

### 5. `models/`（与过程解耦，只预测噪声）
- **`unet.py`**：统一 UNet，输入 `(x, t)`，**只预测噪声 ε**；时间嵌入在 `time_embed.py`。
- **`time_embed.py`**：`SinusoidalPosEmb`，供 UNet 对 t 编码。
- 各扩散过程共用同一 model；需要 score 时在过程内用 ε 换算（如 score = -ε/√(1-ᾱ_t)）。

### 6. `methods/`（schedule 与 ddpm、ddim、sde 同级，可复用）
- **`methods/base/`**：抽象接口 `ForwardResult`、`DiffusionProcess`，无时间嵌入。
- **`methods/schedule/`**（与 ddpm、ddim、sde **同级**）：**可复用的加噪**。
  - **`DDPMSchedule`**：β、α_bar、`forward_step`、`get_score_target`；**无逆向**。
  - DDPM、DDIM、SDE(VP) 等共用同一套 α_bar，前向都调 `schedule.forward_step`。
- **各方法**只实现 **逆向**（继承 `DiffusionProcess`，`forward_step` 委托给 `methods.schedule`）：
  - **`methods/ddpm/`**：`DDPMProcess`，逆向 p_sample。
  - **`methods/ddim/`**：用同一 `DDPMSchedule`，逆向为 DDIM 确定性子序列。
  - **`methods/sde/`**：`SDEProcess`，用同一 `DDPMSchedule`，逆向欧拉；`get_score_target` 来自 schedule。
  - **normflow/**、**flow_matching/** 等：可复用或自定 schedule，只写逆向。

**schedule 与 ddpm、ddim、sde 同级**，专门放可复用的加噪；model 独立在 `models/`，只预测噪声。

### 7. `scripts/`
- **`train_ddpm.py`**, **`train_ddim.py`**, **`train_sde.py`**, **`train_flow_matching.py`** …
- **`sample_ddpm.py`**, **`sample_ddim.py`** …
- 每个脚本：解析参数 → 调 `utils.dataloader` 取数据 → 调对应 `methods.xxx` 训练/采样 → 结果写到 `output/xxx/`。
- 可选：一个 `scripts/run_all.py` 按配置依次跑各方法，便于做对比。

### 8. `configs/`（可选）
- 每个方法一个 yaml/json，如 `ddpm.yaml`、`flow_matching.yaml`，写清数据路径、T、batch、lr、output 子目录等，脚本读配置，避免硬编码。

---

## 三、数据流与复用小结

| 层次       | 内容           | 复用方式 |
|------------|----------------|----------|
| 数据       | `data/` + `utils/dataloader.py` | 所有方法同一套读数据 |
| 输出       | `output/<method>/` | 统一目录，便于对比 |
| 加噪       | `methods/schedule/`（与 ddpm、ddim、sde 同级） | DDPMSchedule 可被 DDPM/DDIM/SDE 复用 |
| 逆向       | `methods/ddpm/`、`methods/sde/` 等 | 各方法实现 DiffusionProcess，forward 委托 schedule |
| 去噪网络   | `models/unet.py` | 只预测噪声 ε；各过程内部按需换算 score/v |

这样你可以：
- **复用它**：数据、UNet（只预测噪声）、时间嵌入；各方法只换 `methods/<name>/schedule`。
- **对比它**：同一 data、同一 model、同一 output 目录，不同方法 = 不同 DiffusionProcess（schedule）。

---

## 四、迁移建议

1. **models/unet.py**：实现统一 UNet（输入 x,t，输出噪声 ε），用 `models.SinusoidalPosEmb`。
2. **methods**：`methods/schedule/` 提供可复用加噪（`DDPMSchedule`）；各方法（ddpm、sde、ddim…）实现 `DiffusionProcess`，前向用 `schedule.forward_step`，只写逆向。训练用 `process.forward_step` 取 xt/noise，loss 对 ε；采样用 `process.sample_loop(model, shape)`。
3. **scripts**：`train_xx.py` 用 `utils.get_dataloader`、`utils.get_save_dir`，实例化对应 `methods.xxx` 的 schedule 和 `models` 的 UNet，写 checkpoint 到 `output/<method>/`。
