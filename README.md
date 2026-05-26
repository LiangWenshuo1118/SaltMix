# SaltMix

面向熔盐热物性与输运性质的物理启发神经模型代码库。项目围绕密度、粘度、表面张力和电导率四类性质，使用一元/二元体系训练，在三元及以上体系上评估混合阶数外推能力。

## 项目内容

- `salt_property/`: 核心 Python 包，包含数据预处理、化学式解析、模型、训练、评估和推理入口。
- `configs/`: 各性质和模型变体的 YAML 配置。
- `data/`: 整理后的 Janz 相关 JSON 数据。Janz 页面 PNG、原始 PDF 和 MSTDB-TP 相关文件不随仓库发布。
- `tests/`: CLI、预处理、解析和模型行为的单元测试。
- `docs/`: 论文草稿、补充材料和结果文档。
- `outputs/`: 精选训练产物和处理后的结果汇总；远程 bundle 不随仓库发布。

## 模型概览

所有模型共享组分级 Transformer 编码器。每个盐组分会被解析为结构化 token，并与摩尔分数、温度一起输入模型。

支持的输出结构包括：

- `use_phy: true`, `use_rk: true`: composition-weighted + Redlich-Kister style pair correction。
- `use_phy: true`, `use_rk: false`: composition-weighted 模型。
- `use_phy: false`: Set Transformer pooling 后直接预测的无显式物理偏置模型。
- `use_linear: true`: 用于密度和表面张力的线性温度约束模型。
- `use_arrhenius: true`: 用于粘度和电导率的 Arrhenius 温度约束模型。

密度和表面张力在原始性质空间训练；粘度和电导率在自然对数空间训练，并在评估和预测时同时输出原始值与 log 值。

## 环境安装

本仓库目前没有打包为 `pip install` 项目，推荐直接从仓库根目录用 `PYTHONPATH` 运行。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-remote.txt
export PYTHONPATH="$PWD"
```

依赖列表很小：

- `numpy`
- `pandas`
- `PyYAML`
- `torch`

## 快速开始

以密度模型为例：

```bash
export PYTHONPATH="$PWD"

python -m salt_property preprocess \
  --config configs/density_config.yaml \
  --property density

python -m salt_property train \
  --config configs/density_config.yaml \
  --property density

python -m salt_property eval \
  --config configs/density_config.yaml \
  --property density
```

`train` 会在缺少预处理产物时自动先运行 `preprocess`。默认输出目录由配置文件中的 `training.output_dir` 决定。

## 支持的性质与配置

| Property | 默认 RK 配置 | 温度约束配置 | 默认输出目录 |
| --- | --- | --- | --- |
| `density` | `configs/density_config.yaml` | `configs/density_linear_config.yaml` | `outputs/density_results` |
| `viscosity` | `configs/viscosity_config.yaml` | `configs/viscosity_arrhenius_config.yaml` | `outputs/viscosity_results` |
| `surface_tension` | `configs/surface_tension_config.yaml` | `configs/surface_tension_linear_config.yaml` | `outputs/surface_tension_results` |
| `electrical_conductivity` | `configs/electrical_conductivity_config.yaml` | `configs/electrical_conductivity_arrhenius_config.yaml` | `outputs/electrical_conductivity_results` |

CLI 的 `--property` 只能使用：

```text
density
viscosity
surface_tension
electrical_conductivity
```

Arrhenius 变体仍使用原性质名。例如：

```bash
python -m salt_property train \
  --config configs/viscosity_arrhenius_config.yaml \
  --property viscosity
```

## 数据预处理

预处理会读取配置中的原始数据目录，例如：

```yaml
data:
  raw_density_dir: data/density
  validation_ratio: 0.2
  temperatures_per_record: 5
```

处理逻辑包括：

- 解析温度区间、组成区间和性质关联式。
- 过滤有机盐、氧化物、硫化物以及配置中显式排除的体系。
- 将一元和二元体系划分为训练集/验证集。
- 将三元及以上体系作为测试集，用于 mixture-order extrapolation。
- 按温度和组成采样展开 record-level 数据。
- 构建组分 token 词表。

预处理产物写入 `training.output_dir`：

- `train_samples.json`
- `val_samples.json`
- `test_samples.json`
- `filter_report.json`
- `vocab.json`
- `summary.json`

## 训练与评估产物

训练过程使用 AdamW、EMA 权重和 early stopping。训练与评估会生成：

- `best_model.pt`: 验证集最优 EMA checkpoint。
- `epoch_metrics.csv`: 每个 epoch 的训练/验证指标。
- `metrics.json`: 验证集和三元及以上测试集指标。
- `val_predictions.csv`: 验证集预测。
- `test_predictions.csv`: 测试集预测。

对于粘度和电导率，`metrics.json` 中会同时包含原始值空间指标和 log-space 指标。

本仓库只包含可公开保存的整理后 JSON 数据。MSTDB-TP 相关 CSV/JSON 文件未随仓库发布；如需运行依赖 MSTDB-TP 的补充分析，请从有授权的数据来源自行准备对应文件，并放回配置或脚本期望的位置。

## 推理

推理输入是 raw-data-style JSON 数组。预测时 `equation` 和 `unit` 可以留空，但 `system`、`composition` 和 `temperature` 必须存在。

示例 `examples/predict_input.json`：

```json
[
  {
    "index": "PRED_1",
    "data": {
      "system": "MgCl2-CaCl2-NaCl-KCl",
      "composition": "4.96-3.27-51.44-40.33 mol%",
      "equation": "",
      "temperature": "973 K",
      "unit": ""
    }
  }
]
```

运行预测：

```bash
python -m salt_property predict \
  --config configs/viscosity_config.yaml \
  --property viscosity \
  --input examples/predict_input.json \
  --output outputs/predict/viscosity_predictions.csv
```

如果未传 `--checkpoint`，默认读取配置输出目录中的 `best_model.pt`。

## 批量预测工具

`tools/` 中有面向论文最终模型目录的批量预测脚本。MSTDB-TP 输入路径仅作为示例，实际文件不随仓库发布：

```bash
python tools/predict_mstdb_density_models.py \
  --input data/density/density_raw.mstdb-tp-v4.json \
  --output outputs/mstdb_density_predictions.csv
```

这些脚本读取 raw-data-style JSON，并依赖特定的历史输出目录和 checkpoint 布局。MSTDB-TP 输入文件需要用户自行准备；如果只是训练新模型，优先使用 `python -m salt_property predict`。

## 测试

```bash
export PYTHONPATH="$PWD"
python3 -m unittest discover tests
```

测试覆盖：

- CLI 分发。
- 温度、组成和关联式解析。
- 组分化学式 token 化。
- RK、linear、Arrhenius 和 no-physics 模型分支。
- 预测输出字段和 log-space 指标。

## 相关文档

- [docs/Salt_en.md](docs/Salt_en.md): English manuscript draft.
- [docs/Salt_SI.md](docs/Salt_SI.md): English supporting-information draft.
- [docs/jctc-latex-template/manuscript_en.pdf](docs/jctc-latex-template/manuscript_en.pdf): English manuscript PDF.
- [docs/Janz-data-corrections.md](docs/Janz-data-corrections.md): Janz 数据修正记录。
- MSTDB-TP 数据未随仓库发布；使用前请从有授权的数据源自行准备。
