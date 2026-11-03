# SpaGenX完整使用说明（PyTorch命令版）

所有命令从仓库根目录运行，并将`/external/...`替换为实际路径。

```bash
git clone https://github.com/JMJ-rbgss/SpaGenX.git
cd SpaGenX
```

SpaGenX提供配置化预处理、原生训练和WSI推理入口。

## 1. 选择使用路线

| 目的 | 所需输入 | 使用入口 |
| --- | --- | --- |
| 对新H&E/WSI预测基因表达 | 图像、MPP、器官、基因列表、冻结资产 | `scripts/infer_wsi.sh` |
| 对已有可信特征预测 | 特征、匹配邻域、器官和基因列表 | `inference/spagenx_cli.py infer` |
| 构建研究训练数据 | 图像、实测表达、spot坐标、冻结划分和元数据 | `preprocessing/run.py` |
| 从零训练SpaGenX | 完整训练资产、GPU | `training/run.py`或`torchrun` |

推理需准备冻结权重和元数据；训练还需配对数据；完整恢复状态仅用于断点续训。资产准备见第3节，训练输入见第7节。

## 2. 环境安装

推荐使用Linux、Python3.9、PyTorch2.6.0环境。

### 2.1 创建独立环境

```bash
conda create -n spagenx python=3.9 -y
conda activate spagenx
python -m pip install --upgrade pip
```

### 2.2 安装PyTorch（GPU与CPU二选一）

Linux/NVIDIA，CUDA11.8构建示例：

```bash
python -m pip install torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu118
```

只进行CPU验证：

```bash
python -m pip install torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cpu
```

按[PyTorch官方版本说明](https://pytorch.org/get-started/previous-versions/)选择与驱动兼容的构建。

### 2.3 安装其他依赖及OpenSlide

```bash
python -m pip install -r requirements.txt
python -m pip install anndata==0.10.8 Pillow==11.3.0
```

OpenSlide需要Python绑定和底层动态库；若未安装动态库，可按[OpenSlide官方说明](https://openslide.org/api/python/)选择适合平台的方式。例如适用平台可安装：

```bash
python -m pip install openslide-bin
```

检查：

```bash
python -c "import torch, torchvision, timm, openslide, anndata; print('torch:', torch.__version__); print('CUDA:', torch.version.cuda); print('GPU available:', torch.cuda.is_available())"
python -m pip check
```

按上述命令创建独立环境。记录的验证环境版本见`docs/validation.json`。

## 3. 准备冻结推理资产

### 3.1 推荐目录

```text
repository/
├── preprocessing/
├── training/
├── inference/
├── scripts/
├── configs/
├── bundle_config.json          # 本地配置，不提交GitHub
└── external_assets/            # 外部获取，不提交GitHub
    ├── SpaGenX_weights.pt
    ├── uni2h/pytorch_model.bin
    ├── gene_to_program_m16_overlay.pt
    ├── global_gene_vocab.tsv
    └── codebooks.json
```

SpaGenX冻结资产请向项目维护者申请，下载链接待补充。UNI2-h从[官方模型页面](https://huggingface.co/MahmoodLab/UNI2-h)按其访问与许可条件获取。

### 3.2 配置资产路径

```bash
cp configs/bundle.example.json bundle_config.json
```

用编辑器修改`bundle_config.json`中的`paths`。路径可以相对仓库根目录，也可以是绝对路径。保持`model_code`指向`inference/spagenx_model.py`，`preprocess_code`指向`preprocessing/step1_extract_uni2h_features.py`。不要修改哈希值来绕过权重检查。

当前正式身份：

| 资产 | SHA256 |
| --- | --- |
| SpaGenX权重 | `0bcb29527826a54f6be4ffd1e9140fc35167d6b48b9ced5ff21ab98e60cc7603` |
| UNI2-h | `6e077eda234bebc595868d918d3458d9dd32a050199b0ff04443b2f46a0a3b1e` |

其余冻结资产按`docs/external_assets.json`中的SHA256校验。

### 3.3 查看支持的器官

```bash
python - <<'PY'
import json
from pathlib import Path
root = Path.cwd()
cfg = json.loads((root / 'bundle_config.json').read_text())
p = Path(cfg['paths']['codebooks'])
if not p.is_absolute():
    p = root / p
print('\n'.join(json.loads(p.read_text())['organ_id']))
PY
```

使用冻结codebook的17类之一，保留原编号，并通过`--organ`指定器官。CLI匹配名称时忽略大小写。

## 4. 快速开始：自动WSI推理

### 4.1 准备基因清单

创建一个UTF-8文本文件，每行一个符号或一个`global_gene_id`，不带表头。例如：

```text
VIM
COL1A1
CD74
```

查询基因须存在于冻结词典；符号歧义时使用明确ID。基因清单按身份去重，同一基因以符号或ID输入一次。

### 4.2 一次设置输入

```bash
export CUDA_VISIBLE_DEVICES=4
export PYTHON_BIN=python
export WSI=/external/slides/example.svs
export ORGAN=Breast
export GENES_FILE=/external/query_genes.txt
export RUN_DIR=/external/spagenx_runs/example_run_001
export DEVICE=cuda:0
```

`CUDA_VISIBLE_DEVICES=4`选择物理GPU4，进程内使用`DEVICE=cuda:0`。CPU模式设置`DEVICE=cpu`；WSI特征提取与推理推荐使用GPU。

若图像没有可信MPP元数据，显式设置：

```bash
export MPP=0.5
```

MPP须使用**输入图像的真实像素尺度**。图像元数据已有可靠MPP时无需设置；清除旧设置可运行`unset MPP`。

### 4.3 预览与执行

```bash
bash scripts/infer_wsi.sh --dry-run
bash scripts/infer_wsi.sh --execute
```

脚本检查文件、配置及全新`RUN_DIR`，依次执行图像切块、UNI2-h编码、邻域构建和SpaGenX推理。第一阶段失败时停止。日志位于`preprocess.log`和`inference.log`。

固定应用协议：0.5µm/px、224×224、不重叠网格；按RGB均值≥205识别白色背景，剔除白色比例>0.8的patch。最多接受16000个有效patch。缺少MPP、空白图、损坏图像、超过patch上限或输出已存在时停止。超过6400万像素的普通栅格图需转换为金字塔WSI后处理。

### 4.4 等价的逐步PyTorch命令

```bash
CUDA_VISIBLE_DEVICES=4 python inference/spagenx_cli.py --bundle . preprocess \
  --image /external/slides/example.svs \
  --out /external/spagenx_runs/stepwise_001/preprocessed \
  --device cuda:0

CUDA_VISIBLE_DEVICES=4 python inference/spagenx_cli.py --bundle . infer \
  --features /external/spagenx_runs/stepwise_001/preprocessed/features.pt \
  --neighbors /external/spagenx_runs/stepwise_001/preprocessed/neighbors.pt \
  --genes-file /external/query_genes.txt --organ Breast \
  --out /external/spagenx_runs/stepwise_001/predictions \
  --patch-batch 16 --gene-chunk 256 --device cuda:0
```

对缺少MPP的图片，在第一条命令补`--mpp <真实值>`。已有可信特征及匹配邻域时，直接运行第二条命令。

推理配置为FP32、gene bias开启、tech_zero、no_prompt。`gene-chunk`控制计算分块，程序按输入顺序预测全部查询基因。

## 5. 输出与读取结果

```text
RUN_DIR/
├── preprocess.log
├── inference.log
├── preprocessed/
│   ├── features.pt
│   ├── neighbors.pt
│   ├── patches.tsv
│   └── preprocess.json
└── predictions/
    ├── predictions.npz
    └── inference.json
```

`predictions.npz`包含：

| 字段 | 内容 |
| --- | --- |
| `pred_log1p` | patch×查询基因的预测矩阵，顺序对应以下字段 |
| `global_gene_ids` | 有序冻结基因ID |
| `gene_symbols` | 同序基因符号 |
| `coords_level0_center` | level0像素坐标下的patch中心 |
| `patch_ids` | 与预测行一一对应的patch身份 |

```bash
python - <<'PY'
import numpy as np
p = '/external/spagenx_runs/example_run_001/predictions/predictions.npz'
with np.load(p, allow_pickle=False) as z:
    x = z['pred_log1p']
    print('patches × genes:', x.shape)
    print('genes:', z['gene_symbols'].tolist())
    print('finite:', bool(np.isfinite(x).all()))
    print('first patch:', z['patch_ids'][0], z['coords_level0_center'][0])
PY
```

`pred_log1p`为patch×gene表达预测矩阵，采用log1p原始计数尺度；相关字段包含有序基因ID、符号及patch身份和坐标。

`inference.json`记录SpaGenX权重身份、no_prompt/gene bias状态、输入哈希、器官、精度及分块参数。保存它与图像、MPP、基因顺序的来源记录，以便追溯。

## 6. 研究数据预处理

此路线使用配对图像、实测表达和spot坐标，按spot构建特征与邻域；第4节WSI流程使用规则网格。

### 6.1 输入合同

准备可核验的样本inventory、图像路径、表达来源、spot身份及坐标、器官/技术标签和冻结划分。inventory中图像使用`wsi_path`或`image_path`；坐标来自`h5ad_path`或`coord_path`，具体格式由原始加载器决定。不同数据源不能只改扩展名后混用。

rowmap必须按样本与spot ID精确匹配，不按近似坐标或排序位置猜测。measured zero保留；未实测基因不填成0。canonical编号、Program256/M16和器官codebook须相互一致。详细格式见[数据合同](DATA_CONTRACT.md)。

### 6.2 编辑配置

```bash
cp configs/preprocessing.example.json configs/preprocessing.local.json
```

逐项修改`data_root`、物理GPU和每个stage的输入/输出参数。示例中各路径是占位符，**不会自动把上一stage的输出接到下一stage**。你必须核对manifest及内部路径并更新后续输入。

### 6.3 可自动执行的阶段

```bash
python preprocessing/run.py --stage discover --config configs/preprocessing.local.json
python preprocessing/run.py --stage features --config configs/preprocessing.local.json
python preprocessing/run.py --stage merge --config configs/preprocessing.local.json
python preprocessing/run.py --stage expression --config configs/preprocessing.local.json
python preprocessing/run.py --stage rowmap --config configs/preprocessing.local.json
python preprocessing/run.py --stage neighbors --config configs/preprocessing.local.json
python preprocessing/run.py --stage canonical --config configs/preprocessing.local.json
python preprocessing/run.py --stage hvg384 --config configs/preprocessing.local.json
```

上述命令默认预览。核对每一步输入输出后，在对应命令末尾加`--execute`。canonical修正需要词典、划分、rowmap及Program源表；HVG重建需要校正manifest、有效rowmap和冻结split。将校正后的manifest/Program配置为后续训练输入。

| 阶段 | 主要产物/作用 |
| --- | --- |
| discover | 发现已有数据源并生成inventory；需人工审核遗漏与映射 |
| features | spot对应的1536维UNI2-h特征分片 |
| merge | 合并特征与对应manifest |
| expression | 表达raw-log1p chunks、实测mask和基因映射 |
| rowmap | 图像特征行与表达行精确关联及审计 |
| neighbors | 原生candidate32/homo12/hetero8邻域 |
| canonical | 身份修正、canonical Program M16映射与审计 |
| hvg384 | 实际有效行上重建section HVG384 |

`step2_add_organs.py`及旧`rebuild_section_hvg384_17organs.py`为历史迁移存档，保留`ARCHIVAL_ONLY`保护。当前使用入口为上表中的配置化阶段命令。

## 7. PyTorch四卡训练

### 7.1 训练前必备

需要完整外置feature store、校正gene chunks、rowmap、划分文件、typed-neighbor overlay、Program映射及active-row HVG384。按照配置提供对应的数据清单和路径。

默认示例从step 0、seed42随机初始化开始，不读取任何checkpoint。断点恢复另用`configs/train.resume.example.json`，指定本训练入口生成的完整状态文件和其中保存的实际step；GPU数量保持一致。

### 7.2 推荐的配置化命令

```bash
cp configs/train.example.json configs/train.local.json
python training/run.py --config configs/train.local.json
```

核对所有路径、物理GPU及全新输出目录后再执行：

```bash
python training/run.py --config configs/train.local.json --execute
```

包装层使用PyTorch分布式启动器，默认四进程，GPU来自配置中的`gpus`。训练参数为每卡batch64、BF16、AdamW、学习率/weight decay均1e-4、gradient clip1.0、gene chunk2048、seed42；prompt/gene bias调度从step5000开始。

### 7.3 标准`torchrun`命令模板

下面是同一类原生入口的命令模板；原生入口不提供包装层的dry-run保护，**直接运行就会开始训练**。先完成7.2的检查。所有输入路径必须真实存在，`OUT`必须是新目录。

```bash
export DATA_ROOT=/external/spagenx_data
export OUT=/external/spagenx_runs/new_training_001

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun \
  --standalone --nnodes=1 --nproc_per_node=4 \
  training/train_spagenx.py \
  --rowmap_root "$DATA_ROOT/rowmap" \
  --v3_split_root "$DATA_ROOT/split" \
  --feature_store "$DATA_ROOT/features.pt" \
  --gene_chunk_manifest "$DATA_ROOT/gene_label_chunks_manifest.tsv" \
  --neighbor_overlay "$DATA_ROOT/neighbors.pt" \
  --program_overlay_path "$DATA_ROOT/gene_to_program_m16_overlay.pt" \
  --section_hvg_path "$DATA_ROOT/section_hvg384_active_rows.pt" \
  --expected_n_organs 17 --expected_n_techs 4 \
  --batch_size 64 --amp_dtype bf16 --target_k 1024 --target_chunk_size 2048 \
  --prompt_ratio 0.08 --prompt_k_cap 64 --prompt_start_step 5000 \
  --lr 0.0001 --weight_decay 0.0001 --grad_clip 1.0 \
  --eval_micro_batch_rows_per_rank 4 --eval_sampler_mode all_gene_20 \
  --seed 42 --base_seed 42 --mode full \
  --max_steps 500000 --eval_interval 1000 --checkpoint_interval 1000 \
  --save_checkpoint --save_full_state --write_best_model \
  --early_stopping --early_stop_start_step 40000 \
  --early_stop_patience_evals 20 --early_stop_min_delta 0.0005 \
  --scratch17 --model_init_seed 42 \
  --resume_global_step 0 --output_root "$OUT"
```

`max_steps`为本次运行新增的优化器步数；最终step为`resume_global_step + max_steps`。从零训练时`resume_global_step=0`。续训时修改恢复配置中的checkpoint路径、实际已保存step和剩余训练步数，保持GPU数量一致。验证指标和优化设置由原生训练器定义。

### 7.4 训练产物

按启用选项保存`last_checkpoint.pt`、`last_full_state_checkpoint.pt`、`best_per_gene_pcc_checkpoint.pt`、`eval_metrics.tsv`等。完整状态checkpoint用于断点恢复，模型权重用于加载模型。

## 8. 自检与验证

```bash
python tests/test_release.py
python inference/spagenx_cli.py --help
python preprocessing/run.py --help
python training/run.py --help
```

当前六项合成测试涵盖邻域padding、基因身份、实测零/身份合并、行索引约定、MPP/空白图/输出冲突和patch上限。测试不依赖真实图像或权重，但需要已安装依赖。

`docs/validation.json`记录使用私有测试资产完成的模型检查：原生训练函数AST一致、调度边界等价，以及SpaGenX权重固定输入比较的最大误差为0。

## 9. 常见问题

| 报错/现象 | 处理 |
| --- | --- |
| 权重SHA256不匹配 | 核对SpaGenX或UNI2-h文件来源和SHA256；不要修改代码跳过检查 |
| `CHECKPOINT_STEP` / `PROGRAM_BUFFER_IDENTITY` | 权重或Program映射不匹配；使用同一冻结版本 |
| `MPP_REQUIRED` | 提供可信真实MPP，不随意填0.5 |
| `NO_VALID_TISSUE_PATCHES` | 检查图像、区域、尺度及白色过滤；不要自动放宽阈值 |
| `PATCH_LIMIT_EXCEEDED_NO_SAMPLING` | 有效patch超过16000；选择较小的输入区域后新建运行 |
| `FEATURE_NEIGHBOR_COORDINATE_IDENTITY` | 特征与邻域不属于同一patch集合/顺序，重新核对资产 |
| `UNKNOWN_OR_AMBIGUOUS_GENE` | 查看冻结词典，用明确ID消歧，不做模糊映射 |
| `OUTPUT_ALREADY_EXISTS`或`FileExistsError` | 选择新的输出目录 |
| `ARCHIVAL_ONLY` | 这是受保护的历史脚本，不是当前自动化入口 |
| CUDA OOM | 停止当前运行，检查显存占用；调整计算分块后新建运行并记录配置 |
| `libopenslide`/动态库加载失败 | 同时核对Python绑定及底层库，参见官方平台安装说明 |
| `invalid device ordinal` | 检查物理GPU与`CUDA_VISIBLE_DEVICES`后的本地编号 |
| 断点恢复缺少checkpoint | 仅续训时提供本训练入口生成的完整状态文件；从零训练无需checkpoint |

OOM后可新建运行，调整`PATCH_BATCH`、`GENE_CHUNK`并记录配置。该封装使用torch反序列化，仅加载可信来源的`.pt`文件。

## 10. 发布、引用与支持

代码许可证待维护者确认；UNI2-h及其他依赖遵循各自条款。正式论文发表后补充引用信息。提交代码前排除私有审计、权重、真实日志、患者文件及凭据。

问题报告建议包含命令（去掉患者及私有路径）、环境版本、错误摘要、模型/词典哈希、图像尺寸及MPP来源，不上传受限图像、表达或凭据。

相关工具：[UNI](https://github.com/mahmoodlab/UNI)、[scGPT](https://github.com/bowang-lab/scGPT)。
