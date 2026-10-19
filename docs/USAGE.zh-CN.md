# SpaGenX完整使用说明（PyTorch命令版）

本文对应当前代码包，所有命令从仓库根目录运行。用你自己的真实路径替换`/external/...`。仓库为私有，以下克隆命令需要已授权的GitHub账号；SpaGenX权重下载地址尚未配置，不会自动下载权重。

```bash
git clone https://github.com/JMJ-rbgss/SpaGenX.git
cd SpaGenX
```

模型对外统一命名为**SpaGenX**。代码中的历史入口文件名、配置查找键和序列化字段仅作为兼容性标识保留，并不表示另一个模型；不要直接改动这些标识而破坏既有资产读取。

## 1. 选择使用路线

| 目的 | 所需输入 | 使用入口 |
| --- | --- | --- |
| 对新H&E/WSI预测基因表达 | 图像、MPP、器官、基因列表、冻结资产 | `scripts/infer_wsi.sh` |
| 对已有可信特征预测 | 特征、匹配邻域、器官和基因列表 | `inference/spagenx_cli.py infer` |
| 构建研究训练数据 | 图像、实测表达、spot坐标、冻结划分和元数据 | `preprocessing/run.py` |
| 按原生合同继续训练 | 完整训练资产、step1000恢复状态、GPU | `training/run.py`或`torchrun` |

代码包不含权重、词典、Program映射、真实示例、表达矩阵、训练数据或优化器状态。**只有代码时可以检查接口和部分合成测试，不能直接完成真实预测或完整训练。**

## 2. 环境安装

推荐使用与验收一致的Linux、Python3.9、PyTorch2.6.0环境。以下安装命令需联网，但本说明书生成过程没有替你安装任何环境。

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

构建选择与驱动兼容性以[PyTorch官方历史版本说明](https://pytorch.org/get-started/previous-versions/)为准。这里不承诺所有操作系统及驱动组合均通过测试。

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

不要把一个已有环境中的`site-packages`直接复制到另一台机器。验收环境版本见`docs/validation.json`；它不是完整环境锁文件。

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

向项目维护者获取已批准的SpaGenX冻结资产；发布下载地址尚未配置。UNI2-h应从[官方模型页面](https://huggingface.co/MahmoodLab/UNI2-h)按其访问条件独立获取。代码包不自动下载或分发UNI2-h权重；公开/商业用途须另行核对适用条款。

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

其余冻结资产哈希见`docs/external_assets.json`。文件名相同不代表内容相同。

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

必须使用该冻结codebook的17类之一，不要自行重编号。CLI可忽略名称大小写，但不会推断未知器官。本代码包未集成网站的器官辅助分类器。

## 4. 快速开始：自动WSI推理

### 4.1 准备基因清单

创建一个UTF-8文本文件，每行一个符号或一个`global_gene_id`，不带表头。例如：

```text
VIM
COL1A1
CD74
```

示例不表示这些基因在所有器官均已获得性能保证。所有查询必须存在于冻结词典；符号歧义时应使用明确ID，不允许重复基因。同一基因不能以符号和ID重复输入。

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

`CUDA_VISIBLE_DEVICES=4`选择物理GPU4；进程内部该卡显示为`cuda:0`。不要再把`DEVICE`写成`cuda:4`。CPU可设置`DEVICE=cpu`，但大图编码不承诺实用速度；未完成独立CPU编码性能验证。

若图像没有可信MPP元数据，显式设置：

```bash
export MPP=0.5
```

这里的MPP是**输入图像的真实像素尺度**，不能为了绕过报错随意填写。图像元数据已有可靠MPP时无需设置；需要清除旧设置可运行`unset MPP`。

### 4.3 预览与执行

```bash
bash scripts/infer_wsi.sh --dry-run
bash scripts/infer_wsi.sh --execute
```

脚本先检查文件、配置及全新`RUN_DIR`；执行时依次进行图像切块、UNI2-h编码、邻域构建和SpaGenX推理。第一阶段失败不会继续第二阶段。日志位于`preprocess.log`和`inference.log`。这不是网站服务，不会启动端口或上传图像。

固定应用协议：0.5µm/px、224×224、不重叠网格；白色判据为RGB均值≥205，白色比例>0.8的patch剔除；接受patch超过16000则停止，不静默采样。该白色判据不是肿瘤区域识别。缺少MPP、空白图、损坏图像或输出已存在时停止。超过6400万像素的普通栅格图应转换为合适的金字塔WSI，而不是整张读入。

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

对缺少MPP的图片，在第一条命令补`--mpp <真实值>`。已有可信特征及匹配邻域时，只运行第二条命令；不要为了套用此说明而重新建图。

推理使用冻结的SpaGenX权重配置：FP32、gene bias开启、tech_zero、空表达prompt并显式disable_prompt，内部调度由实现锁定。`gene-chunk`是计算分块，不是只预测前256基因。脚本不会根据效果改基因，不会自动训练模型。

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

这是模型原生raw-counts-log1p尺度预测，不是实测RNA、TPM或已校准的绝对分子计数。不要未经验证直接反变换并当作真实计数。图像patch也不等同于分割后的单细胞。查询可用、训练覆盖和逐基因可靠性是不同概念。

`inference.json`记录SpaGenX权重身份、no_prompt/gene bias状态、输入哈希、器官、精度及分块参数。保存它与图像、MPP、基因顺序的来源记录，以便追溯。

## 6. 研究数据预处理

此路线面向拥有配对图像、实测表达和spot坐标的研究者。与第4节新WSI自动网格不是同一采样协议。

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

上述均为预览。核对每一步输入输出后，在对应命令末尾加`--execute`。这是一组阶段接口，不是适用于任意数据集的无条件顺序脚本：canonical修正需要已有词典、划分、rowmap及Program源表；HVG重建需要校正manifest、有效rowmap和冻结split。校正后的manifest/Program应作为后续训练输入。

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

`step2_add_organs.py`、旧`rebuild_section_hvg384_17organs.py`及实验专用pseudobulk脚本已设置`ARCHIVAL_ONLY`保护；不要移除保护去修改原数据。原17器官数据构建包含历史迁移，当前代码包不冒称已提供面向任意新数据的一键全重建功能。

## 7. PyTorch四卡训练

### 7.1 训练前必备

需要完整外置feature store、校正gene chunks、rowmap、划分文件、typed-neighbor overlay、Program映射及active-row HVG384。按照配置提供对应的数据清单和路径。

当前示例是**恢复/继续训练合同**，还需要step1000完整状态（模型、优化器及恢复信息）。缺少该状态时不能用SpaGenX推理权重冒充它。代码目录存在，不等于完整训练沿革可以从零复现。

### 7.2 推荐的配置化命令

```bash
cp configs/train.example.json configs/train.local.json
python training/run.py --config configs/train.local.json
```

核对所有路径、物理GPU及全新输出目录后再执行：

```bash
python training/run.py --config configs/train.local.json --execute
```

包装层使用PyTorch分布式启动器，默认四进程，GPU来自配置中的`gpus`。训练参数与原生模型一致：每卡batch64、BF16、AdamW、学习率/weight decay均1e-4、gradient clip1.0、gene chunk2048、seed42；保留训练step5000的prompt/gene bias调度。该合同不包含本次without homo/hetero消融。

### 7.3 标准`torchrun`命令模板

下面是同一类原生入口的命令模板；原生入口不提供包装层的dry-run保护，**直接运行就会开始训练**。先完成7.2的检查。恢复状态及所有路径必须真实存在，`OUT`必须是新目录。

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
  --max_steps 499000 --eval_interval 1000 --checkpoint_interval 1000 \
  --save_checkpoint --save_full_state --write_best_model \
  --early_stopping --early_stop_start_step 40000 \
  --early_stop_patience_evals 20 --early_stop_min_delta 0.0005 \
  --direct_17organ_model_only_resume --resume_full_state \
  --resume_checkpoint "$DATA_ROOT/recovery_step1000_full_state.pt" \
  --resume_global_step 1000 --output_root "$OUT"
```

`max_steps`与恢复global step必须按真实训练状态解释，不是把新模型标记为已发布SpaGenX权重的开关。当前代码未添加新的学习率调度，也未替换原训练验证指标。若要改训练预算、验证PCC或模型结构，应另立实验合同。

### 7.4 训练产物

按启用选项可保存`last_checkpoint.pt`、`last_full_state_checkpoint.pt`、`best_per_gene_pcc_checkpoint.pt`、`eval_metrics.tsv`等。完整状态文件与仅模型权重用途不同。小型Smoke通过不代表训练已收敛，也不代表新训练复现了既有SpaGenX权重。

## 8. 自检与复现边界

```bash
python tests/test_release.py
python inference/spagenx_cli.py --help
python preprocessing/run.py --help
python training/run.py --help
```

当前六项合成测试涵盖邻域padding、基因身份、实测零/身份合并、行索引约定、MPP/空白图/输出冲突和patch上限。测试不依赖真实图像或权重，但需要已安装依赖。

`docs/validation.json`记录打包时远端私有资产验证：原生训练函数AST一致、调度边界等价性及SpaGenX权重小样本误差0。真实输入、参考预测和权重未随包分发，第三方仅凭当前代码包不能重做该私有测试。

## 9. 常见问题

| 报错/现象 | 处理 |
| --- | --- |
| 权重SHA256不匹配 | 核对SpaGenX或UNI2-h文件来源和SHA256；不要修改代码跳过检查 |
| `CHECKPOINT_STEP` / `PROGRAM_BUFFER_IDENTITY` | 权重或Program映射不匹配；使用同一冻结版本 |
| `MPP_REQUIRED` | 提供可信真实MPP，不随意填0.5 |
| `NO_VALID_TISSUE_PATCHES` | 检查图像、区域、尺度及白色过滤；不要自动放宽阈值 |
| `PATCH_LIMIT_EXCEEDED_NO_SAMPLING` | 该封装不处理>16000有效patch；不能偷偷抽样后声称同协议 |
| `FEATURE_NEIGHBOR_COORDINATE_IDENTITY` | 特征与邻域不属于同一patch集合/顺序，重新核对资产 |
| `UNKNOWN_OR_AMBIGUOUS_GENE` | 查看冻结词典，用明确ID消歧，不做模糊映射 |
| `OUTPUT_ALREADY_EXISTS`或`FileExistsError` | 选择新输出目录；不要删除旧结果以掩盖冲突 |
| `ARCHIVAL_ONLY` | 这是受保护的历史脚本，不是当前自动化入口 |
| CUDA OOM | 本次失败即停止；检查占卡情况，另起运行并显式记录调整，不静默改协议 |
| `libopenslide`/动态库加载失败 | 同时核对Python绑定及底层库，参见官方平台安装说明 |
| `invalid device ordinal` | 检查物理GPU与`CUDA_VISIBLE_DEVICES`后的本地编号 |
| 缺少恢复checkpoint | 可读训练代码，但不能执行原恢复合同或宣称复现SpaGenX权重 |

OOM时可以在明确记录的新运行中用`PATCH_BATCH`、`GENE_CHUNK`调整应用推理分块；这不是本包已验证所有组合的承诺。不要使用不可信用户上传的`.pt`文件，因为该封装使用受信任资产的torch反序列化路径。

## 10. 发布、引用与支持

只上传`repository/`，不要上传兄弟目录`private_audit/`、`control/`、权重、真实日志或患者文件。当前未指定自有代码许可证；UNI2-h及其他依赖遵循各自条款。论文、DOI、作者及正式引用信息须由项目所有者确认，本说明不虚构BibTeX。

问题报告建议包含命令（去掉患者及私有路径）、环境版本、错误摘要、模型/词典哈希、图像尺寸及MPP来源，不上传受限图像、表达或凭据。

文档结构参考科研项目[UNI](https://github.com/mahmoodlab/UNI)和[scGPT](https://github.com/bowang-lab/scGPT)常用的安装、资产、快速开始与复现分节；SpaGenX算法、参数及使用边界均以本代码包为准，不从这些项目复制性能或许可声明。
