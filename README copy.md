# 🧠Orange-Pi Multimodal Korean  
## 基于昇腾 AI 的多模态韩语应用

利用 **“万卷·丝路”** 开源图文语料库，对 **Qwen2-VL-2B以及Qwen2.5-VL-3B** 模型在服务器端进行 LoRA 微调，并基于mindspore、mindnlp和gradio部署于 **OrangePi AIpro**（20 TOPS Ascend SoC）。项目提供：

1. 🖼️+📝**图文联合问答** 微调与部署流程（基于LLamafactory微调）
2. 📚🔍**专业术语问答** 微调与部署流程（基于原生transformer+Peft库微调）

> 适用于端侧低资源小语种的 AI 场景，并且学习使用llamafactory框架微调和原生transformer+Peft进行微调。

---

## 🚀 主要特性

| 模块 | 说明 |
| ---- | ---- |
| **底座模型** | `Qwen2-VL-2B-Instruct` / `Qwen2.5-VL-3B-Instruct` |
| **数据集** | [万卷·丝路](https://opendatalab.com/OpenDataLab/WanJuanSiLu2O)（韩语） |
| **训练框架** | LLaMA-Factory 0.9.4.dev0 /  | 
| **部署平台** | OrangePi AIpro（Ascend 20 TOPS，24 GB RAM） |
| **微调方法** | LoRA + SFT |

---

## ⚙️ 环境准备

### 服务器端（训练）

| 硬件 | 规格 |
| ---- | ---- |
| GPU  | NVIDIA A100 80 GB × 1 |
| CPU  | 32 cores |
| RAM  | 224 GB |

| 软件 | 版本 |
| ---- | ---- |
| OS   | Ubuntu 22.04 LTS |
| Python | 3.10 |
| PyTorch | 2.7.2 + CUDA 12.2 |
| Deepspeed | 可选（多卡） |

#### llamafactory安装流程（参考官方github，这里加了国内源）  
    git clone https://github.com/hiyouga/LLaMA-Factory.git
    conda create -n llama_factory python=3.10
    conda activate llama_factory
    cd LLaMA-Factory
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -e .[metrics]

---

### Edge 端（OrangePi AIpro）

    MindSpore       2.5.0
    MindNLP         0.4.1
    CANN Toolkit    8.1.RC1.beta1
    Python          3.9
    gradio          4.4.0

---

## 📚 说明
以下训练流程是针对🖼️+📝**图文联合问答** 微调与部署流程（基于LLamafactory微调）应用的
对于📚🔍**专业术语问答** 微调与部署流程（基于原生transformer+Peft库微调）参考文件**ko_text.ipynb**。

## 📥 数据集准备与处理

#### 1. 下载韩语图文标注  

      https://opendatalab.com/OpenDataLab/WanJuanSiLu2O/blob/main/raw/image/ko/ko_image_caption.jsonl

#### 2. 图片拉取与索引重建  

       python scripts/get_all_img.py \
           --input ko_image_caption.jsonl \
           --output ko_caption_clean.json \
           --outdir data/images/ko \
           --max-lines 20000        # 可选：限制下载数量(0代表全部下载)

#### 3. 清洗坏图  

       python scripts/delete_break_img.py \
           --input_json_path ko_caption_clean.json \
           --output_json_path data/ko_sharegpt.json

#### 4. 转换为 ShareGPT 格式  

       python scripts/convert_to_sharegpt.py \
           --input_json_path ko_caption_clean.json \
           --output_json_path data/ko_sharegpt.json

   样例  

       {
         "messages": [
           {"role": "user", "content": "<image>그들은 누구입니까?"},
           {"role": "assistant", "content": "그들은 바이에른 뮌헨의 케인과 고레츠카입니다."},
           {"role": "user", "content": "무엇을 하고 있습니까?"},
           {"role": "assistant", "content": "축구장에서 세리머니 중입니다."}
         ],
         "images": ["demo_data/1.jpg"]
       }

#### 5. 训练 / 测试拆分  

      python split_data.py \
          --src ko_sharegpt.json.json \
          --train_out ko_train.json \
          --eval_out ko_eval.json \
          --n_train 30000 \
          --n_eval 6000

---

## 🔧 微调流程

#### 1. 下载底座模型  

       git lfs install
       git clone https://www.modelscope.cn/Qwen/Qwen2-VL-2B-Instruct.git models/Qwen2-VL-2B-Instruct

#### 2. 添加数据集描述（编辑 `LLaMA-Factory/data/dataset_info.json`）  

       "ko_train": {
         "path": "data/ko_train.json",
         "type": "sharegpt_multi_modal"
       },
       "ko_val": {
         "path": "data/ko_val.json",
         "type": "sharegpt_multi_modal"
       }

#### 3. 启动 WebUI  

       llamafactory-cli webui

#### 4. 关键参数示例(具体参数参考train_param文件夹下yaml文件) 

   | 选项 | 值 |
   | ---- | -- |
   | Model name  | Qwen2-VL-2B-Instruct |
   | Model path  | models/Qwen2-VL-2B-Instruct |
   | Finetune    | LoRA |
   | Stage       | Supervised Fine-Tuning |
   | Dataset     | ko_train |
   | Max epochs  | 3 |
   | Batch size  | 16 |
   | Save steps  | 200 |
   | lora_rank   | 64 |
   | lora_alpha  | 128（一般是rank的两倍） |
   | lora_dropout | 0.05（防止过拟合） |
   | Output dir  | saves/Qwen2-VL/lora/Qwen2-VL-sft-ko |

### 5. 监控显存  

       watch -n 1 nvidia-smi

### 6. 训练耗时  
   单张 A100 约 **10 h**；最终 loss 曲线见 `docs/training_loss.png`。

---

## 🗜️ 合并 LoRA & 导出

在 WebUI **Expert** 标签执行  

    Model path      = models/Qwen2-VL-2B-Instruct
    Checkpoint path = saves/Qwen2-VL/lora/Qwen2-VL-sft-ko
    Export path     = models/Qwen2-VL-sft-final

点击“开始导出”，得到合并权重。

---

## 📦 边缘端部署

1. 将 `models/Qwen2-VL-sft-final` 拷贝至 OrangePi AIpro  
2. 参考 `notebooks/ko.ipynb` 进行推理测试：  
   • 多模态图片问答  
   • 专业术语翻译 / QA

---

## 📝 引用

    @misc{orangepi2024multimodal,
      title   = {Orange Pi Multimodal Korean},
      author  = {Your Name},
      year    = {2024},
      url     = {https://github.com/yourrepo/orange-pi-multimodal-korean}
    }

---

