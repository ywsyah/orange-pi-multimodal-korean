# orange-pi-multimodal-korean
基于昇腾AI的多模态韩语应用 —— 使用“万卷·丝路”数据集微调的QwenVL2模型
## 🔍 项目简介

本项目基于 OpenDataLab 发布的“万卷·丝路”开源语料库中韩语和俄语多模态图文数据，利用llamafactory进行大模型框架微调在昇腾 OrangePi AIpro 开发板上构建了图文联合推理和专业术语问答两个应用。该项目使用 MindSpore + MindNLP + QwenVL2 模型微调实现，满足低资源语种在边缘端进行AI推理的场景需求。

- 模型架构：QwenVL2-2B / QwenVL2.5-2B
- 数据集来源：[万卷·丝路](https://opendatalab.com/OpenDataLab/WanJuanSiLu2O)
- 推理平台：OrangePi AIpro (Ascend AI 20TOPS)
- 框架版本：
  - MindSpore 2.5.0
  - MindNLP 0.4.1
  - CANN Toolkit: 8.1RC1beta1
  - Python 3.9
  - llamafactory 0.9.4.dev0

模型训练流程，在服务器端进行训练：
如不想自己训练可以直接下载权重：path

1、下载韩语图文数据集(该数据集中其他语种的也适用)：https://opendatalab.com/OpenDataLab/WanJuanSiLu2O/blob/main/raw/image/ko/ko_image_caption.jsonl
![alt text](image.png)
2、运行脚本get_all_img.py --input your_json_path --output new_json_path --outdir your_image_save_dir   将图片文件下载到本地（可选择下载个数），并且建立新的文件索引，防止训练过程中图片加载失败
3、运行脚本delete_break_img.py   将数据集中无法加载，格式出错的图片过滤掉
4、运行脚本convert_to_sharegpt.py   转换成llamafactory支持的sharegpt训练数据格式:
[
  {
   
    "messages": [
      {
   
        "content": "<image>他们是谁？",
        "role": "user"
      },
      {
   
        "content": "他们是拜仁慕尼黑的凯恩和格雷茨卡。",
        "role": "assistant"
      },
      {
   
        "content": "他们在做什么？",
        "role": "user"
      },
      {
   
        "content": "他们在足球场上庆祝。",
        "role": "assistant"
      }
    ],
    "images": [
      "demo_data/1.jpg"
    ]
  }
]

5、运行脚本split.py 划分数据集与测试集, 可在其中设置数据集与测试集的数据个数，不然训练过慢

6、机器环境：
硬件：
显卡：A100
显存：80G

软件：
系统：Ubuntu 22.04 LTS
python：3.10
pytorch：2.7.1 + cuda12.2

6、安装llamafactory
按照GitHub上介绍的过程安装即可，为了加快速度，增加了国内的pip源。

git clone https://github.com/hiyouga/LLaMA-Factory.git
conda create -n llama_factory python=3.10
conda activate llama_factory
cd LLaMA-Factory
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -e .[metrics]


将数据集json拷贝到到LLaMaFactory的data目录下，图片文件拷贝到LLaMaFactory下
![alt text](4fb289fa5afded7e59ca6a3210859b29.png)
![alt text](image-1.png)

修改 LLaMaFactory data目录下的dataset_info.json，增加自定义数据集：
![alt text](image-2.png)



本次微调使用阿里最新发布的多模态大模型：Qwen2-VL-2B-Instruct或者Qwen2.5-VL-3B-Instruct 作为底座模型。
但由于后续在mindspore以及mindnlp的版本适配问题，最后部署Qwen2.5-VL-3B-Instruct时出现问题，故训练Qwen2-VL-2B-Instruct
模型说明地址：https://modelscope.cn/models/Qwen/Qwen2-VL-2B-Instruct

使用如下命令下载模型，QwenVL2.5-3B同理：
git lfs install
# 下载模型
git clone https://www.modelscope.cn/Qwen/Qwen2-VL-2B-Instruct.git



访问LLaMaFactory的web页面:
llamafactory-cli webui

配置微调的训练参数：

Model name: Qwen2-VL-2B-Instruct
Model path: models/Qwen2-VL-2B-Instruct
Finetuning method: lora
Stage : Supervised Fine-Tuning
Dataset: ko_train
Output dir: saves/Qwen2-VL/lora/Qwen2-VL-sft-ko_train

配置参数中将save_steps设置大一点，否则训练过程会生成非常多的训练日志，导致硬盘空间不足而训练终止,具体训练参数参考train文件夹下对于各模型以及数据集的配置，如果有多张卡可以使用deepspeed进行多卡训练。

经过10个小时的训练，模型训练完成，损失函数如下：
![alt text](training_loss.png)


点击Preview Command预览命令行无误后，点击Run按钮开始训练。
训练的过程中，可以通过 watch -n 1 nvidia-smi 实时查看GPU显存的消耗情况。

8.3 合并导出模型
接下来，我们将 Lora与 原始模型 合并导出：

切换到 Expert 标签下
Model path: 选择Qwen2-VL的基座模型，即：models/Qwen2-VL-2B-Instruct
Checkpoint path: 选择lora微调的输出路径，即 saves/Qwen2-VL/lora/Qwen2-VL-sft-demo1
Export path：设置一个新的路径，例如：Qwen2-VL-sft-final
点击 开始导出 按钮


找到导出的路径：
将该模型权重下载并且导入到orangepi aipro20T24G的板子上
具体运行脚本参见：ko.ipynb



