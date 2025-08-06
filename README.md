# orange-pi-multimodal-korean
基于昇腾AI的多模态韩语应用 —— 使用“万卷·丝路”数据集微调的QwenVL2模型
## 🔍 项目简介

本项目基于 OpenDataLab 发布的“万卷·丝路”开源语料库中阿拉伯语多模态图文数据，在昇腾 OrangePi AIpro 开发板上构建了图文联合推理应用。该项目使用 MindSpore + MindNLP + QwenVL2 模型微调实现，满足低资源语种在边缘端进行 AI 推理的场景需求。

- 模型架构：QwenVL2 / QwenVL2.5B
- 数据集来源：[万卷·丝路](https://opendatalab.com/OpenDataLab/WanJuanSiLu2O)
- 推理平台：OrangePi AIpro (Ascend AI 20TOPS)
- 框架版本：
  - MindSpore 2.5.0 / 2.6
  - MindNLP 0.4.1
  - CANN Toolkit: 8.0.0.beta1 / 8.1RC1beta1
  - Python 3.9

git clone
