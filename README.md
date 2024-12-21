# IconMatting-main

## 代码功能
该代码是为了复现 CVPR 2024 论文《In-Context Matting》，可以从这个[链接](https://arxiv.org/pdf/2403.15789.pdf)找到原论文。代码的详细功能见论文介绍。

本代码是在[该代码](https://github.com/tiny-smart/in-context-matting)的基础上进行了完善补充，并在扩充的ICM-57数据集上进行了实验。

## 代码运行
### 所需文件下载
1. **预训练模型下载：** 
可以从[该链接](https://pan.baidu.com/s/1HPbRRE5ZtPRpOSocm9qOmA?pwd=BA1c)下载代码所需要的预训练模型。
2. **数据集下载：**
- ICM-57：可以从[该链接](https://pan.baidu.com/s/1bcy5aqTGwHw_03m8TBkA6Q?pwd=BA1c)下载代码所需要的数据集。
- 扩充了17类的ICM-57：已与报告一起提交
3. **Stable-Diffusion-2-1下载：**
可以从[该链接](https://huggingface.co/stabilityai/stable-diffusion-2-1/tree/main)下载Stable-Diffusion预训练模型。

### 格式设置
1. **config文件设置：**
将```config/eval.yaml```中的```sd_id```改为```Stable-Diffusion-2-1```的你的下载路径
2. **数据集格式设置：**
请保证数据集放于该项目的文件夹下，且数据集满足以下格式：
```
    datasets
    |____ICM57
    |    |____alpha
    |    |____image
    |    |____trimap
    |
    |____ICM57.json
```

### 代码运行
1. **配置环境：**
执行以下命令配置环境：
```
    conda env create -f environment.yml
```
2. **代码运行：**
执行以下命令运行代码：
```
    python eval.py --checkpoint PATH_TO_PRETRAINED_MODEL --save_path results/ --config config/eval.yaml
```
3. **结果查看：**
- 代码所运行的结果保存在```results/```下
- 代码运行的日志文件保存在```logs/```下，可以运行```tensorboard```命令查看可视化日志
