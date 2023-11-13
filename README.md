# 如何运行

## 环境
运行前先配置环境
conda create -n llm
conda activate llm
pip install openai request

## 配置config文件

1. 配置config.yml

    配置 args 到 config.yml



2. 配置key.yml文件

    如果使用GPT接口，需要在根目录中配置key.yml

- 如果 model = "GPT..." 并且 GPT["api"] = "openai"（不填GPT["api"]默认为openai：

    在根目录创建 openai_key.yml 

- 如果 model = "GPT..." 并且 GPT["api"] = "hkbu"：

    在根目录创建 hkbu_key.yml

在创建的xxxx_key.yml 中包含下面字段

``` txt
key = xxxxxx
```

## 运行程序
在根目录中
执行`python noise_test.py`

结果会保存在result文件夹中


