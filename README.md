# 如何运行

## 环境
运行前先配置环境
``` bash
conda create -n llm
conda activate llm
pip install openai requests pandas nltk
```

## 配置config文件

1. 配置config.yml

    配置 args 到 config.yml

| args | description |examples|
| ------ | ------ | ------ |
|model|llm model name|"gpt-3.5-turbo", "gpt-3.5-turbo-1106", "llama2"|
|dataset|the dataset used for the experiment.|"base_math", "family_relation"|
|start_num|the starting number of the experiment.| 0 |
|test_num|the number of test instances.|200|
|run_times|the number of times the experiment is run.|1|
|batch_size|the size of the data processed per batch.|1, 5|


2. 配置key.yml文件

    如果使用GPT接口，需要在根目录中配置key.yml

- 如果 model = "GPT..." 并且 GPT["api"] = "openai"（不填GPT["api"]默认为openai：

    构建openai_key.yml：
    echo "key = xxxxxx" > openai_key.yml 

- 如果 model = "GPT..." 并且 GPT["api"] = "hkbu"：
    构建hkbu_key.yml：
    echo "key = xxxxxx" > hkbu_key.yml

在创建的xxxx_key.yml 中包含下面字段

``` txt
key = xxxxxx
```

## 运行程序
在根目录中
执行`python noise_test.py`

结果会保存在result文件夹中


