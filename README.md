# 如何运行

## 环境
运行前先配置环境
``` bash
conda create -n llm
conda activate llm
pip install openai=0.28 requests pandas nltk pyyaml scikit-learn
```

## 配置config文件

1. 配置config.yml

    配置 args 到 config.yml

Category | Parameter | Description |Examples|
| ------ | ------ | ------ | ------ |
|Model|model|llm model name|"gpt-3.5-turbo", "gpt-3.5-turbo-1106", "llama2"|
|Dataset|dataset|the dataset used for the experiment.|"base_math", "SCAN", "family_relation"|
||start_num|the starting number of the experiment.| 0 |
||test_num|the number of test instances.|200|
||batch_size|the size of the data processed per batch.|1, 5|
|ICL|if_in_context| symbol of whether use in-context demos |True, False|
||n_shots| w/o noise shots num | 1, 2, 3|
|Noise|if_noise|symbol of whether use noisy demos|True, False (be False if if_in_context is False)|
||n_noisy_shots| noisy shots num | 1, 2, 3|
||noise_type| type of noise | "irrelavant", "minor-error" |
||noise_ratio| ratio of each thought insert a sentence irrelavant noise or become a minor-error thought|0.2, 0.5, 0.8|
||noise_distribution| fixed noise num in a example shot or just same possibilty to insert noise in each thought| "fixed", "random"|


2. 配置key.yml文件

    如果使用GPT接口，需要在根目录中配置key.yml

- 如果 model = "gpt..." 并且 gpt["api"] = "openai"（不填gpt["api"]默认为openai：

    构建openai_key.yml：
    echo "key = xxxxxx" > openai_key.yml 

- 如果 model = "gpt..." 并且 gpt["api"] = "hkbu"：
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


