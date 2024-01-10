import random
import json
import pandas as pd
import zipfile
import os
import re
import ast
import copy
from collections import deque
import math

class bbh():
    def __init__(self, n_shots=0, n_noisy_shots=0, noise_type="irrelevant", noise_ratio = 0.5, noise_distribution = "fixed", prefix_context =True, config: dict = None, reasoning_type = "length") -> None:
        self.n_shots = n_shots
        self.n_noisy_shots = n_noisy_shots
        self.total_shots = self.n_shots + self.n_noisy_shots 
        if self.total_shots > 0:
            self.if_in_context = True
        else:
            self.if_in_context = False
        self.prefix_context = prefix_context
        if self.n_noisy_shots > 0:
            if noise_type != "irrelevant" and noise_type != "inaccurate":
                raise ValueError(f"noise type: {noise_type} not supported")
            self.noise_type = noise_type
            self.noise_ratio = noise_ratio
            self.noise_distribution = noise_distribution
        else:
            self.noise_type = None
            self.noise_ratio = 0
            self.noise_distribution = None
            
        if config is not None:
            self.reasoning_type = config["reasoning_type"]
        else:
            self.reasoning_type = reasoning_type
        
        self.file_path = os.path.join("data", "BBH")
        return
    
    def load_data(self):
        dataset_file = os.path.join(self.file_path, "data", "{}.json".format(self.reasoning_type))
        with open(dataset_file, "r") as f:
            raw_dataset = json.load(f)
        examples = raw_dataset["examples"]
        dataset = []
        for example in examples:
            input = example["input"]
            target = example["target"]
            dataset.append([input, target])
        prompt_file = os.path.join(self.file_path, "lib_prompt", "{}.txt".format(self.reasoning_type))
        with open(prompt_file, "r") as f:
            self.prompt = f.read()
        return dataset

    def get_in_context(self):
        return self.prompt
    
    def get_label(self, raw_data):
        return raw_data[1]
    
    def get_question(self, raw_data):
        return raw_data[0]
        
    def get_case(self, raw_data):
        prompt = ""
        if self.if_in_context:
            prompt = self.get_in_context()
        case = dict()
        prompt_tail = "Please reason it step by step. End this question in format of \"Answer: {valid or invalid, a single word}.\""
        case["question"] = prompt + "Q:" + self.get_question(raw_data) + prompt_tail
        case["label"] = self.get_label(raw_data)
        return case
    
    def match_answer(self, answer_str):
        match = re.search(r'[Aa]nswer:\s*(.*)', answer_str)
        if match:
            squence = match.group(1).replace(".", "").lower()
            print("match: " + str(squence))
            return str(squence)
        else:
            return None
            