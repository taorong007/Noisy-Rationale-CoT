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

class tracking_shuffled_objects():
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
            if noise_type != "irrelevant" and noise_type != "minor_error":
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
        
        self.file_path = os.path.join("data", "tracking_shuffled_objects")
        self.unzip_data()
        self.init_noise_data()
        return

    
    def init_noise_data(self):
        with open(os.path.join(self.file_path, "noise", "action_facts.json"), "r") as f:
            noise_facts = json.load(f)
        self.noise_facts = dict()
        for noise_fact in noise_facts:
            phrase = noise_fact["phrase"]
            facts = noise_fact["facts"] 
            self.noise_facts[phrase] = facts

    def unzip_data(self):
        zip_files = "SCAN-master.zip"
        unzip_path = os.path.join(self.file_path, "unzip_data")
        if not os.path.exists(unzip_path):
            os.makedirs(unzip_path)
        with zipfile.ZipFile(os.path.join(self.file_path, zip_files), 'r') as zip_ref:
            zip_ref.extractall(unzip_path)
    
    def read_raw_file(self, file_path):
        dataset = []
        pattern = r'IN:\s*(.*?)\s*OUT:\s*(.*)'
        with open(file_path) as f:
            lines = f.readlines()
            for line in lines:
                match = re.search(pattern, line)
                action_list = []
                if match:
                    in_content = match.group(1)
                    out_content = match.group(2)
                    action_list = out_content.split(" ")
                    dataset.append([in_content, action_list])
                else:
                    print(f"No match found, {line}")
        return dataset
    
    def load_data(self):
        # unzip_path = os.path.join(self.file_path, "unzip_data", str(self.trainset))
        # trainset_file_name = f"{self.trainset}.2,{self.trainset}.3_train.csv"
        # testset_file_name = f"{self.testset}_test.csv"
        # self.trainset = pd.read_csv(os.path.join(unzip_path, trainset_file_name))
        unzip_path = os.path.join(self.file_path, "unzip_data", f"{self.reasoning_type}_split")
        if self.reasoning_type ==  "length":
            train_file_name = "tasks_train_length.txt"
            test_file_name = "tasks_test_length.txt"
        if self.reasoning_type ==  "simple":
            train_file_name = "tasks_train_simple.txt"
            test_file_name = "tasks_test_simple.txt"
        else:
            raise ValueError(f"reasoning type{self.reasoning_type} not support")
        
        with open(os.path.join(self.file_path, "base_example.json"), "r") as f:
            self.base_example = json.load(f)
        
        trainset = self.read_raw_file(os.path.join(unzip_path, train_file_name))
        testset = self.read_raw_file(os.path.join(unzip_path, test_file_name))
        self.trainset = trainset
        return testset
    
    def get_sys_prompt(self):
        prompt = "I will send you some instructions and you should provide me their corresponding action sequences.  These are the actions an agent should perform to execute the commands successfully. The commands and actions are defined compositionally based on primitives ('I_JUMP', 'I_LOOK', 'I_RUN', 'I_TURN_LEFT', 'I_TURN_RIGHT', 'I_WALK') and modifiers such as \"twice\", \"thrice\", \"and\", \"after\", \"around left\", etc. Here are some basic examples.\n"
        for i, base_example in enumerate(self.base_example):
            prompt += "example{}:\nIN:\n{}OUT:{}\n".format(i, base_example["IN"], base_example["OUT"])

        return prompt
    
    def _prepare_noise_distribution_iteration_state(self, n_thought, noise_ratio, noise_distribution):
        noise_distribution_list = [0] * n_thought
        if noise_distribution == "fixed":
            noise_count = math.ceil(n_thought * noise_ratio)
            noise_positions = random.sample(range(n_thought), noise_count)
            for pos in noise_positions:
                noise_distribution_list[pos] = 1
        else:
            for pos in range(len(noise_distribution_list)):
                if random.random() < noise_ratio:
                    noise_distribution_list[pos] = 1 
        return [noise_distribution_list, 0]
        
    def _should_add_noise(self, noise_distribution_state):
        if noise_distribution_state == None:
            return 0
        distribution_list = noise_distribution_state[0]
        pos = noise_distribution_state[1]
        if_noise = distribution_list[pos]
        noise_distribution_state[1] += 1
        return if_noise
        
    def get_question(self, raw_data):
        in_content = raw_data[0]
        question = f"With IN:{in_content}, what is OUT?\n"
        question += "Please reason it step by step, and provide the final action sequence as the answer. End the question with \"So, final answer is OUT: <action sequence>\"\n"
        return question
    
    def get_label(self, raw_data):
        return str(raw_data[1])
        
    def get_random_fact(self, rephrase, selected_set):
        facts = self.noise_facts[rephrase]
        while(1):
            random_index = random.randrange(0, len(facts))
            selected = f"{rephrase}_{random_index}"
            if selected not in selected_set:
                fact = facts[random_index]
                selected_set.add(selected)
                break
        return fact
    
        
    def find_sentence_containing_strings(self, text, name1, name2):
        sentences = text.split('.')
        for sentence in sentences:
            if name1 in sentence and name2 in sentence:
                return sentence
        return None
    
    def error_action(self, origin_action):
        actions = ["jump", "run", "walk", "look"]
        while 1:
            random_index = random.randrange(0, len(actions))
            choose_action = actions[random_index]
            if choose_action != origin_action:
                break
        return choose_action
    
    def error_direction(self, direction):
        if direction == "left":
            return "right"
        else:
            return "left"
    
    
    def _get_answer(self, in_content, ir_noise_distrib_state=None, mn_noise_distrib_state=None):    
        return answer, action_sequence, n_ir_pos, n_mn_pos
    
    def get_answer(self, raw_data, if_noise):
        in_content = raw_data[0]
        label = raw_data[1]
        
        _, _, n_ir_pos, n_mn_pos = self._get_answer(in_content)
        
        
        if self.noise_type == "irrelevant":
            ir_noise_p = self.noise_ratio
            mn_noise_p = 0
        else:
            ir_noise_p = 0
            mn_noise_p = self.noise_ratio
        ir_noise_distrib_state = self._prepare_noise_distribution_iteration_state(n_ir_pos, ir_noise_p, self.noise_distribution) 
        mn_noise_distrib_state = self._prepare_noise_distribution_iteration_state(n_mn_pos, mn_noise_p, self.noise_distribution)                 
        answer,action_sequence, _, _ = self._get_answer(in_content, ir_noise_distrib_state, mn_noise_distrib_state)
        
        # print(action_sequence)
        assert str(action_sequence) == str(label)
        return answer

    def get_random_demos(self, num):
        assert len(self.trainset) > num
        demos = random.sample(self.trainset, num)
        return demos
    
    def get_case(self, raw_data):
        case = dict()
        qustion  = self.get_question(raw_data)
        label = self.get_label(raw_data)
        shots = []
        system_prompt = self.get_sys_prompt()
        prefix = ""
        if(self.if_in_context):
            n_shots = self.n_shots
            n_noisy_shots = self.n_noisy_shots
            n_total_shots = n_shots + n_noisy_shots
            if n_total_shots:
                # demos = self.example
                # demos = demos[:n_total_shots]
                # for demo in demos:
                #     shots.append([demo["question"], demo["answer"]])
                demos = self.get_random_demos(num=n_total_shots)
                normal_demos = demos[:n_shots]
                noise_demos = demos[n_shots:]
                for demo in normal_demos:
                    shot_q = self.get_question(demo)
                    shot_a = self.get_answer(demo, if_noise=False)
                    shots.append([shot_q, shot_a])
                for demo in noise_demos:
                    shot_q = self.get_question(demo)
                    shot_a = self.get_answer(demo, if_noise=True)
                    shots.append([shot_q, shot_a])
                if self.prefix_context:
                    prefix += prefix
                    for shot in shots:
                        prefix += "user:{}\nassistant:{}\n".format(shot[0], shot[1])
                    prefix += "user:"
                else:    
                    case["in-context"] = shots
                    case["system-prompt"] = system_prompt
                    
        case["question"] = prefix + qustion
        case["label"] = label
        return case
    
    def match_answer(self, answer_str):
        match = re.search(r'OUT:\s*(.*)', answer_str)
        if match:
            squence_str = re.sub(r'[^a-zA-Z0-9_\s]+', '', match.group(1)).strip()
            squence = squence_str.split()
            print("match: " + str(squence))
            return str(squence)
        else:
            return None
            
            
        
        