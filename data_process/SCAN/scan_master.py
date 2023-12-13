import random
import json
import pandas as pd
import zipfile
import os
import re
import ast
import copy
from collections import deque

class scan_master():
    def __init__(self, if_in_context = False, n_shots=0, n_noisy_shots=0, noisy_type="irrelative", noisy_level = 1, prefix_context =True, config: dict = None, reasoning_type = "length") -> None:
        self.if_in_context = if_in_context
        self.n_shots = n_shots
        self.n_noisy_shots = n_noisy_shots
        self.prefix_context = prefix_context
        if self.n_noisy_shots > 0:
            self.noisy_type = noisy_type
            self.noisy_level = noisy_level
        else:
            self.noisy_type = None
            self.noisy_level = 0
        if config is not None:
            self.reasoning_type = config["reasoning_type"]
        else:
            self.reasoning_type = reasoning_type
        
        self.file_path = os.path.join("data", "SCAN-master")
        # self.unzip_data()
        # self.init_noise_data()
        # self.init_relation_dict()
        return
    
    
    # def init_relation_dict(self):
    #     with open(os.path.join(self.file_path,  "relation_dict.json"), "r") as f:
    #         kv_list = json.load(f)
    #     self.relation_dict = dict()
    #     for kv in kv_list:
    #         key = kv["key"]
    #         key = (key[0], key[1])
    #         value = kv["value"]
    #         self.relation_dict[key] = value
    #     return
        
    
    # def get_config(self):
    #     config = dict()
    #     config["reasoning_type"] = self.reasoning_type
    #     config["hop"] = self.hop
    #     return config
    
    # def init_noise_data(self):
        # with open(os.path.join(self.file_path, "noise", "noise_relation_facts.json"), "r") as f:
        #     noise_facts = json.load(f)
        # self.noise_relation_facts = dict()
        # for relation_facts in noise_facts:
        #     relation = relation_facts["relation"]
        #     facts = relation_facts["facts"] 
        #     self.noise_relation_facts[relation] = facts
            
        # with open(os.path.join(self.file_path, "noise", "noise_facts.json"), "r") as f:
        #     noise_facts = json.load(f)
        # self.noise_facts = dict()
        # for relation_facts in noise_facts:
        #     relation = relation_facts["relation"]
        #     facts = relation_facts["facts"] 
        #     self.noise_facts[relation] = facts
    
    def unzip_data(self):
        zip_files = ["data_7c5b0e70.zip", "data_06b8f2a1.zip", "data_523348e6.zip", "data_d83ecc3e.zip", "data_db9b8f04.zip"]
        unzip_path = os.path.join(self.file_path, "unzip_data")
        if not os.path.exists(unzip_path):
            os.makedirs(unzip_path)
        for index, zip_file in enumerate(zip_files):
            with zipfile.ZipFile(os.path.join(self.file_path, zip_file), 'r') as zip_ref:
                unzip_file_path = os.path.join(unzip_path, str(index + 1))
                if not os.path.exists(unzip_file_path):
                    os.makedirs(unzip_file_path)
                zip_ref.extractall(unzip_file_path)
    
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
        
    def get_question(self, raw_data):
        in_content = raw_data[0]
        question = f"With IN:{in_content}, what is OUT?\n"
        question += "Please reason it step by step, and provide the final action sequence as the answer. End the question with \"So, final answer is OUT: <action sequence>\"\n"
        return question
    
    def get_label(self, raw_data):
        return str(raw_data[1])
        
    def get_random_fact(self, relation, selected_set):
        facts = self.noise_facts[relation]
        while(1):
            random_index = random.randrange(0, len(facts))
            selected = f"{relation}_{random_index}"
            if selected not in selected_set:
                fact = facts[random_index]
                selected_set.add(selected)
                break
        return fact
    
    def get_random_relation_fact(self, relation):
        facts = self.noise_relation_facts[relation]
        random_index = random.randrange(0, len(facts))
        fact = facts[random_index]
        return fact
        
    def get_random_relation(self, original_relation):
        relation_num = len(self.relation_list)
        while 1:
            random_index = random.randrange(0, relation_num)
            new_relation = self.relation_list[random_index]
            if new_relation != original_relation:
                break
        return new_relation
    def find_sentence_containing_strings(self, text, name1, name2):
        sentences = text.split('.')
        for sentence in sentences:
            if name1 in sentence and name2 in sentence:
                return sentence
        return None
    
    def get_answer(self, raw_data, if_noise):
        in_content = raw_data[0]
        label = raw_data[1]
        answer = ""
        direction = ["right", "left"]
        angle = ["opposite", "around"]
        times_phrase = ["twice", "thrice"]
        
        action_sequence = []
        if "and" in in_content.split():
            sub_action_list = [actions.split() for actions in in_content.split("and")]
            answer += "Since command is {}, we should consider Step1: \"{}\" firstly. \n".format(in_content, " ".join(sub_action_list[0]))
        elif "after" in in_content.split():
            sub_action_list = [actions.split() for actions in in_content.split("after")]
            answer += "Since command is {}, we should consider Step1: \"{}\" firstly. \n".format(in_content, " ".join(sub_action_list[1]))
            sub_action_list.reverse()
        else:
            sub_action_list = [in_content.split()]
            answer += "Let's consider {}. \n".format(" ".join(sub_action_list[0]))
        
        
        for i, actions in enumerate(sub_action_list):
            
            sub_action_sequence = []
            actions_str = " ".join(actions)
            if i > 0:
                answer += "Now, we consider Step2:\"{}\". ".format(actions_str)
            if len(actions) > 4:
                print(f"err:{actions_str}, length")
                continue
            this_times = ""
            this_direction = ""
            this_angle = ""
            if actions[0] == "turn":
                action_kind = 1
            else:
                action_kind = 2
            this_action = actions[0]
            
            if len(actions) > 1:
                if actions[1] in direction:
                    this_direction = actions[1]
                    if len(actions) == 3:
                        if actions[2] not in times_phrase:
                            print(f"err:{actions_str}, times_phrase")
                            continue
                        this_times = actions[2]
                elif actions[1] in angle:
                    this_angle = actions[1]
                    if actions[2] not in direction:
                        print(f"err:{actions_str}, direction")
                        continue
                    this_direction = actions[2]
                    if len(actions) == 4:
                        if actions[3] not in times_phrase:
                            print(f"err:{actions_str}, times_phrase")
                            continue
                        this_times = actions[3]
                elif actions[1] in times_phrase:
                    this_times = actions[1]
                else:
                    print(f"err:{actions_str}, no angle and direction")
                    continue
                
                    
            if this_direction == "":
                once_action = []
                once_action.append(f"I_{this_action.upper()}")
                answer += "\"{}\" means the agent needs to {}. So, in action sequence is {}. ".format(this_action, this_action, " ".join(once_action))
            elif this_angle == "":
                once_action = []
                answer += f"\"{this_action} {this_direction}\" means the agent needs to turn {this_direction}"
                once_action.append(f"I_TURN_{this_direction.upper()}")
                if action_kind == 2:
                    answer += f" and {this_action}"
                    once_action.append(f"I_{this_action.upper()}")
                answer += ". "
                if if_noise:
                    if this_direction == "left":
                        answer += "In countries with right-hand traffic, turning left at a traffic light often requires a green arrow to indicate a protected turn. "
                    elif this_direction == "right":
                        answer += "Turning right in countries that drive on the right side of the road typically does not intersect with oncoming traffic. "
                answer += "So, in action sequence is {}. ".format(" ".join(once_action))
            else:
                once_action = []
                if this_angle == "opposite":
                    answer += f"\"{this_action} {this_angle} {this_direction}\" means the agent needs to turn {this_direction} twice"
                    once_action.append(f"I_TURN_{this_direction.upper()}")
                    once_action.append(f"I_TURN_{this_direction.upper()}")
                    if action_kind == 2:
                        answer += f" before {this_action}. "
                        once_action.append(f"I_{this_action.upper()}")
                    else:
                        answer += f". "
                    if if_noise:
                        if this_direction == "left":
                            answer += "The command to 'turn opposite' is not standard in directional terms but implies facing or moving in the opposite direction."
                            answer += "Making a left turn typically exposes a driver to oncoming traffic, increasing the complexity of the turn relative to turning right."
                        elif this_direction == "right":
                                answer += "In military drill commands, a command to 'about face' is the equivalent of turning to the opposite direction."
                                answer += "In left-hand traffic jurisdictions, such as the UK, turning right is analogous to turning left in right-hand traffic jurisdictions, crossing the path of oncoming vehicles."
                    
                elif this_angle == "around":
                    answer += f"\"{this_action} {this_angle} {this_direction}\" means the agent needs to turn {this_direction}"
                    once_action.append(f"I_TURN_{this_direction.upper()}")
                    if action_kind == 2:
                        answer += f" and {this_action}"
                        once_action.append(f"I_{this_action.upper()}")
                    angle_times = 4
                    answer += ", and repeat this action sequence four times to complete a 360-degree loop"
                    answer += ". "
                    once_action = once_action * angle_times
                    if if_noise:
                        answer += "Many GPS navigation systems will issue a 'turn around' command if the driver deviates from the planned route."
                answer += "So, in action sequence is {}. ".format(" ".join(once_action))
                
            if this_times != "":
                if this_times == "twice":
                    action_times = 2
                if this_times == "thrice":
                    action_times = 3
                sub_action_sequence = once_action * action_times
                answer += f"Since we need do {this_times} in command \"{actions_str}\",  this entire sequence is repeated {action_times} times. "
                answer += "So the action sequence to \"{}\" is :{}".format(actions_str, " ".join(sub_action_sequence))        
            else:
                sub_action_sequence = once_action
            answer += "\n"

            action_sequence = action_sequence + sub_action_sequence
        
        answer += "Above all -- So, final answer is OUT:{}".format(" ".join(action_sequence))
        
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
            
            
        
        