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

class symbolic():
    def __init__(self, n_shots=0, n_noisy_shots=0, noise_type="irrelevant", noise_ratio = 0.5, noise_distribution = "fixed", prefix_context =True, config: dict = None, subtask = "longer") -> None:
        
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
            self.subtask = config["subtask"]
        else:
            self.subtask = subtask
        
        self.file_path = os.path.join("data", "symbolic")
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
        zip_files = "symbolic.zip"
        unzip_path = os.path.join(self.file_path, "unzip_data")
        if not os.path.exists(unzip_path):
            os.makedirs(unzip_path)
        with zipfile.ZipFile(os.path.join(self.file_path, zip_files), 'r') as zip_ref:
            zip_ref.extractall(unzip_path)
    
    # def shuffle_file(self, file_path):
    #     with open(file_path, 'r') as f:
    #         lines = f.readlines()
    #     random.shuffle(lines)
    #     with open("shuffled_test.txt", 'w') as f:
    #         f.writelines(lines)
    
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
        if self.subtask == "longer":
            split_name = "length"
        else:
            split_name = "simple"
        unzip_path = os.path.join(self.file_path, "unzip_data", f"{split_name}_split")
        if self.subtask ==  "longer":
            train_file_name = "tasks_train_length.txt"
            test_file_name = "tasks_test_length.txt"
        elif self.subtask ==  "equal":
            train_file_name = "tasks_train_simple.txt"
            test_file_name = "tasks_test_simple.txt"
        else:
            raise ValueError(f"reasoning type{self.subtask} not support")
        
        with open(os.path.join(self.file_path, "base_example.json"), "r") as f:
            self.base_example = json.load(f)
        
        trainset = self.read_raw_file(os.path.join(unzip_path, train_file_name))
        testset = self.read_raw_file(os.path.join(unzip_path, test_file_name))
        self.trainset = trainset
        if self.subtask == "longer":
            random.shuffle(testset)
            # self.shuffle_file(os.path.join(unzip_path, test_file_name))
            # testset = self.read_raw_file("shuffled_test.txt")
        return testset
    
    def get_sys_prompt(self):
        prompt = "I will send you some instructions and you should provide me their corresponding action sequences.  These are the actions an agent should perform to execute the commands successfully. The commands and actions are defined compositionally based on primitives ('I_JUMP', 'I_LOOK', 'I_RUN', 'I_TURN_LEFT', 'I_TURN_RIGHT', 'I_WALK') and modifiers such as \"twice\", \"thrice\", \"and\", \"after\", \"around left\", etc. Here are some basic examples.\n"
        for i, base_example in enumerate(self.base_example):
            prompt += "example{}:\nIN:\n{}OUT:{}\n".format(i, base_example["IN"], base_example["OUT"])

        return prompt
    
    def _prepare_noise_distribution_iteration_state(self, n_thought, noise_ratio, noise_distribution):
        noise_distribution_list = [0] * n_thought
        if noise_distribution == "fixed":
            noise_count = round(n_thought * noise_ratio)
            if noise_ratio > 0 and noise_count == 0:
                noise_count = 1
            # if noise_ratio > 0:
            #     print(noise_count)
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
        question += "Please reason it step by step, and provide the final action sequence as the answer. End the response with \"So, final answer is OUT: <action sequence>\"\n"
        return question
    
    def get_label(self, raw_data):
        return str(raw_data[1])
        
    def get_random_fact(self, phrase, selected_set):
        facts = self.noise_facts[phrase]
        while(1):
            random_index = random.randrange(0, len(facts))
            selected = f"{phrase}_{random_index}"
            if selected not in selected_set:
                fact = facts[random_index] + " "
                selected_set.add(selected)
                break
        return fact
    
        
    def find_sentence_containing_strings(self, text, name1, name2):
        sentences = text.split('.')
        for sentence in sentences:
            if name1 in sentence and name2 in sentence:
                return sentence
        return None
    
    def another_action(self, origin_action):
        actions = ["jump", "run", "walk", "look"]
        while 1:
            random_index = random.randrange(0, len(actions))
            choose_action = actions[random_index]
            if choose_action != origin_action:
                break
        return choose_action
    
    def other_direction(self, direction):
        if direction == "left":
            return "right"
        else:
            return "left"
    
    def capitalize(self, s):
        return s[0].upper() + s[1:]
        
       
    def get_inaccurate_thought_of_action(self, action):
        inaccurate_thought = ""
        another_action = self.another_action(action)
        error_action = self.another_action(another_action)
        inaccurate_thought += f"The term '{another_action}' corresponds to the command I_{error_action.upper()}. "
        return inaccurate_thought
    
    def get_inaccurate_thought_of_direction(self, direction):
        inaccurate_thought = ""
        the_other_direction = self.other_direction(direction)
        error_direction = direction
        inaccurate_thought += f"The term '{the_other_direction}' translates to  I_TURN_{error_direction.upper()}. "
        return inaccurate_thought
    
    def get_inaccurate_thought_of_angle(self, angle):
        inaccurate_thought = ""
        if angle == "around":
            random_squence = random.choice(["I_TURN_LEFT, I_TURN_RIGHT",  "I_TURN_RIGHT, I_TURN_LEFT", "I_TURN_LEFT", "I_TURN_RIGHT"])
            inaccurate_thought = f"The term 'opposite' implies a 180-degree turn, which requires {random_squence}. "
        else:
            random_squence = random.choice(["I_TURN_LEFT, I_TURN_RIGHT, I_TURN_LEFT, I_TURN_RIGHT",  "I_TURN_LEFT, I_TURN_RIGHT, I_TURN_RIGHT", "I_TURN_RIGHT, I_TURN_RIGHT, I_TURN_RIGHT", "I_TURN_LEFT, I_TURN_LEFT, I_TURN_LEFT"])
            inaccurate_thought = f"The term 'around' implies a 360-degree loop, which requires {random_squence}. "
        return inaccurate_thought

    def get_inaccurate_thought_of_times(self, times_phrase):
        if times_phrase == "twice":
            another_phrase = "thrice"
            times = 3
        else:
            another_phrase = "twice"
            times = 2
        err_times =  times + random.randrange(1,5,1)
        inaccurate_thought =  f"{another_phrase} means the entire sequence is repeated {err_times} times"
        return inaccurate_thought
    
    def _get_answer(self, in_content, ir_noise_distrib_state=None, mn_noise_distrib_state=None):
        answer = ""
        direction = ["right", "left"]
        angle = ["opposite", "around"]
        times_phrase = ["twice", "thrice"]
        action_sequence = []
        selected_set = set()
        
        if "and" in in_content.split():
            sub_action_list = [actions.split() for actions in in_content.split("and")]
            answer += "Since command is {}, we should consider Step1: \"{}\" firstly, \n".format(in_content, " ".join(sub_action_list[0]))
        elif "after" in in_content.split():
            sub_action_list = [actions.split() for actions in in_content.split("after")]
            answer += "Since command is {}, we should consider Step1: \"{}\" firstly, \n".format(in_content, " ".join(sub_action_list[1]))
            sub_action_list.reverse()
        else:
            sub_action_list = [in_content.split()]
            answer += "Let's consider {}, \n".format(" ".join(sub_action_list[0]))
        
        n_ir_pos = 0
        n_mn_pos = 0
        
        for i, actions in enumerate(sub_action_list):
            actions_str = " ".join(actions)
            if i > 0:
                answer += "Now, we consider Step2:\"{}\", ".format(actions_str)
            if len(actions) > 4:
                print(f"err:{actions_str}, len")
                continue
            this_times = ""
            this_direction = ""
            this_angle = ""
            if actions[0] == "turn":
                this_action_kind = 1
            else:
                this_action_kind = 2
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
                
                # action
                answer += "\"{}\" means the agent needs to {}, in action sequence is {}. ".format(this_action, this_action, " ".join(once_action))
                
                n_mn_pos += 1
                if self._should_add_noise(mn_noise_distrib_state):
                    answer += self.get_inaccurate_thought_of_action(this_action)
                
                n_ir_pos += 1
                if self._should_add_noise(ir_noise_distrib_state):
                    noise_fact = self.get_random_fact(this_action, selected_set)
                    answer += noise_fact
                    
            elif this_angle == "":
                once_action = []
                once_action.append(f"I_TURN_{this_direction.upper()}")
                answer += f"\"{this_action} {this_direction}\" means the agent needs to turn {this_direction}"
                if this_action_kind == 2:
                    once_action.append(f"I_{this_action.upper()}")
                    answer += f" and {this_action}"
                answer += ". "
                
                n_ir_pos += 1
                if self._should_add_noise(ir_noise_distrib_state):
                    noise_fact = self.get_random_fact(this_direction, selected_set)
                    answer += noise_fact
                
                n_mn_pos += 1
                if self._should_add_noise(mn_noise_distrib_state):
                    answer += self.get_inaccurate_thought_of_direction(this_direction)
                
                if self.subtask == "equal":
                    # direction
                    answer += f"The '{this_direction}' corresponds to the command I_TURN_{this_direction.upper()}. "
                    
                    n_ir_pos += 1
                    if self._should_add_noise(ir_noise_distrib_state):
                        noise_fact = self.get_random_fact(this_direction, selected_set)
                        answer += noise_fact
                    
                    n_mn_pos += 1
                    if self._should_add_noise(mn_noise_distrib_state):
                        answer += self.get_inaccurate_thought_of_direction(this_direction)
                    
                    
                    if this_action_kind == 2:
                        # action
                        answer +=  f"Subsequently, '{this_action}' translates to I_{this_action.upper()}. "
                        n_ir_pos += 1
                        if self._should_add_noise(ir_noise_distrib_state):
                            noise_fact = self.get_random_fact(this_action, selected_set)
                            answer += noise_fact
                        n_mn_pos += 1
                        if self._should_add_noise(mn_noise_distrib_state):
                            answer += self.get_inaccurate_thought_of_action(this_action)
                
                answer += "Therefore, the action sequence is {}. ".format(" ".join(once_action))
                n_ir_pos += 1
                if self._should_add_noise(ir_noise_distrib_state):
                    noise_fact = self.get_random_fact("action sequence", selected_set)
                    answer += noise_fact
                n_mn_pos += 1
                if self._should_add_noise(mn_noise_distrib_state):
                    answer += self.get_inaccurate_thought_of_action(this_direction)
            else:
                once_action = []
                if this_angle == "opposite":
                    
                    answer += f"\"{this_action} {this_angle} {this_direction}\" means the agent needs to turn {this_direction} twice"
                    once_action.append(f"I_TURN_{this_direction.upper()}")
                    once_action.append(f"I_TURN_{this_direction.upper()}")
                    if this_action_kind == 2:
                        once_action.append(f"I_{this_action.upper()}")
                        answer += f" before {this_action}"
                    answer += ". "       
                    n_ir_pos += 1
                    if self._should_add_noise(ir_noise_distrib_state):
                        noise_fact = self.get_random_fact("opposite", selected_set)
                        answer += noise_fact
                    n_mn_pos += 1
                    if self._should_add_noise(mn_noise_distrib_state):
                        answer += self.get_inaccurate_thought_of_angle(this_angle)
                    
                    if self.subtask == "equal":
                        # direction
                        answer += f"'{this_direction}' corresponds to the command I_TURN_{this_direction.upper()}. "
                        n_ir_pos += 1
                        if self._should_add_noise(ir_noise_distrib_state):
                            noise_fact = self.get_random_fact(this_direction, selected_set)
                            answer += noise_fact
                        n_mn_pos += 1
                        if self._should_add_noise(mn_noise_distrib_state):
                            answer += self.get_inaccurate_thought_of_direction(this_direction)
                        
                        
                        if this_action_kind == 2:
                            # action
                            answer += f"'{this_action}' translates to I_{this_action.upper()}. "
                            n_ir_pos += 1
                            if self._should_add_noise(ir_noise_distrib_state):
                                noise_fact = self.get_random_fact(this_action, selected_set)
                                answer += noise_fact
                            n_mn_pos += 1
                            if self._should_add_noise(mn_noise_distrib_state):
                                answer += self.get_inaccurate_thought_of_action(this_action)
                        
                        # angle
                        if this_action_kind == 2:    
                            answer += f"The term 'opposite' implies a 180-degree turn, which requires the agent to perform the turn {this_direction} twice before {this_action}. " 
                        else:
                            answer += f"The term 'opposite' implies a 180-degree turn, which requires the agent to perform the turn {this_direction} twice. " 
                        n_ir_pos += 1
                        if self._should_add_noise(ir_noise_distrib_state):
                            noise_fact = self.get_random_fact("opposite", selected_set)
                            answer += noise_fact
                        n_mn_pos += 1
                        if self._should_add_noise(mn_noise_distrib_state):
                            answer += self.get_inaccurate_thought_of_angle(this_angle)
                        
                elif this_angle == "around":
                    answer += f"\"{this_action} {this_angle} {this_direction}\" means the agent needs to turn {this_direction}"
                    once_action.append(f"I_TURN_{this_direction.upper()}")
                    if this_action_kind == 2:
                        once_action.append(f"I_{this_action.upper()}")
                        answer += f" and {this_action}"            
                    once_action = 4 * once_action
                    answer += ", and repeat this action sequence four times to complete a 360-degree loop. "
                    n_ir_pos += 1
                    if self._should_add_noise(ir_noise_distrib_state):
                        noise_fact = self.get_random_fact("around", selected_set)
                        answer += noise_fact
                    n_mn_pos += 1
                    if self._should_add_noise(mn_noise_distrib_state):
                        answer += self.get_inaccurate_thought_of_angle(this_angle)

                    if self.subtask == "equal":
                        # direction
                        answer += f"'{this_direction}' corresponds to the command I_TURN_{this_direction.upper()}. "
                        n_ir_pos += 1
                        if self._should_add_noise(ir_noise_distrib_state):
                            noise_fact = self.get_random_fact(this_direction, selected_set)
                            answer += noise_fact
                        n_mn_pos += 1
                        if self._should_add_noise(mn_noise_distrib_state):
                            answer += self.get_inaccurate_thought_of_direction(this_direction)     
                        
                        if this_action_kind == 2:
                            # action
                            answer += f"'{this_action}' translates to I_{this_action.upper()}. "
                            n_ir_pos += 1
                            if self._should_add_noise(ir_noise_distrib_state):
                                noise_fact = self.get_random_fact(this_action, selected_set)
                                answer += noise_fact
                            n_mn_pos += 1
                            if self._should_add_noise(mn_noise_distrib_state):
                                answer += self.get_inaccurate_thought_of_action(this_action)

                        # angle
                        if this_action_kind == 2:
                            answer += f"The term 'around' implies to complete a 360-degree loop by repeating the sequence of I_TURN_{this_direction.upper()}, I_{this_action.upper()} four times. "     
                        else:
                            answer += f"The term 'around' implies to complete a 360-degree loop by repeating the sequence of I_TURN_{this_direction.upper()} four times. "  
                        n_ir_pos += 1
                        if self._should_add_noise(ir_noise_distrib_state):
                            noise_fact = self.get_random_fact("around", selected_set)
                            answer += noise_fact
                        n_mn_pos += 1
                        if self._should_add_noise(mn_noise_distrib_state):
                            answer += self.get_inaccurate_thought_of_angle(this_angle)

                    
                answer += "Therefore, the action sequence is {}. ".format(" ".join(once_action))
                n_ir_pos += 1
                if self._should_add_noise(ir_noise_distrib_state):
                    noise_fact = self.get_random_fact("action sequence", selected_set)
                    answer += noise_fact
                n_mn_pos += 1
                if self._should_add_noise(mn_noise_distrib_state):
                    answer += self.get_inaccurate_thought_of_angle(this_angle)
                
            if this_times != "":
                if this_times == "twice":
                    sub_action_sequence = once_action * 2
                    action_times = 2
                if this_times == "thrice":
                    sub_action_sequence = once_action * 3
                    action_times = 3
                answer += f"Since we need do {this_times} in command \"{actions_str}\", this entire sequence is repeated {action_times} times, "
                answer += "so the action sequence to \"{}\" is :{}. ".format(actions_str, " ".join(sub_action_sequence))
                n_ir_pos += 1
                if self._should_add_noise(ir_noise_distrib_state):
                    noise_fact = self.get_random_fact(this_times, selected_set)
                    answer += noise_fact
                    
                n_mn_pos += 1
                if self._should_add_noise(mn_noise_distrib_state):
                    answer += self.capitalize(self.get_inaccurate_thought_of_times(this_times) + ". ")
            else:
                sub_action_sequence = once_action
            answer += "\n"

            action_sequence = action_sequence + sub_action_sequence
        
        answer += "Above all -- So, final answer is OUT:{}. ".format(" ".join(action_sequence))
        
        return answer, action_sequence, n_ir_pos, n_mn_pos
    
    def get_generation_config(self, noise_distribution_state, generate_info):
        noise_distribution_list = noise_distribution_state[0]
        generate_info["total_thought"] = len(noise_distribution_list) + noise_distribution_list.count(1)
        generate_info["noise_thought"] = noise_distribution_list.count(1)
        generate_info["sentences_with_noise"] = []
        for if_noise in noise_distribution_list:
            generate_info["sentences_with_noise"].append(0)
            if if_noise:
                generate_info["sentences_with_noise"].append(1)
        generate_info["sentences_with_noise"].append(0)
    
    def get_correct_answer(self, raw_data):
        in_content = raw_data[0]
        label = raw_data[1]
        _, _, n_ir_pos, n_mn_pos = self._get_answer(in_content) 
        ir_noise_p = 0
        mn_noise_p = 0
        ir_noise_distrib_state = self._prepare_noise_distribution_iteration_state(n_ir_pos, ir_noise_p, self.noise_distribution) 
        mn_noise_distrib_state = self._prepare_noise_distribution_iteration_state(n_mn_pos, mn_noise_p, self.noise_distribution) 
        answer,action_sequence, _, _ = self._get_answer(in_content, ir_noise_distrib_state, mn_noise_distrib_state)
        assert str(action_sequence) == str(label)
        return answer
    
    def get_answer(self, raw_data, generate_info = None):
        in_content = raw_data[0]
        label = raw_data[1]
        
        _, _, n_ir_pos, n_mn_pos = self._get_answer(in_content)
        if self.noise_type == "irrelevant":
            ir_noise_p = self.noise_ratio
            mn_noise_p = 0
        elif self.noise_type == "inaccurate":
            ir_noise_p = 0
            mn_noise_p = self.noise_ratio
        else:
            ir_noise_p = 0
            mn_noise_p = 0
        ir_noise_distrib_state = self._prepare_noise_distribution_iteration_state(n_ir_pos, ir_noise_p, self.noise_distribution) 
        mn_noise_distrib_state = self._prepare_noise_distribution_iteration_state(n_mn_pos, mn_noise_p, self.noise_distribution) 
        if generate_info is not None:          
            if(self.noise_type == "irrelevant"):
                self.get_generation_config(ir_noise_distrib_state, generate_info)
            elif(self.noise_type == "inaccurate"):
                self.get_generation_config(mn_noise_distrib_state, generate_info)
            else:
                self.get_generation_config(ir_noise_distrib_state, generate_info)
        
        answer,action_sequence, _, _ = self._get_answer(in_content, ir_noise_distrib_state, mn_noise_distrib_state)
        # print(action_sequence)
        assert str(action_sequence) == str(label)
        return answer

    def get_random_demos(self, num, index_list = None):
        assert len(self.trainset) > num
        indexed_trainset = list(enumerate(self.trainset))
        selected_samples = random.sample(indexed_trainset, num)
        if index_list is not None:
            index_list.extend([index for index, _ in selected_samples])
        demos = [demo for _, demo in selected_samples]
        return demos
    
    def get_demos_by_index_list(self, num, index_list):
        demos = []
        for i in range(num):
            index = index_list[i]
            demos.append(self.trainset[index])
        return demos
    
    def get_case(self, raw_data, if_generate_info=False, ICL_index_list=None):
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
                if ICL_index_list is None:
                    demos = self.get_random_demos(num=n_total_shots)
                else:
                    demos = self.get_demos_by_index_list(num=n_total_shots, index_list= ICL_index_list)
                for demo in demos:
                    if if_generate_info:
                        generate_info = dict()
                    else:
                        generate_info = None
                    shot_q = self.get_question(demo)
                    shot_a = self.get_answer(demo, generate_info=generate_info)
                    if if_generate_info:
                        shots.append([shot_q, shot_a, generate_info])
                    else:
                        shots.append([shot_q, shot_a])
                # for demo in noise_demos:
                #     shot_q = self.get_question(demo)
                #     shot_a = self.get_answer(demo)
                #     shots.append([shot_q, shot_a])
                # if self.prefix_context:
                #     prefix += prefix
                #     for shot in shots:
                #         prefix += "user:{}\nassistant:{}\n".format(shot[0], shot[1])
                #     prefix += "user:"
                # else:    
                case["in-context"] = shots
                case["system-prompt"] = system_prompt
                    
        case["question"] = prefix + qustion
        case["label"] = label
        return case
    
    @staticmethod
    def match_answer(answer_str):
        match = re.search(r'(?:.*OUT:\s*)([^\n]*)', answer_str, re.DOTALL)
        if match:
            squence_str = re.sub(r'[^a-zA-Z0-9_\s]+', '', match.group(1)).strip()
            squence = squence_str.split()
            print("match: " + str(squence))
            return str(squence)
        else:
            return None
            
            
        
        