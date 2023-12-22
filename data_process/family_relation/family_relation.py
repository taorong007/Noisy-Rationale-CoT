import random
import json
import pandas as pd
import zipfile
import os
import re
import ast
import copy
from collections import deque

class family_relation():
    def __init__(self, if_in_context = False, n_shots=0, n_noisy_shots=0, noise_type="irrelevant", noisy_level = 1, prefix_context =False, config: dict = None, reasoning_type = "symbolic", hop = 3, trainset=5, testset = 5.3) -> None:
        self.if_in_context = if_in_context
        self.n_shots = n_shots
        self.n_noisy_shots = n_noisy_shots
        self.prefix_context = prefix_context
        if self.n_noisy_shots > 0:
            self.noise_type = noise_type
            self.noisy_level = noisy_level
        else:
            self.noise_type = None
            self.noisy_level = 0
        if config is not None:
            self.trainset = config["train_set"]
            # self.testset = config["test_set"]
            self.reasoning_type = config["reasoning_type"]
            self.hop = config["hop"]
        else:
            self.trainset = trainset
            self.testset = testset
            self.reasoning_type = reasoning_type
            self.hop = hop
        if reasoning_type == "symbolic":
            assert self.trainset >= 5
        elif reasoning_type == "story":
            assert self.trainset < 5
        else:
            raise ValueError(f"reasoning type not support {reasoning_type}")

        self.not_support_relation_reason = []
        self.replace_num = 0
        self.error_reason_num = 0
        self.file_path = os.path.join("data", "data_emnlp_final")
        self.unzip_data()
        self.init_noise_data()
        self.init_relation_dict()
        return
    
    def init_relation_dict(self):
        with open(os.path.join(self.file_path,  "relation_dict.json"), "r") as f:
            kv_list = json.load(f)
        self.relation_dict = dict()
        for kv in kv_list:
            key = kv["key"]
            key = (key[0], key[1])
            value = kv["value"]
            self.relation_dict[key] = value
        return
        
    
    def get_config(self):
        config = dict()
        config["reasoning_type"] = self.reasoning_type
        config["hop"] = self.hop
        return config
    
    def init_noise_data(self):
        with open(os.path.join(self.file_path, "noise", "noise_relation_facts.json"), "r") as f:
            noise_facts = json.load(f)
        self.noise_relation_facts = dict()
        for relation_facts in noise_facts:
            relation = relation_facts["relation"]
            facts = relation_facts["facts"] 
            self.noise_relation_facts[relation] = facts
            
        with open(os.path.join(self.file_path, "noise", "noise_facts.json"), "r") as f:
            noise_facts = json.load(f)
        self.noise_facts = dict()
        for relation_facts in noise_facts:
            relation = relation_facts["relation"]
            facts = relation_facts["facts"] 
            self.noise_facts[relation] = facts
    
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
    
    def shuffle_dataset(self, step2_set, step3_set):
        set_list = []
        for i in range(max(len(step2_set), len(step3_set))):
                if i < len(step2_set):
                    set_list.append(step2_set.iloc[i])
                if i < len(step3_set):
                    set_list.append(step3_set.iloc[i])
        shuffled_set = pd.concat(set_list, axis=1).T.reset_index(drop=True)
        return shuffled_set 
    
    def load_data(self):
        unzip_path = os.path.join(self.file_path, "unzip_data", str(self.trainset))
        # trainset_file_name = f"{self.trainset}.2,{self.trainset}.3_train.csv"
        # testset_file_name = f"{self.testset}_test.csv"
        # self.trainset = pd.read_csv(os.path.join(unzip_path, trainset_file_name))
        
        if self.reasoning_type !=  "symbolic":
            file_name = f"{self.trainset}.2,{self.trainset}.3_train.csv"
            dataset = pd.read_csv(os.path.join(unzip_path, file_name))
            self.relation_list = list(set(dataset["target"]))
            
            step3_set = dataset[dataset['query_edge'] == "(0, 3)"]
            step2_set = dataset[dataset['query_edge'] == "(0, 2)"]

            train_num = int(len(step3_set) * 0.5)
            step3_train = step3_set.iloc[:train_num]
            step3_test = step3_set.iloc[train_num:]

            train_num = int(len(step2_set) * 0.5)
            step2_train = step2_set.iloc[:train_num]
            step2_test = step2_set.iloc[train_num:]
            
            testset = self.shuffle_dataset(step2_test, step3_test)
            self.trainset = self.shuffle_dataset(step2_train, step3_train)
        else:
            file_name = f"1.2,1.3,1.4_train.csv"
            dataset = pd.read_csv(os.path.join(unzip_path, file_name))
            self.relation_list = list(set(dataset["target"]))
            
            step2_set = dataset[dataset['query_edge'] == "(0, 2)"]
            step3_set = dataset[dataset['query_edge'] == "(0, 3)"]
            step4_set = dataset[dataset['query_edge'] == "(0, 4)"]
            
            if self.hop == 3:
                dataset = step3_set
            elif self.hop == 4:
                dataset = step4_set
            elif self.hop == 2:
                dataset = step2_set
            else:
                raise ValueError(f"hop {self.hop} not support")

            train_num = int(len(dataset) * 0.5)
            self.trainset = dataset[:train_num]
            testset = dataset[train_num:]
        
        with open(os.path.join(unzip_path, "example_set.json"), "r") as f:
            self.example = json.load(f)
        return testset
    
    def get_question(self, raw_data):
        question = ""
        if self.reasoning_type != "symbolic":
            story = raw_data["story"]
            question += f"Story:{story}\n"
            # genders = raw_data["genders"]
            # question += f"Genders:{genders}\n"
            query = ast.literal_eval(raw_data["query"])
            head_name = query[0]
            tail_name = query[1]
            # question += f"Given the relationships described in the story and the gender information, please infer what relationship {tail_name} has to {head_name}. Please note, the answer must be one of the following options: {str(self.relation_list)}.\n"
            question += f"Question: Given the relationships described in the story information, please infer { tail_name } is { head_name }'s what. "
        else:
            relation_path = ast.literal_eval(raw_data["edge_types"])
            query = ast.literal_eval(raw_data["query"])
            head_name = query[0]
            tail_name = query[1]
            relation_str = "'s ".join(relation_path)
            question += f"In a family tree, if {tail_name} is {head_name}'s {relation_str}. \nQuestion: {tail_name} is {head_name}'s what? "
        question += "Please reason it step by step, and provide a single word answer describing the relationship, in the format  \"Answer: {{relation}}\"\n"
        return question
    
    def get_label(self, raw_data):
        return raw_data["target"]
        
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
    
    def _search_relation_in_path(self, relation_path, r1, r2):
        search_elements = (r1, r2)
        path_index = -1
        for path_i in range(len(relation_path) - 1):
            if (relation_path[path_i], relation_path[path_i+1]) == search_elements:
                path_index = path_i
                break
        return path_index
        
    def get_random_relation(self, original_relation):
        relation_num = len(self.relation_list)
        while 1:
            random_index = random.randrange(0, relation_num)
            new_relation = self.relation_list[random_index]
            if new_relation != original_relation:
                break
        return new_relation
    
    def create_proof_chain(self, relation_path, proof_chain = None):
        if proof_chain == None:
            proof_chain = []
        if len(relation_path) <= 1:
            return proof_chain
        for index in range(len(relation_path) - 1):
            r1 = relation_path[index]
            r2 = relation_path[index + 1]
            
            if (r1, r2) not in self.relation_dict:
                self.not_support_relation_reason.append((r1, r2))
                continue
            else:
                r_mix = self.relation_dict[(r1, r2)]
                try_path = copy.deepcopy(relation_path)
                del try_path[index:index+2]
                try_path.insert(index, r_mix)
                if self.create_proof_chain(try_path, proof_chain) != None:
                    proof_chain.append([r1, r2, r_mix])
                    return proof_chain
        return None
            
    
    def get_symbolic_relation_reason(self, relation_path, proofs, noise_type = None, noisy_level = 1):
        try_count =0
        # if_replace = False
        if noise_type == None:
            noisy_p = 0
        elif noisy_level == 1:
            noisy_p = 0.2
        elif noisy_level == 2:
            noisy_p = 0.35
        elif noisy_level == 3:
            noisy_p = 0.5
        proof_chain = []
        for proof in reversed(proofs):
            for conclusion, reasons in proof.items():
                proof_chain.append([reasons[0][1], reasons[1][1], conclusion[1]])
        
        reasoning_relation_path = copy.deepcopy(relation_path)
        new_proof_chain = copy.deepcopy(proof_chain)
        if noise_type == "minor_error":
            for i in range(len(proof_chain)):
                relation_mix = new_proof_chain[i][2]
                # whether to replace index i's reasoning
                if random.random() < noisy_p:
                    # if_replace = True
                    while 1: # to make sure the replaced reasoning is proper
                        # new_relation_path = copy.deepcopy(relation_path)
                        new_relation = self.get_random_relation(relation_mix)
                        
                        try_relation_path = copy.deepcopy(reasoning_relation_path)
                        index = self._search_relation_in_path(try_relation_path, new_proof_chain[i][0], new_proof_chain[i][1])
                        del try_relation_path[index:index+2]
                        try_relation_path.insert(index, new_relation)
                        
                        remain_proof_chain = self.create_proof_chain(try_relation_path)
                        
                        if remain_proof_chain != None:
                            reasoning_relation_path = try_relation_path
                            new_proof_chain[i][2] = new_relation
                            reversed(remain_proof_chain)
                            
                            update_proof_chain = new_proof_chain[:i+1] + remain_proof_chain
                            if(len(update_proof_chain) > 2):
                                print("???")
                            
                            new_proof_chain = update_proof_chain
                            self.replace_num += 1
                            break
                        else:
                            try_count += 1
                        if try_count > 1000:
                            self.error_reason_num += 1
                            index = self._search_relation_in_path(reasoning_relation_path, new_proof_chain[i][0], new_proof_chain[i][1])
                            del reasoning_relation_path[index:index+2]
                            reasoning_relation_path.insert(index, relation_mix)
                            break
                            # raise ValueError(f"can't replace {str(reasoning_relation_path)}, {index}")
                else:
                    index = self._search_relation_in_path(reasoning_relation_path, new_proof_chain[i][0], new_proof_chain[i][1])
                    del reasoning_relation_path[index:index+2]
                    reasoning_relation_path.insert(index, relation_mix)
                    
        reasoning_relation_path = copy.deepcopy(relation_path)
        answer = ""
        r_mix = None
        selected_noise_set = set()
        for proof in new_proof_chain:
            r1 = proof[0]
            r2 = proof[1]
            r_mix = proof[2]
            index = self._search_relation_in_path(reasoning_relation_path, r1, r2)
            del reasoning_relation_path[index:index+2]
            reasoning_relation_path.insert(index, r_mix)
            relation_str = ", ".join(reasoning_relation_path)
            answer += f"For {r1}'s {r2}, we have {r1}'s {r2} is {r_mix}. " 
            if noise_type == "irrelevant":
                if random.random() < noisy_p:
                    noise_fact = self.get_random_relation_fact(r_mix)
                    answer += noise_fact
                
            
            answer += f"So the relations path are reduced to {relation_str}. "
            if noise_type == "irrelevant":
                if random.random() < noisy_p:
                    noise_fact = self.get_random_relation_fact(r_mix)
                    answer += noise_fact
        answer += f"Therefore, the answer is {r_mix}. \n"
        answer += f"Answer:{r_mix}\n"  
        return answer
    
    def find_sentence_containing_strings(self, text, name1, name2):
        sentences = text.split('.')
        for sentence in sentences:
            if name1 in sentence and name2 in sentence:
                return sentence
        return None
    
    def get_answer(self, raw_data, if_noise):
        answer = ""
        story = raw_data["story"]
        proofs =  ast.literal_eval(raw_data["proof_state"])
        count = self.noisy_level
        selected_noise_set = set()
        
        if (self.reasoning_type != "symbolic"):
            for proof in reversed(proofs): 
                for conclusion, reasons in proof.items():
                    explaination = "Since "
                    for i, reason in enumerate(reasons):
                        head = reason[0]
                        relation = reason[1]
                        tail = reason[2]
                        sentence  = self.find_sentence_containing_strings(story, head, tail)
                        if sentence:
                            answer += f"Based on the sentence \"{sentence}\", we can infer that "
                        answer += f"{tail} is {head}'s {relation}."
                        if i > 0:
                            explaination += " and "
                        explaination += f"{tail} is {head}'s {relation}"
                            
                        if(if_noise and count>0):
                            noise_fact = self.get_random_fact(relation, selected_noise_set).replace("[name1]", head).replace("[name2]", tail)
                            answer += noise_fact
                            count -= 1
                    head = conclusion[0]
                    relation = conclusion[1]
                    tail = conclusion[2]
                    answer += f"{explaination}, {tail} is {head}'s {relation}. \n"
            answer += f"Answer:{relation_mix}\n"  
        else:
            relation_path = ast.literal_eval(raw_data["edge_types"])
            relation_path_str = ", ".join(relation_path)
            relation_desciption = "'s ".join(relation_path)
            
            query = ast.literal_eval(raw_data["query"])
            head_name = query[0]
            tail_name = query[1]
            relation_mix = None
            answer += f"The relations path are {relation_path_str}, which means {tail_name} is {head_name}'s {relation_desciption}. "
            if if_noise:
                answer += self.get_symbolic_relation_reason(relation_path, proofs, self.noise_type, self.noisy_level)
            else:
                answer += self.get_symbolic_relation_reason(relation_path, proofs)
        return answer
    
    def get_random_demos(self, num):
        assert len(self.trainset) > num
        demos = self.trainset.sample(n=num)
        return demos
    
    
    def get_case(self, raw_data):
        case = dict()
        qustion  = self.get_question(raw_data)
        label = self.get_label(raw_data)
        shots = []
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
                normal_demos = demos.iloc[:n_shots]
                noise_demos = demos.iloc[n_shots:]
                for _, demo in normal_demos.iterrows():
                    shot_q = self.get_question(demo)
                    shot_a = self.get_answer(demo, if_noise=False)
                    shots.append([shot_q, shot_a])
                for _, demo in noise_demos.iterrows():
                    shot_q = self.get_question(demo)
                    shot_a = self.get_answer(demo, if_noise=True)
                    shots.append([shot_q, shot_a])
                if self.prefix_context:
                    for shot in shots:
                        prefix += "user:{}\nassistant:{}\n".format(shot[0], shot[1])
                    prefix += "user:"
                else:    
                    case["in-context"] = shots
        case["question"] = prefix + qustion
        case["label"] = label
        
        return case
    
    def match_answer(self, answer_str):
        match = re.search(r'[Aa]nswer:.*?([A-Za-z\-]+)', answer_str)
        if match:
            return match.group(1).lower()
        else:
            return None
            
            
        
        