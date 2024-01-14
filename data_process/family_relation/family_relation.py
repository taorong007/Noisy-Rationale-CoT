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

class family_relation():
    def __init__(self, if_in_context = False, n_shots=0, n_noisy_shots=0, noise_type="irrelevant", noise_ratio = 0.5, noise_distribution = "fixed", prefix_context =False, config: dict = None, reasoning_type = "symbolic", hop = 3, trainset=5, testset = 5.3) -> None:
        self.if_in_context = if_in_context
        self.n_shots = n_shots
        self.n_noisy_shots = n_noisy_shots
        self.prefix_context = prefix_context
        if self.n_noisy_shots > 0:
            self.noise_type = noise_type
            self.noise_ratio = noise_ratio
            self.noise_distribution = noise_distribution
        else:
            self.noise_type = None
            self.noise_ratio = 0
            self.noise_distribution = None
        if config is not None:
            self.trainset = config["train_set"] if "train_set" in config else trainset
            # self.testset = config["test_set"]
            self.reasoning_type = config["reasoning_type"]
            self.hop = config["hop"] if "hop" in config else hop
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
    
    def _to_dict(self, df_row):
        columns_to_extract = ['edge_types', 'query', "target", "proof_state"]
        shot = {col: df_row[col] for col in columns_to_extract}
        return shot
        
    def _to_list(self, df):
        columns_to_extract = ['edge_types', 'query', "target", "proof_state"]
        shots = df.loc[:, columns_to_extract].to_dict(orient='records')
        return shots
    
    def generate_json(self, dataset, type):
        data_iter = dataset.iterrows()
        processed_path = os.path.join(self.file_path, "processed")
        if not os.path.exists(processed_path):
            os.makedirs(processed_path)
        if self.reasoning_type == "symbolic":
            file_name = f"{self.reasoning_type}_{self.hop}hop"
            if type == 1:
                file_name += "_demos.json"
                raw_data_IC_list = []
                for _, raw_data in data_iter:
                    demos = self.get_random_demos(num=10)
                    raw_data_IC = dict()
                    raw_data_IC["test"] = self._to_dict(raw_data)
                    raw_data_IC["demos"] = self._to_list(demos)
                    raw_data_IC_list.append(raw_data_IC)
                file_path = os.path.join(processed_path, file_name)
                with open(file_path, 'w') as json_file:
                    json.dump(raw_data_IC_list, json_file, indent=4)
            elif type == 2:
                cases = []
                if self.n_shots > 0:
                    file_name += f"_{self.n_shots}clean"
                if self.n_noisy_shots > 0:
                    file_name += f"_{self.n_shots}{self.noise_type}_{self.noise_ratio}_{self.noise_distribution}"
                file_name += ".json"
                file_path = os.path.join(processed_path, file_name)
                for data in dataset:
                    case = self.get_case(data)
                    cases.append(case)
                with open(file_path, 'w') as json_file:
                    json.dumps(cases, json_file, indent=4)
            
    
    def load_data(self):
        unzip_path = os.path.join(self.file_path, "unzip_data", str(self.trainset))
        # trainset_file_name = f"{self.trainset}.2,{self.trainset}.3_train.csv"
        # testset_file_name = f"{self.testset}_test.csv"
        # self.trainset = pd.read_csv(os.path.join(unzip_path, trainset_file_name))
    
        if self.reasoning_type !=  "symbolic":
            file_name = f"{self.trainset}.2,{self.trainset}.3_train.csv"
            raw_dataset = pd.read_csv(os.path.join(unzip_path, file_name))
            self.relation_list = list(set(raw_dataset["target"]))
            
            step3_set = raw_dataset[raw_dataset['query_edge'] == "(0, 3)"]
            step2_set = raw_dataset[raw_dataset['query_edge'] == "(0, 2)"]

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
            raw_dataset = pd.read_csv(os.path.join(unzip_path, file_name))
            self.relation_list = list(set(raw_dataset["target"]))
            
            step2_set = raw_dataset[raw_dataset['query_edge'] == "(0, 2)"]
            step3_set = raw_dataset[raw_dataset['query_edge'] == "(0, 3)"]
            step4_set = raw_dataset[raw_dataset['query_edge'] == "(0, 4)"]
            
            if self.hop == 3:
                dataset = step3_set
            elif self.hop == 4:
                dataset = step4_set
            elif self.hop == 2:
                dataset = step2_set
            else:
                raise ValueError(f"hop {self.hop} not support")
            
            mask = dataset['edge_types'].apply(lambda x: len(ast.literal_eval(x)) == 3)
            dataset = dataset[mask]
            dataset = dataset.sample(frac=1).reset_index(drop=True)
            unnamed_cols = [col for col in dataset.columns if col.startswith('Unnamed')]
            if unnamed_cols:
                dataset = dataset.drop(unnamed_cols, axis=1)
            self.trainset = dataset
            testset = dataset
        
        with open(os.path.join(unzip_path, "example_set.json"), "r") as f:
            self.example = json.load(f)
    
            # self.generate_json(testset, 1)
        
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
        question += "Please reason it step by step, and provide a single word answer describing the relationship. End the response in the format  \"Answer: {{relation}}\"\n"
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
    
    def get_random_relation_fact(self, relation, selected_set):
        facts = self.noise_relation_facts[relation]
        while(1):
            random_index = random.randrange(0, len(facts))
            selected = f"{relation}_{random_index}"
            fact = facts[random_index] + " "
            if selected not in selected_set:
                selected_set.add(selected)
                break
            go_on_random = 0
            for i in range(len(facts)):
                key = f"{relation}_{i}"
                if key not in selected_set:
                    go_on_random = 1
            if go_on_random == 0:
                break
        return fact
    
    def _search_relation_in_path(self, relation_path, r1, r2):
        search_elements = (r1, r2)
        path_index = -1
        for path_i in range(len(relation_path) - 1):
            if (relation_path[path_i], relation_path[path_i+1]) == search_elements:
                path_index = path_i
                break
        return path_index
        
    def get_random_relation(self, original_relation = None):
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
            
    def _prepare_noise_distribution_iteration_state(self, n_thought, noise_ratio, noise_distribution):
        noise_distribution_list = [0] * n_thought
        if noise_distribution == "fixed":
            noise_thoughts = n_thought * noise_ratio
            integer_part = int(noise_thoughts)
            decimal_part = noise_thoughts - integer_part
            if decimal_part == 0.5:
                noise_count = math.ceil(n_thought * noise_ratio)
            else:
                noise_count = round(n_thought * noise_ratio)
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


    # def get_answer(self, raw_data, if_noise):
    #     in_content = raw_data[0]
    #     label = raw_data[1]
        
    #     _, _, n_ir_pos, n_mn_pos = self.get_symbolic_relation_reason(in_content)
        
        
    #     if self.noise_type == "irrelevant":
    #         ir_noise_p = self.noise_ratio
    #         mn_noise_p = 0
    #     else:
    #         ir_noise_p = 0
    #         mn_noise_p = self.noise_ratio
    #     ir_noise_distrib_state = self._prepare_noise_distribution_iteration_state(n_ir_pos, ir_noise_p, self.noise_distribution) 
    #     mn_noise_distrib_state = self._prepare_noise_distribution_iteration_state(n_mn_pos, mn_noise_p, self.noise_distribution)                 
    #     answer,action_sequence, _, _ = self.get_symbolic_relation_reason(in_content, ir_noise_distrib_state, mn_noise_distrib_state)
        
    #     # print(action_sequence)
    #     assert str(action_sequence) == str(label)
    #     return answer
    # reasoning_relation_path = copy.deepcopy(relation_path)
    # if noise_type == "inaccurate":
        #     for i in range(len(proof_chain)):
        #         relation_mix = new_proof_chain[i][2]
        #         # whether to replace index i's reasoning
        #         if random.random() < noisy_p:
        #             # if_replace = True
        #             while 1: # to make sure the replaced reasoning is proper
        #                 # new_relation_path = copy.deepcopy(relation_path)
        #                 new_relation = self.get_random_relation(relation_mix)
                        
        #                 try_relation_path = copy.deepcopy(reasoning_relation_path)
        #                 index = self._search_relation_in_path(try_relation_path, new_proof_chain[i][0], new_proof_chain[i][1])
        #                 del try_relation_path[index:index+2]
        #                 try_relation_path.insert(index, new_relation)
                        
        #                 remain_proof_chain = self.create_proof_chain(try_relation_path)
                        
        #                 if remain_proof_chain != None:
        #                     reasoning_relation_path = try_relation_path
        #                     new_proof_chain[i][2] = new_relation
        #                     reversed(remain_proof_chain)
                            
        #                     update_proof_chain = new_proof_chain[:i+1] + remain_proof_chain
        #                     if(len(update_proof_chain) > 2):
        #                         print("???")
                            
        #                     new_proof_chain = update_proof_chain
        #                     self.replace_num += 1
        #                     break
        #                 else:
        #                     try_count += 1
        #                 if try_count > 1000:
        #                     self.error_reason_num += 1
        #                     index = self._search_relation_in_path(reasoning_relation_path, new_proof_chain[i][0], new_proof_chain[i][1])
        #                     del reasoning_relation_path[index:index+2]
        #                     reasoning_relation_path.insert(index, relation_mix)
        #                     break
        #                     # raise ValueError(f"can't replace {str(reasoning_relation_path)}, {index}")
        #         else:
                    # index = self._search_relation_in_path(reasoning_relation_path, new_proof_chain[i][0], new_proof_chain[i][1])
                    # del reasoning_relation_path[index:index+2]
                    # reasoning_relation_path.insert(index, relation_mix)
                    

    def get_random_inaccurate_thought(self, r1):
        r2 = self.get_random_relation(r1)
        
        if (r1, r2) not in self.relation_dict:
            err_mix = self.get_random_relation(None)
        else:
            r_mix = self.relation_dict[(r1, r2)]
            err_mix =  self.get_random_relation(r_mix)
        inaccurate_thought = f"We have {r1}'s {r2} is {err_mix}. "
        return inaccurate_thought
    
    def get_symbolic_relation_reason(self, raw_data, ir_noise_distrib_state=None, mn_noise_distrib_state=None):
        proofs =  ast.literal_eval(raw_data["proof_state"])
        relation_path = ast.literal_eval(raw_data["edge_types"])
        relation_path_str = ", ".join(relation_path)
        relation_desciption = "'s ".join(relation_path)
        
        query = ast.literal_eval(raw_data["query"])
        head_name = query[0]
        tail_name = query[1]
        
        n_ir_pos = 0
        n_mn_pos = 0
        selected_noise_set = set()
        
        answer = ""
        answer += f"{tail_name} is {head_name}'s {relation_desciption}, so the relations path is {relation_path_str}. "
        n_mn_pos+=1
        
        if self._should_add_noise(mn_noise_distrib_state):
            noise_fact = self.get_random_inaccurate_thought(relation_path[-1])
            answer += noise_fact
            
        n_ir_pos += 1
        if self._should_add_noise(ir_noise_distrib_state):
            noise_fact = self.get_random_relation_fact("family relation", selected_noise_set)
            answer += noise_fact
        
        # if_replace = False
        noise_type = self.noise_type      
        proof_chain = []
        for proof in reversed(proofs):
            for conclusion, reasons in proof.items():
                proof_chain.append([reasons[0][1], reasons[1][1], conclusion[1]])
        
        
        new_proof_chain = copy.deepcopy(proof_chain)
        
        reasoning_relation_path = copy.deepcopy(relation_path)
        r_mix = None
        for proof in new_proof_chain:
            r1 = proof[0]
            r2 = proof[1]
            r_mix = proof[2]
            index = self._search_relation_in_path(reasoning_relation_path, r1, r2)
            del reasoning_relation_path[index:index+2]
            reasoning_relation_path.insert(index, r_mix)
            
            
            # if not self._should_add_noise(mn_noise_distrib_state):
            
            answer += f"For {r1}'s {r2}, we have {r1}'s {r2} is {r_mix}. "
            # else:
            #     answer += f"For {r1}'s {r2}, we have {r1}'s {r2} is {self.get_random_relation(r_mix)}. "
            n_mn_pos+=1
            if self._should_add_noise(mn_noise_distrib_state):
                noise_fact = self.get_random_inaccurate_thought(r2)
                answer += noise_fact
            
            n_ir_pos += 1
            if self._should_add_noise(ir_noise_distrib_state):
                noise_fact = self.get_random_relation_fact(r2, selected_noise_set)
                answer += noise_fact
            
            relation_str = ", ".join(reasoning_relation_path)
            answer += f"So the relations path are reduced to {relation_str}. "
            
            n_mn_pos+=1
            if self._should_add_noise(mn_noise_distrib_state):
                noise_fact = self.get_random_inaccurate_thought(r_mix)
                answer += noise_fact
                    
            n_ir_pos += 1
            if self._should_add_noise(ir_noise_distrib_state):
                noise_fact = self.get_random_relation_fact(r_mix, selected_noise_set)
                answer += noise_fact
                
        answer += f"Therefore, Answer: {r_mix}. \n"
        return answer, n_ir_pos, n_mn_pos
    
    def find_sentence_containing_strings(self, text, name1, name2):
        sentences = text.split('.')
        for sentence in sentences:
            if name1 in sentence and name2 in sentence:
                return sentence
        return None
    
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
    
    def get_answer(self, raw_data, generate_info = None):
        answer = ""
        # story = raw_data["story"]
        # proofs =  ast.literal_eval(raw_data["proof_state"])
        # selected_noise_set = set()
        
        # if (self.reasoning_type != "symbolic"):
        #     for proof in reversed(proofs): 
        #         for conclusion, reasons in proof.items():
        #             explaination = "Since "
        #             for i, reason in enumerate(reasons):
        #                 head = reason[0]
        #                 relation = reason[1]
        #                 tail = reason[2]
        #                 sentence  = self.find_sentence_containing_strings(story, head, tail)
        #                 if sentence:
        #                     answer += f"Based on the sentence \"{sentence}\", we can infer that "
        #                 answer += f"{tail} is {head}'s {relation}."
        #                 if i > 0:
        #                     explaination += " and "
        #                 explaination += f"{tail} is {head}'s {relation}"
                            
        #                 if(if_noise and count>0):
        #                     noise_fact = self.get_random_fact(relation, selected_noise_set).replace("[name1]", head).replace("[name2]", tail)
        #                     answer += noise_fact
        #                     count -= 1
        #             head = conclusion[0]
        #             relation = conclusion[1]
        #             tail = conclusion[2]
        #             answer += f"{explaination}, {tail} is {head}'s {relation}. \n"
        #     answer += f"Answer:{relation_mix}\n"  
        # else:
        _, n_ir_pos, n_mn_pos = self.get_symbolic_relation_reason(raw_data)
        if self.noise_type == "irrelevant":
            ir_noise_p = self.noise_ratio
            mn_noise_p = 0
        else:
            ir_noise_p = 0
            mn_noise_p = self.noise_ratio
        ir_noise_distrib_state = self._prepare_noise_distribution_iteration_state(n_ir_pos, ir_noise_p, self.noise_distribution) 
        mn_noise_distrib_state = self._prepare_noise_distribution_iteration_state(n_mn_pos, mn_noise_p, self.noise_distribution)
        if generate_info is not None:          
            if(self.noise_type == "irrelevant"):
                self.get_generation_config(ir_noise_distrib_state, generate_info)
            elif(self.noise_type == "inaccurate"):
                self.get_generation_config(mn_noise_distrib_state, generate_info)
            else:
                self.get_generation_config(ir_noise_distrib_state, generate_info)                 
        answer, _, _ = self.get_symbolic_relation_reason(raw_data, ir_noise_distrib_state, mn_noise_distrib_state)
        return answer
    
    def get_random_demos(self, num, expr=None, index_list = None):
        if expr is not None:
            assert len(self.trainset) > num - 1
            expr_edge_types = expr["edge_types"]
            mask = self.trainset["edge_types"] == expr_edge_types
            trainset = self.trainset[~mask]
            demos = trainset.sample(n=num)
        else:
            demos = self.trainset.sample(n=num)
        if index_list is not None:
            index_list.extend(demos.index.tolist())
        return demos
    
    def get_demos_by_index_list(self, num, index_list):
        # demos = []
        # for i in range(num):
        #     index = index_list[i]
        #     demos.append(self.trainset[index])
        demos = self.trainset.loc[index_list[:num]]
        return demos
    
    def get_case(self, raw_data, if_generate_info=None, ICL_index_list=None):
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
                if ICL_index_list is None:
                    demos = self.get_random_demos(num=n_total_shots, expr=raw_data)
                else:
                    demos = self.get_demos_by_index_list(num=n_total_shots, index_list=ICL_index_list)
                # noise_demos = demos.iloc[n_shots:]
                for _, demo in demos.iterrows():
                    if if_generate_info:
                        generate_info = dict()
                    else:
                        generate_info = None
                    shot_q = self.get_question(demo)
                    shot_a = self.get_answer(demo, generate_info)
                    if if_generate_info:
                        shots.append([shot_q, shot_a, generate_info])
                    else:
                        shots.append([shot_q, shot_a])
                # for _, demo in noise_demos.iterrows():
                #     shot_q = self.get_question(demo)
                #     shot_a = self.get_answer(demo)
                #     shots.append([shot_q, shot_a])
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
            
            
        
        