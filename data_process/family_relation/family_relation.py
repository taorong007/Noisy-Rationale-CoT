import random
import json
import pandas as pd
import zipfile
import os
import re
import ast
from collections import deque

class family_relation():
    def __init__(self, if_in_context = False, n_shots=0, n_noisy_shots=0, noisy_type="irrelative", noisy_level = 1, prefix_context =False, config: dict = None, reasoning_type = "symbolic", hop = 3, trainset=5, testset = 5.3) -> None:
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
        self.file_path = os.path.join("data", "data_emnlp_final")
        self.unzip_data()
        self.init_noise_data()
        return
    
    def init_noise_data(self):
        # with open(os.path.join(self.file_path, "noise", "noise_relation_facts.json"), "r") as f:
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
        # self.in_context_dataset = pd.read_csv(os.path.join(unzip_path, trainset_file_name))
        
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
            self.in_context_dataset = self.shuffle_dataset(step2_train, step3_train)
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
            self.in_context_dataset = dataset[:train_num]
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
            question += f"Question: Given the relationships described in the story information, please infer { tail_name } is { head_name }'s what."
        else:
            relation_path = ast.literal_eval(raw_data["edge_types"])
            query = ast.literal_eval(raw_data["query"])
            head_name = query[0]
            tail_name = query[1]
            relation_str = ", ".join(relation_path)
            question += f"Context: The relations on the path from {head_name} to {tail_name} are {relation_str}.\n Question: {tail_name} is {head_name}'s what?"
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
                    # answer += "Since "
                    explaination = "Since "
                    for i, reason in enumerate(reasons):
                        head = reason[0]
                        relation = reason[1]
                        tail = reason[2]
                        # if i > 0:
                        #     answer += " and "
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
                    # answer += ","
                    head = conclusion[0]
                    relation = conclusion[1]
                    tail = conclusion[2]
                    answer += f"{explaination}, {tail} is {head}'s {relation}. \n"
        else:
            relation_path = ast.literal_eval(raw_data["edge_types"])
            relation_mix = None
            for proof in reversed(proofs): 
                for conclusion, reasons in proof.items():
                    relation_mix = conclusion[1]
                    head = conclusion[0]
                    tail = conclusion[2]
                    # relation1 = relation_path.popleft()
                    # relation2 = relation_path.popleft()
                    search_elements = (reasons[0][1], reasons[1][1])
                    for i in range(len(relation_path) - 1):
                        if (relation_path[i], relation_path[i+1]) == search_elements or (relation_path[i+1], relation_path[i]) == search_elements:
                            relation1 = relation_path[i]
                            relation2 = relation_path[i+1]
                            del relation_path[i:i+2]
                            relation_path.insert(i, relation_mix)
                            break
                    relation_str = ", ".join(relation_path)
                    answer += f"For {relation1}'s {relation2}, we have {relation1}'s {relation2} is {relation_mix}." 
                    if(if_noise and self.noisy_level > 0):
                        answer += self.get_random_fact(relation_mix, selected_noise_set).replace("[name1]", head).replace("[name2]", tail)
                    answer += f"So the relations are reduced to {relation_str}. "
                    # if(if_noise and self.noisy_level > 0):
                    #     answer += self.get_random_fact(relation_mix, selected_noise_set) + " "
            answer += f"Therefore, the answer is {relation_mix}. "
        answer += f"Answer:{relation_mix}\n"  
        return answer
    
    def get_random_demos(self, num):
        assert len(self.in_context_dataset) > num
        demos = self.in_context_dataset.sample(n=num)
        return demos
    

        
    
    #id	story	query	text_query	target	text_target	clean_story	proof_state	f_comb	task_name	story_edges	edge_types	query_edge	genders	syn_story	node_mapping	task_split	
    #73a26e9c-62c0-489c-aae3-f293e8564ff9	
    #[Ellen] wanted to bring all her siblings together for a family reunion so she called up her brother, [Francisco], and her sister, [Victoria]. [Francisco] is in the sixth grade. He looks up to his sister [Louise], who is in the seventh. [Lisbeth] picked up her son [Francisco] from the mall	
    # ('Victoria', 'Lisbeth')		
    # mother	
    # ['[Lisbeth] took her daughter, [Victoria], to lunch.']	
    # [Ellen] wanted to bring all her siblings together for a family reunion so she called up her brother, [Francisco], and her sister, [Victoria]. [Lisbeth] picked up her son [Francisco] from the mall	
    # [{('Victoria', 'mother', 'Lisbeth'): [('Victoria', 'sister', 'Ellen'), ('Ellen', 'mother', 'Lisbeth')]}, 
    # {('Ellen', 'mother', 'Lisbeth'): [('Ellen', 'brother', 'Francisco'), ('Francisco', 'mother', 'Lisbeth')]}]	
    # sister-brother-mother	
    # task_3.3	
    # [(0, 1), (1, 2), (2, 3), (2, 4)]	
    # ['sister', 'brother', 'mother']	
    # (0, 3)	Victoria:female,Ellen:female,Francisco:male,Lisbeth:female,Louise:female		{19: 0, 18: 1, 20: 2, 16: 3, 17: 4}	train

    
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
            
            
        
        