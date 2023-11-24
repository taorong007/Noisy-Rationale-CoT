import random
import json
import pandas as pd
import zipfile
import os
import re
import ast

class family_relation():
    def __init__(self, if_in_context = False, n_shots=0, n_noisy_shots=0, noisy_type="irrelative", noisy_level = 1, config: dict = None, trainset=3, testset = 3.3) -> None:
        self.if_in_context = if_in_context
        self.n_shots = n_shots
        self.n_noisy_shots = n_noisy_shots
        if self.n_noisy_shots > 0:
            self.noisy_type = noisy_type
            self.noisy_level = noisy_level
        else:
            self.noisy_type = None
            self.noisy_level = 0
        if config is not None:
            self.trainset = config["train_set"]
            # self.testset = config["test_set"]
        else:
            self.trainset = trainset
            self.testset = testset
        self.file_path = os.path.join("data", "data_emnlp_final")
        self.unzip_data()
        self.init_noise_data()
        return
    
    def init_noise_data(self):
        with open(os.path.join(self.file_path, "noise", "noise_facts.json"), "r") as f:
            noise_facts = json.load(f)
        self.noise_facts = dict()
        for relation_facts in noise_facts:
            relation = relation_facts["relation"]
            facts = relation_facts["facts"] 
            self.noise_facts[relation] = facts
    
    def unzip_data(self):
        zip_files = ["data_7c5b0e70.zip", "data_06b8f2a1.zip", "data_523348e6.zip", "data_d83ecc3e.zip"]
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
        return testset
    
    def get_question(self, raw_data):
        question = ""
        story = raw_data["story"]
        question += f"story:{story}\n"
        genders = raw_data["genders"]
        question += f"genders:{genders}\n"
        query = raw_data["query"]
        query = query[1:-1]
        names = query.split(', ')
        head_name = names[0][1:-1]
        tail_name = names[1][1:-1]
        question += f"Given the relationships described in the story and the gender information, please infer what relationship {tail_name} has to {head_name}. Please note, the answer must be one of the following options: {str(self.relation_list)}.\n"
        question += "You must end the answer in format \"Answer: relation\"\n"
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
        answer += f"Answer:{relation}"  
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
        if(self.if_in_context):
            n_shots = self.n_shots
            n_noisy_shots = self.n_noisy_shots
            n_total_shots = n_shots + n_noisy_shots
            if n_total_shots:
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
        case["question"] = qustion
        case["label"] = label
        case["in-context"] = shots
        return case
    
    def match_answer(self, answer_str):
        match = re.search(r'[Aa]nswer:.*?([A-Za-z\-]+)', answer_str)
        if match:
            return match.group(1).lower()
        else:
            return None
            
            
        
        