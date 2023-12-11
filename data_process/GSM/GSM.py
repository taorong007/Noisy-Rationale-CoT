import os
import json
import re
import random
import math

class GSM:
    def __init__(self, n_shots=0,  n_noisy_shots=0, noisy_type="irrelative", noisy_level = 1, prefix_context = False) -> None:
        self.dataset_path = "./data/GSM/"
        self._suffix_prompt = "\nEnd the response with the result in \"The answer is : {result}\""
        self.n_shots = n_shots
        self.n_noisy_shots = n_noisy_shots
        self.prefix_context = prefix_context
        noise_file = "./data/base_math/noise/factsOfNumber.json"
        with open(noise_file, encoding="utf-8") as f:
            self.noise_data = json.load(f)["noise_info"]
        self.noisy_type = noisy_type
        self.noisy_level = noisy_level
        
    def get_question(self, raw_data):
        original_question = raw_data["query"]
        question = original_question + self._suffix_prompt
        return question
    
    
    def get_answer(self, raw_data):
        answer = raw_data["response"]
        return answer
    
    def _random_choose_fact(self):
        random_number = random.randrange(0, 19)
        facts = self.noise_data[random_number]["facts"]
        random_index = random.randrange(0, len(facts))
        selected_fact = facts[random_index]
        return selected_fact
            
    def get_irrelative_answer(self, raw_data):
        answer = raw_data["response"]
        sentences = answer.split('\n')
        lenth = len(sentences)
        
        if self.noisy_level == 1:
            noise_num = math.ceil(0.1 * lenth)
        elif self.noisy_level == 2:
            noise_num = math.ceil(0.3 * lenth)  
        elif self.noisy_level == 3:
            noise_num = math.ceil(0.5 * lenth)
        
        
        for index in range(noise_num):
            # noise = self._random_choose_fact()
            # new_sentences.append(sentences[index])
            # new_sentences.append(noise)
            index = random.randint(0, len(sentences))
            noise = self._random_choose_fact()
            sentences.insert(index, noise)
        answer = '\n'.join(sentences)
        return answer

    def get_case(self, raw_data):
        n_shots = self.n_shots
        n_noisy_shots = self.n_noisy_shots
        case = dict()
        
        label = raw_data["label"]
        shots = []
        if n_shots > 0:
            demos = random.sample(self.ICL_set, n_shots)
            for demo in demos:
                question = self.get_question(demo)
                answer = self.get_answer(demo)
                shots.append([question, answer])
        if n_noisy_shots > 0:
            demos = random.sample(self.ICL_set, n_noisy_shots)
            for demo in demos:
                question = self.get_question(demo)
                if self.noisy_type == "irrelative":
                    answer = self.get_irrelative_answer(demo)
                else:
                    raise ValueError(f"error type {self.error_type} not support")
                shots.append([question, answer])
        
        prefix = ""
        if self.prefix_context:
            prefix += "Here is some examples:\n"
            for shot in shots:
                prefix += "User:{}\n\nAssistant:{}\n\n".format(shot[0], shot[1])
            prefix += "Now, please this question according to the examples above: \nUser:"
        else:    
            case["in-context"] = shots
        # print(prefix)
        question = self.get_question(raw_data)
        case["question"] = prefix + question
        case["label"] = label 
        return case
        # self._case_list.append({"question": question, "label": label})
    
    
    def match_answer(self, answer_str):
        match = re.search(r'[Aa]nswer.*?:.*?(-?\d+(\.\d+)?)', answer_str)
        if match:
            answer = match.group(1)
            if int(float(answer)) == float(answer):
                answer = int(float(answer))
            else:
                answer = float(answer)
        else:
            answer = None
        return answer
    
    
    def load_data(self):
        data_file = os.path.join(self.dataset_path, "GSM_SV.json")
        with open(data_file, "r") as f:
            dataset = json.load(f)
        
        for d in dataset:
            response = d["response"]
            match = re.search(r'[Aa]nswer.*?:.*?(-?\d+(\.\d+)?)', response)
            assert match is not None
            if match:
                pure_answer = match.group(1)
                pure_answer = float(pure_answer)
            else:
                pure_answer = None
            d["label"] = pure_answer
            
        self.ICL_set = dataset[-1000:]
        testset = dataset[:-1000]
        return testset
               