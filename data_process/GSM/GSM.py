import os
import json
import re
import random

class GSM:
    def __init__(self, n_shots=0,  n_noisy_shots=0, noisy_type="miscalculation", noisy_level = 1, prefix_context = False) -> None:
        self.dataset_path = "./data/GSM/"
        self._suffix_prompt = "\nEnd the response with the result in \"The answer is : {result}\""
        self.n_shots = n_shots
        self.n_noisy_shots = n_noisy_shots
        self.prefix_context = prefix_context
        
    def get_question(self, raw_data):
        original_question = raw_data["query"]
        question = original_question + self._suffix_prompt
        return question
    
    
    def get_answer(self, raw_data):
        answer = raw_data["response"]
        return answer
        
    def get_noisy_answer(self, raw_data):
        answer = raw_data["response"]
        
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
        if n_noisy_shots >= 0:
            demos = random.sample(self.ICL_set, n_shots)
            for demo in demos:
                question = self.get_question(demo)
                answer = self.get_answer(demo)
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
               