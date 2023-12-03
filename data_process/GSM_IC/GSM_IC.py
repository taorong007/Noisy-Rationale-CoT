import os
import json
import re
class GSM_IC:
    def __init__(self, n_shots=0) -> None:
        self.dataset_path = "./data/GSM/"
        self._suffix_prompt = "\nEnd the response with the result in \"answer is : {pure result}\""
        self.n_shots = 0
        
    def get_case(self, raw_data,                         ):
        case = dict()
        original_question = raw_data["query"]
        label = raw_data["answer"]
        question = original_question + self._suffix_prompt
        case["question"] = question
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
        data_file = os.path.join(self.dataset_path, "GSM_AnsAug.json")
        with open(data_file, "r") as f:
            dataset = json.load(f)
        for d in dataset:
            response = d["response"]
            match = re.search(r'[Aa]nswer.*?:.*?(-?\d+(\.\d+)?)', response)
            assert match is not None
            if match:
                answer = match.group(1)
                answer = int(answer)
            else:
                answer = None
            d["answer"] = answer
        return dataset
               