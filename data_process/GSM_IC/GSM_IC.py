import os
import json
import re
class GSM_IC:
    def __init__(self, n_shots=0) -> None:
        self.dataset_path = "./data/GSM_IC/"
        self._suffix_prompt = "\nEnd the response with the result in \"Answer:\\boxed{{result}}\""
        self.n_shots = 0
        
    def get_case(self, raw_data):
        case = dict()
        original_question = raw_data["original_question"]
        label = raw_data["answer"]
        question = original_question + self._suffix_prompt
        case["question"] = question
        case["label"] = label 
        return case
        # self._case_list.append({"question": question, "label": label})
    def match_answer(self, answer_str):
        match = re.search(r'[Aa]nswer:.*?(-?\d+(\.\d+)?)', answer_str)
        if match:
            answer = match.group(1)
            if int(float(answer)) == float(answer):
                answer = str(int(float(answer)))
        else:
            answer = None
        return answer
    def load_data(self):
        data_file = os.path.join(self.dataset_path, "GSM-IC_mstep.json")
        with open(data_file, "r") as f:
            dataset = json.load(f)
        return dataset
               