import yaml
import os
from typing import List, Optional
# import fire
# from my_llama import my_llama
import json
import re
import pickle
from llm_model.my_gpt import my_gpt
import data_process.base_math.base_math as base_math
import time

    
with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)

def convert_to_number(s):
    try:
        return float(s)
    except ValueError:
        return s
    
def wr_log(obj, log_file):
    print(obj)
    log_file.write(str(obj)+"\n")
    log_file.flush()

    
class noise_test:
    def _init_model(self):
        
        if self._model_name == "llama2":
            self._model = my_llama()
        elif self._model_name.split("-")[0] == "GPT":
            model_config =  config["GPT"] if "GPT" in config else None
            self._model = my_gpt(config=model_config)
        else:
            raise ValueError("Unsupported model {}".format(self._model_name))
    
    def _init_dataset(self):
        processor_config =  config[self._dataset_name] if self._dataset_name in config else None
        
        if self._dataset_name == "base_math":
            self._dataset_processor = base_math.base_math(if_COT = self._if_COT, n_shots= self._n_shots, ex_shots = self._ex_shots, error_shots = self._error_shots, prefix_context = self._prefix_context, config = processor_config)
            self._dataset = self._dataset_processor.load_data()
        else:
            raise ValueError("Unsupported dataset {}".format(self._dataset_name))
    
    def _get_log_file_name(self):
        log_path = os.path.join("result", self._dataset_name, self._model_name)
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        log_file = "log"
        if self._if_COT:
            if self._prefix_context:
                log_file = log_file +"_prefix"
            log_file = log_file + "_COT_{}_{}".format(self._n_shots, self._ex_shots)
        if self._if_noise:
            log_file = log_file + "_noise_{}{}".format(self._error_shots, self._error_type)
        else:
            log_file = log_file + "_origin"
        log_file = log_file + ".log"
        
        log_file_path = os.path.join(log_path, log_file)
        return log_file_path
    
    def __init__(self, args = config) -> None:
        self._model_name = args["model"]
        self._dataset_name = args["dataset"]
        self._start_num = args["start_num"]
        self._test_num = args["test_num"]
        self._run_time = args["run_time"]
        self._batch_size = args["batch_size"]
        assert  self._run_time * self._test_num / self._batch_size == int(self._run_time * self._test_num / self._batch_size), "run_time * test_num / batch_size should be a positive integer"
        self._if_COT = args["if_COT"] if "if_COT" in config else False
        if self._if_COT:
            self._if_noise = args["if_noise"] if "if_noise" in config else False
            self._n_shots = args["n_shots"] if "if_noise" in config else 1
            self._ex_shots = args["ex_shots"] if "if_noise" in config else self._n_shots
        else:
            self._if_noise =False
            self._n_shots = 0
            self._ex_shots = 0
        
        if self._if_noise:
            self._error_shots = config["error_shots"] if "error_shots" in config else 0
            if self._error_shots == 0:
                self._if_noise = False
            else:
                self._error_type = config["error_type"] if "error_type" in config else "miscalculation"
            
        
        self._prefix_context = config["prefix_context"] if "prefix_context" in config else False
        
        log_name = args["log_name"] if "log_name" in config else self._get_log_file_name()
        self._log_file = open(log_name, 'w')
        self._pickle_name = log_name.split('.')[0] + '.pkl'
        self._log(config)
        
        
        self._init_model()
        self._init_dataset()
        
        self._correct_num = 0
        self._error_num = 0
        self._not_match_num = 0
        self._case_list = []
        self._answers_list = []
        self._contents_list = []
        self._noise_test_result = None
        
        # self._suffix_prompt = "Please generate the answer and put it into the answer box in the following format: 'Answer: \\boxed{pure number}.'"
        return
    
    def run(self):
        if self._noise_test_result == None:
            count = 0
            test_num = self._test_num
            for raw_data in self._dataset:
                if count < self._start_num:
                    count += 1
                    continue
                for i in range(int(self._run_time)):
                    self._question_insert(raw_data)       
                test_num -= 1
                if(test_num <= 0):
                    break
                count += 1
            self._query_process()
            self._noise_test_result = [self._correct_num, self._error_num, self._answers_list, self._contents_list]
            self._save_result()
            self._log("correct_num:{}, error_num:{}, correate_rate:{}".format(self._correct_num, self._error_num, self._correct_num/(self._correct_num+self._error_num)))
        return self._noise_test_result
    
    def _log(self, obj):
        wr_log(obj, self._log_file)
    
    def _response_process(self, responses_batch, label_batch):
        for response, label in zip(responses_batch, label_batch):
            self._log(json.dumps(response))
            self._log("\ncorrect answer is {}\n".format(label))
            answer = response[-1]["content"]
            self._contents_list.append(answer)
            self._log(answer)
            match = re.search(r'[Aa]nswer:.*?(-?\d+(\.\d+)?)', answer)
            if match:
                result = convert_to_number(match.group(1))
                if (result == float(label)):
                    self._log("right")
                    self._correct_num += 1
                    self._answers_list.append([result, 1])
                else:
                    self._log("wrong")
                    self._error_num += 1
                    self._answers_list.append([result, 0])
            else:
                self._log("not match")
                self._not_match_num += 1
                self._answers_list.append("not match")
        return
    
    def _query_process(self):
        batch_size = self._batch_size
        run_time = self._run_time
        case_list = [self._case_list[i:i+batch_size] for i in range(0, len(self._case_list), batch_size)]
        for case_batch in case_list:
            responses = []
            labels = []
            for case in case_batch:
                while(1):
                    retval, response = self._model.query_case(case)
                    # self._log(retval)
                    if retval[0]:
                        break
                    time.sleep(1)
                responses.append(response)
                labels.append(case["label"])
            self._response_process(responses, labels)
        self._answers_list = [self._answers_list[i:i+run_time] for i in range(0, len(self._answers_list), run_time)]
        self._contents_list = [self._contents_list[i:i+run_time] for i in range(0, len(self._contents_list), run_time)]
            
            
    def _question_insert(self, case):
        # original_question = case["original_question"]
        # label = case["answer"]
        # new_qustion = case["new_question"]
    
        # if not self._if_noise:
        #     question = original_question + self._suffix_prompt
        # else:
        #     question = new_qustion + self._suffix_prompt
        # self._case_list.append({"question": question, "label": label})
        processed_case = self._dataset_processor.get_prompt_case(case)
        self._case_list.append(processed_case)        
    def _save_result(self):
        with open(self._pickle_name, 'wb') as f:
            pickle.dump(self._noise_test_result, f)
        
    
def COT_SC_correct_rate(answers_list):
    from collections  import Counter
    valid_count = 0
    SC_right_count = 0
    for answers in answers_list:
        answers = [sublist for sublist in answers if isinstance(sublist, list)] # clean answers
        if(len(answers) == 0):
            continue
        else:
            valid_count+=1
            
        second_elements_are_1 = [sublist[1] == 1 for sublist in answers]
        any_second_element_is_1 = any(second_elements_are_1)
        if not any_second_element_is_1:
            continue
        true_answer = next((sublist[0] for sublist in answers if sublist[1] == 1), None)
        counter = Counter(sublist[0] for sublist in answers)
        guess_value, _ = counter.most_common(1)[0]
        if guess_value == true_answer:
            SC_right_count += 1
        
        # if(valid_count <= 10):
        #     print("answers: {}, guess:{}, Correct:{}".format(answers, guess_value, true_answer))
    print("SC_correct_num{}, vaild_num{}, SC_correct_rate{}".format(SC_right_count, valid_count, SC_right_count/valid_count))
    return SC_right_count, valid_count
    
    
if __name__ == "__main__":
    in_context = [
        {
            "role": "user", 
            "content" : "There are one hundred tickets to be sold for a volleyball game. Andrea sold twice as many tickets as Jude while Sandra sold 4 more than half the number of tickets Jude sold. The shoe size of Ada is 5000. If Jude sold 16 tickets, how many tickets need to be sold? Please generate the answer and put it into the answer box in the following format: 'Answer: \\boxed{pure number}.'."
        },
        {
            "role": "assistant",
            "content": "Andrea sold 16 x 2 = 32 tickets.\n Half the number of Jude's tickets is 16/2 = 8 tickets.\n So, Sandra sold 8 + 4 = 12 tickets.\nTherefore, a total of 16 + 32 + 12 = 60 tickets were sold.\n Thus, 100 - 60 = 40 tickets need to be sold. Answer: \\boxed{40}."
        }
    ]
    test = noise_test()
    [correct_num, error_num, answer_list, answer_cotents] = test.run()
    
    # with open('log_origin_COT.pkl', 'rb') as f:
    #     lists = pickle.load(f)
    
    # [correct_num, error_num, answer_list, answer_cotents]  = lists
    
    COT_SC_correct_rate(answer_list)
    
    