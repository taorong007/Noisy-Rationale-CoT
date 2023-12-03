import yaml
import os
from typing import List, Optional
# import fire
import json
import re
import pickle
import data_process.base_math.base_math as base_math
import data_process.family_relation.family_relation as family_relation
import data_process.GSM.GSM as GSM
import pandas as pd
import nltk
import random
import time
from datetime import datetime
import copy
import string

    
with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)

    
def wr_log(obj, log_file):
    print(obj)
    log_file.write(str(obj)+"\n")
    log_file.flush()

    
class noise_test:
    def __init__(self, args = config) -> None:
        self._model_name = args["model"]
        self._dataset_name = args["dataset"]
        self._start_num = args["start_num"]
        self._test_num = args["test_num"]
        self._run_times = args["run_times"]
        self._batch_size = args["batch_size"]
        assert  self._run_times * self._test_num / self._batch_size == int(self._run_times * self._test_num / self._batch_size), "run_times * test_num / batch_size should be a positive integer"
        self._if_in_context = args["if_in_context"] if "if_in_context" in config else False
        if self._if_in_context:
            self._if_noise = args["if_noise"] if "if_noise" in config else False
            self._n_shots = args["n_shots"] if "n_shots" in config else 1
            self._n_weak_shots = args["n_weak_shots"] if "n_weak_shots" in config else 0
        else:
            self._if_noise =False
            self._n_shots = 0
            self._n_weak_shots = 0
        
        if self._if_noise:
            self._n_noisy_shots = config["n_noisy_shots"] if "n_noisy_shots" in config else 0
            if self._n_noisy_shots  == 0:
                self._if_noise = False
                self._noisy_type = None
                self._noisy_level = 0
            else:
                self._noisy_type = config["noisy_type"] if "noisy_type" in config else "miscalculation"
                self._noisy_level = int(config["noisy_level"]) if "noisy_level" in config else 1
        else:
            self._n_noisy_shots = 0 
            self._noisy_type = None
            self._noisy_level = 0
        
        self._prefix_context = config["prefix_context"] if "prefix_context" in config else False
        random.seed(time.time())
        log_name = args["log_name"] if "log_name" in config else self._get_log_file_name()
        self._log_file = open(log_name, 'w',  encoding='utf-8')
        dirname = os.path.dirname(log_name)
        basename = os.path.basename(log_name)
        name_without_ext = os.path.splitext(basename)[0]
        self._pickle_name = os.path.join(dirname, name_without_ext + '.pkl')
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
    
    def _init_model(self):
        if self._model_name == "llama2":
            from llm_model.llama.my_llama import my_llama
            model_config =  config["llama2"] if "llama2" in config else None
            self._model = my_llama(config=model_config)
        elif self._model_name.split("-")[0] == "gpt":
            from llm_model.my_gpt import my_gpt
            model_config =  config["gpt"] if "gpt" in config else None
            self._model = my_gpt(model=self._model_name, config=model_config)
        else:
            raise ValueError("Unsupported model {}".format(self._model_name))
    
    def _init_dataset(self):
        processor_config =  config[self._dataset_name] if self._dataset_name in config else None
        if self._dataset_name == "base_math":
            self._dataset_processor = base_math.base_math(if_in_context = self._if_in_context, n_shots= self._n_shots, n_weak_shots = self._n_weak_shots, n_noisy_shots = self._n_noisy_shots, noisy_type=self._noisy_type,  noisy_level=self._noisy_level, prefix_context = self._prefix_context, config = processor_config)
        elif self._dataset_name == "family_relation":
            self._dataset_processor = family_relation.family_relation(if_in_context = self._if_in_context, n_shots= self._n_shots, n_noisy_shots = self._n_noisy_shots, noisy_type=self._noisy_type,  noisy_level=self._noisy_level, config = processor_config)
        elif self._dataset_name == "GSM":
            self._dataset_processor = GSM.GSM(n_shots=self._n_shots, prefix_context=self._prefix_context)
        else:
            raise ValueError("Unsupported dataset {}".format(self._dataset_name))
        self._dataset = self._dataset_processor.load_data()
        assert len(self._dataset) >= self._test_num
    
    def _get_log_file_name(self):
        log_path = os.path.join("result", self._dataset_name, self._model_name)
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        log_file = "log"
        if self._if_in_context:
            if self._prefix_context:
                log_file +="_prefix"
            log_file += "_ICL_{}".format(self._n_shots)
            if self._n_weak_shots > 0:
                log_file += "_{}weak".format(self._n_weak_shots)
        if self._if_noise:
            log_file += "_noise_{}{}_level{}".format(self._n_noisy_shots, self._noisy_type, self._noisy_level)
        else:
            log_file += "_origin"
        log_file += ".log"
        
        log_file_path = os.path.join(log_path, log_file)
        return log_file_path
    
    def run(self):
        self._log("Start time: {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        if self._noise_test_result == None:
            test_num = self._test_num
            if isinstance(self._dataset, pd.DataFrame):
                data_iter = self._dataset.iterrows()
            else:
                data_iter = enumerate(self._dataset)
            for count, raw_data in data_iter:
                if count < self._start_num:
                    continue
                
                self._question_insert(raw_data)       
                test_num -= 1
                if(test_num <= 0):
                    break
            self._query_process()
            self._noise_test_result = [self._correct_num, self._error_num, self._answers_list, self._contents_list]
            self._save_result()
            self._log("correct_num:{}, error_num:{}, correate_rate:{}".format(self._correct_num, self._error_num, self._correct_num/(self._correct_num+self._error_num)))
        self._log("End time: {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        return self._noise_test_result
    
    def _log(self, obj):
        wr_log(obj, self._log_file)
    
    def _random_mask_sentences(self, text, percent):
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        sentences = nltk.sent_tokenize(text)
        num_to_replace = int(len(sentences) * percent)
        # print(num_to_replace)
        to_replace = random.sample(range(len(sentences)), num_to_replace)
        for i in to_replace:
            sentences[i] = ""
        return " ".join(sentences)
    
    def _random_mask_words(self, text, percent, mask = "xxxx"):
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        words  = nltk.word_tokenize(text)
        non_punctuation_words = [i for i, word in enumerate(words) if word not in string.punctuation]
        num_mask = int(len(non_punctuation_words) * percent)
        # print(num_to_replace)
        to_mask = random.sample(non_punctuation_words, num_mask)
        for i in to_mask:
            words[i] = mask
        return " ".join(words)
    
    def _response_process(self, case_batch):
        for case in case_batch:
            messages = case["messages"]
            label = case["label"]
            self._log(json.dumps(messages))
            self._log("\ncorrect answer is {}\n".format(label))
            raw_answer = messages[-1]["content"]
            self._contents_list.append(raw_answer)
            self._log(raw_answer)
            
            answer = self._dataset_processor.match_answer(raw_answer)
            if answer:
                if (answer == label):
                    self._log("right")
                    self._correct_num += 1
                    self._answers_list.append([answer, 1])
                else:
                    self._log("wrong")
                    self._error_num += 1
                    self._answers_list.append([answer, 0])
            else:
                self._log("not match")
                self._not_match_num += 1
                self._answers_list.append("not match")
        return
    
    def _query_process(self):
        batch_size = self._batch_size
        run_times = self._run_times
        case_list = [copy.deepcopy(self._case_list[i:i+batch_size]) for i in range(0, len(self._case_list), batch_size)]
        for index, case_batch in enumerate(case_list):
            # self._method6_contrastive_highlight_noise(case_batch)
            self._model.query_batch(case_batch)
            self._response_process(case_batch)
            self._log(f"index {index}/{len(case_list) - 1}, correct_num {self._correct_num}, error_num {self._error_num}, accuracy {self._correct_num/(self._correct_num+self._error_num)}")
            # COT_SC_correct_rate(self._answers_list)
        self._answers_list = [self._answers_list[i:i+run_times] for i in range(0, len(self._answers_list), run_times)]
        self._contents_list = [self._contents_list[i:i+run_times] for i in range(0, len(self._contents_list), run_times)]
            
    def _question_insert(self, raw_data):
        processed_case = self._dataset_processor.get_case(raw_data)
        for i in range(self._run_times):
            case = copy.deepcopy(processed_case)
            self._case_list.append(case)        
        
    def _save_result(self):
        with open(self._pickle_name, 'wb') as f:
            pickle.dump(self._noise_test_result, f)
        
    def _method6_contrastive_highlight_noise(self, case_batch):
        expr = "47+58"
        for case in case_batch:
            contrastive_query = []
            in_context = case["in-context"]
            for shot in in_context:
                contrastive_case = dict()
                contrastive_question = "The following are two examples for base-9 questions, there is a good example and a bad example, can you refer to the good example and correct steps in the bad example and provided the correct version of assistant's response for me. The filtered version should have the reasoning process similar to good example."
                # contrastive_question = "The following are two examples for base-9 questions, there is a good example and a bad example, can you highlight the important and correct steps in the bad example and provided the modified version for me? "
                contrastive_question += "Good Example:\nQ:"
                contrastive_question += self._dataset_processor.get_question(expr)
                contrastive_question += "\nA:"
                contrastive_question += self._dataset_processor.answer(expr)
                contrastive_question += "\n"
                contrastive_question += "Bad Example:\nQ:"
                contrastive_question += shot[0]
                contrastive_question += "\nA:"
                contrastive_question += shot[1]
                contrastive_question += "\n You should end with format of \"correct version is:{the correct version of assistant reasoning content in bad example}\""
                contrastive_case["question"] = contrastive_question
                contrastive_query.append(contrastive_case)
            self._model.query_batch(contrastive_query)
            for shot, query in zip(in_context, contrastive_query):
                response = query["messages"][-1]["content"]
                match = re.search(r'[Cc]orrect [Vv]ersion.*?:([\s\S]*)', response)
                if match:
                    answer = match.group(1)
                    shot[1] = answer
                    # case["in-context"] = [[contrastive_case["question"], contrastive_case["messages"][-1]["content"]]] + case["in-context"]
                else:
                    # raise ValueError("not match")
                    self._log("not match")
            
        return  
    
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
        
    print("SC_correct_num{}, vaild_num{}, SC_correct_rate{}".format(SC_right_count, valid_count, SC_right_count/valid_count))
    return SC_right_count, valid_count
    
    
if __name__ == "__main__":
    test = noise_test()
    [correct_num, error_num, answer_list, answer_cotents] = test.run()
    
    # with open('log_origin_COT.pkl', 'rb') as f:
    #     lists = pickle.load(f)
    
    # [correct_num, error_num, answer_list, answer_cotents]  = lists
    
    COT_SC_correct_rate(answer_list)
    
    