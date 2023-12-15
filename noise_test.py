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
import data_process.SCAN.scan_master as scan_master
import pandas as pd
import nltk
import random
import time
from datetime import datetime
import copy
import string
import argparse


def wr_log(obj, log_file):
    print(obj)
    log_file.write(str(obj) + "\n")
    log_file.flush()


class noise_test:
    def __init__(self, args) -> None:
        self._model_name = args["model"]
        self._dataset_name = args["dataset"]
        self._start_num = args["start_num"]
        self._test_num = args["test_num"]
        # self._run_times = args["run_times"]
        self._batch_size = args["batch_size"]
        self.temperature_reason = args["temperature_reason"]
        self.n_reason = args["n_reason"]
        

        assert self._test_num / self._batch_size == int(
            self._test_num / self._batch_size), "test_num / batch_size should be a positive integer"
        
        self._if_in_context = args["if_in_context"] if "if_in_context" in args else False
        if self._if_in_context:
            self._if_noise = args["if_noise"] if "if_noise" in args else False
            self._n_shots = args["n_shots"] if "n_shots" in args else 1
            self._n_weak_shots = args["n_weak_shots"] if "n_weak_shots" in args else 0
            self.if_rephrase = args["if_rephrase"]
        else:
            self._if_noise = False
            self._n_shots = 0
            self._n_weak_shots = 0
            self.if_rephrase = False
            
        if self.if_rephrase:
            self.rephrase_aggregate = args["rephrase_aggregate"]
            self.temperature_rephrase = args["temperature_rephrase"]
            if self.rephrase_aggregate:
                self.n_rephrase = args["n_rephrase"]
            else:
                self.n_rephrase = self.n_reason

        if self._if_noise:
            self._n_noisy_shots = args["n_noisy_shots"] if "n_noisy_shots" in args else 0
            if self._n_noisy_shots  == 0:
                self._if_noise = False
                self._noisy_type = None
                self._noisy_level = 0
            else:
                self._noisy_type = args["noisy_type"] if "noisy_type" in args else "miscalculation"
                self._noisy_level = int(args["noisy_level"]) if "noisy_level" in args else 1
        else:
            self._n_noisy_shots = 0
            self._noisy_type = None
            self._noisy_level = 0
        
        self._prefix_context = args["prefix_context"] if "prefix_context" in args else False
        random.seed(time.time())

        self._init_model()
        self._init_dataset()
        
        log_name = args["log_name"] if "log_name" in args else self._get_log_file_name()
        self._log_file = open(log_name, 'w',  encoding='utf-8')
        dirname = os.path.dirname(log_name)
        basename = os.path.basename(log_name)
        name_without_ext = os.path.splitext(basename)[0]
        self._pickle_name = os.path.join(dirname, name_without_ext + '.pkl')
        self._log(args)
        
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
            model_config = config["llama2"] if "llama2" in config else None
            self._model = my_llama(config=model_config)
        elif self._model_name.split("-")[0] == "gpt":
            from llm_model.my_gpt import my_gpt
            model_config = config["gpt"] if "gpt" in config else None
            self._model = my_gpt(model=self._model_name, config=model_config)
        else:
            raise ValueError("Unsupported model {}".format(self._model_name))
        # self._model_config = self._model.get_config()

    def _init_dataset(self):
        processor_config = config[self._dataset_name] if self._dataset_name in config else None
        if self._dataset_name == "base_math":
            self._dataset_processor = base_math.base_math(if_in_context=self._if_in_context, n_shots=self._n_shots,
                                                          n_weak_shots=self._n_weak_shots,
                                                          n_noisy_shots=self._n_noisy_shots,
                                                          noisy_type=self._noisy_type, noisy_level=self._noisy_level,
                                                          prefix_context=self._prefix_context, config=processor_config)
        elif self._dataset_name == "family_relation":
            self._dataset_processor = family_relation.family_relation(if_in_context=self._if_in_context,
                                                                      n_shots=self._n_shots,
                                                                      n_noisy_shots=self._n_noisy_shots,
                                                                      noisy_type=self._noisy_type,
                                                                      noisy_level=self._noisy_level,
                                                                      prefix_context=self._prefix_context,
                                                                      config=processor_config)
            self._dataset_config = self._dataset_processor.get_config()
        elif self._dataset_name == "GSM":
            self._dataset_processor = GSM.GSM(n_shots=self._n_shots, n_noisy_shots=self._n_noisy_shots, noisy_type=self._noisy_type,  noisy_level=self._noisy_level, prefix_context=self._prefix_context)
        elif self._dataset_name == "SCAN":
            self._dataset_processor = scan_master.scan_master(if_in_context = self._if_in_context, n_shots=self._n_shots, n_noisy_shots=self._n_noisy_shots, noisy_type=self._noisy_type,  noisy_level=self._noisy_level, prefix_context=self._prefix_context, config = processor_config)
        else:
            raise ValueError("Unsupported dataset {}".format(self._dataset_name))
        self._dataset = self._dataset_processor.load_data()
        assert len(self._dataset) >= self._test_num

    def _get_log_file_name(self):
        log_path = os.path.join("result", self._dataset_name, self._model_name)

        if self._dataset_name == "family_relation":
            log_path = os.path.join(log_path, self._dataset_config["reasoning_type"])
            if self._dataset_config["reasoning_type"] == "symbolic":
                log_path = os.path.join(log_path, "hop" + str(self._dataset_config["hop"]))
                # elif self._dataset_name == "base_math":
        #     log_path = os.path.join(log_path, "base"
        if self._model_name.split("-")[0] == "gpt":
            log_path = os.path.join(log_path, f"reason_temperature{self.temperature_reason}")
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        log_file = "log"
        if self._if_in_context:
            if self._prefix_context:
                log_file += "_prefix"
            log_file += "_ICL_{}".format(self._n_shots)
            if self._n_weak_shots > 0:
                log_file += "_{}weak".format(self._n_weak_shots)
        if self._if_noise:
            log_file += "_noise_{}{}_level{}".format(self._n_noisy_shots, self._noisy_type, self._noisy_level)
        else:
            log_file += "_origin"
        if self._if_rephrase:
            log_file += "_rephrase_aggregate_{}".format(self._rephrase_aggregate)
            log_file += "_rephrase_temp{}".format(self._temperature_rephrase)
        log_file += "_case{}.log".format(self._test_num - self._start_num)
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
                if test_num <= 0:
                    break
            self._query_process()
            self._noise_test_result = [self._correct_num, self._error_num, self._answers_list, self._contents_list]
            self._save_result()
            self._log("correct_num:{}, error_num:{}, correate_rate:{}".format(self._correct_num, self._error_num,
                                                                              self._correct_num / (
                                                                                          self._correct_num + self._error_num)))
        self._log("End time: {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        return self._noise_test_result

    def _log(self, obj):
        print(obj)
        self._log_file.write(str(obj) + "\n")
        self._log_file.flush()

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

    def _random_mask_words(self, text, percent, mask="xxxx"):
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        words = nltk.word_tokenize(text)
        non_punctuation_words = [i for i, word in enumerate(words) if word not in string.punctuation]
        num_mask = int(len(non_punctuation_words) * percent)
        # print(num_to_replace)
        to_mask = random.sample(non_punctuation_words, num_mask)
        for i in to_mask:
            words[i] = mask
        return " ".join(words)

    def _response_process(self, case_batch):
        for case in case_batch:
            context = case["messages"][:-1]
            label = case["label"]
            self._log(json.dumps(context))
            self._log("\ncorrect answer is {}\n".format(label))
            responses = case["messages"][-1]
            for response in responses:
                raw_answer = response["content"]
                self._contents_list.append(raw_answer)
                self._log(raw_answer)

                answer = self._dataset_processor.match_answer(raw_answer)
                if answer:
                    if answer == label:
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
        case_list = [copy.deepcopy(self._case_list[i:i + batch_size]) for i in
                     range(0, len(self._case_list), batch_size)]
        for index, case_batch in enumerate(case_list):
            temperature_reason = self.temperature_reason
            n_reason = self.n_reason
            if self.if_rephrase:
                if self.rephrase_aggregate:
                    case_batch = self._rephrase_aggregate(case_batch)
                else:
                    case_batch = self._rephrase(case_batch)
            self._model.query_batch(case_batch, temperature_reason, n_reason)
            self._response_process(case_batch)
            self._log(
                f"index {index}/{len(case_list) - 1}, correct_num {self._correct_num}, error_num {self._error_num}, accuracy {self._correct_num / (self._correct_num + self._error_num)}")
            self._log(self._model.compute_cost())
        self._answers_list = [self._answers_list[i:i + self.n_reason]
                              for i in range(0, len(self._answers_list), self.n_reason)]
        self._contents_list = [self._contents_list[i:i + self.n_reason]
                               for i in range(0, len(self._contents_list), self.n_reason)]

    def _question_insert(self, raw_data):
        processed_case = self._dataset_processor.get_case(raw_data)
        # for i in range(self._run_times):
        # case = copy.deepcopy(processed_case)
        self._case_list.append(processed_case)

    def _save_result(self):
        with open(self._pickle_name, 'wb') as f:
            pickle.dump(self._noise_test_result, f)

    def _rephrase_icl_shots(self, case):
        if self._dataset_name == "base_math":
            expr = "47+58"
        elif self._dataset_name == "SCAN":
            expr = ["walk around right twice after run opposite left", ["I_TURN_LEFT","I_TURN_LEFT","I_RUN","I_TURN_RIGHT","I_WALK","I_TURN_RIGHT","I_WALK","I_TURN_RIGHT","I_WALK","I_TURN_RIGHT","I_WALK","I_TURN_RIGHT","I_WALK","I_TURN_RIGHT","I_WALK","I_TURN_RIGHT","I_WALK","I_TURN_RIGHT","I_WALK"]]
        else:
            raise ValueError("dataset type {} not support rephrase".format(self._dataset_name))
        temperature_rephrase = self.temperature_rephrase
        n_rephrase = self.n_rephrase
        contrastive_queries = []
        in_context = case["in-context"]
        for shot in in_context:
            contrastive_case = dict()
            # contrastive_question = "Below are two examples of same kind of questions: one is a good example and the other is a poor one. Could you analyze the good example, identify the issues in the poor one, and provide a corrected version of the assistant's response? The revised response should include a reasoning process similar to that in the good example."
            if self._dataset_name == "SCAN":
                contrastive_question = self._dataset_processor.get_sys_prompt()
            else:
                contrastive_question = ""
            contrastive_question += "The following are two examples for this same kind of tasks: " \
                                   "there is an excellent response and a distracted response. " \
                                   "Please follow the good response and provide a corrected version of the distracted response, " \
                                   "which must be logically consistent with the excellent one."
            # contrastive_question = "The following are two examples for base-9 questions, there is a good example and a bad example, can you highlight the important and correct steps in the bad example and provided the modified version for me? "
            contrastive_question += "Good Example:\nQ:"
            contrastive_question += self._dataset_processor.get_question(expr)
            contrastive_question += "\nA:"
            if self._dataset_name == "base_math": 
                contrastive_question += self._dataset_processor.answer(expr)
            if self._dataset_name == "SCAN":
                contrastive_question += self._dataset_processor.get_answer(expr, False)
            contrastive_question += "\n"
            contrastive_question += "Bad Example:\nQ:"
            contrastive_question += shot[0]
            contrastive_question += "\nA:"
            contrastive_question += shot[1]
            contrastive_question += "\n You must answer in the format of \"correct version is:{only the correct version of response in the bad example with reaoning step like in good example}\". "
            contrastive_question += "Don't offer anything else."
            contrastive_case["question"] = contrastive_question
            contrastive_queries.append(contrastive_case)
        self._model.query_batch(contrastive_queries, temperature_rephrase, n_rephrase)
        n_shot_list = []
        for shot, query in zip(in_context, contrastive_queries):
            # 获取多个response
            n_shot = []
            responses = query["messages"][-1]
            for response in responses:
                new_shot = copy.deepcopy(shot)
                content = response['content']
                match = re.search(r'[Cc]orrect [Vv]ersion.*?:([\s\S]*)', content)
                if match:
                    answer = match.group(1)
                    new_shot[1] = answer
                else:
                    self._log("not match")
                # print("noisy answer:{}".format(shot[1]))
                # print("rephrased answer:{}".format(new_shot[1]))
                n_shot.append(new_shot)
            n_shot_list.append(n_shot)
        return n_shot_list

    def _rephrase(self, case_batch):
        n_case_batch = []
        for case in case_batch:
            n_shot_list = self._rephrase_icl_shots(case)
            for context in zip(*n_shot_list):
                new_case = copy.deepcopy(case)
                new_case['in-context'] = list(context)
                # print(new_case['in-context'])
                n_case_batch.append(new_case)
        return n_case_batch

    def _select_n_shot(self, n_shot):
        answer_list = []
        answer_shot = dict()
        for i in range(len(n_shot)):
            shot = n_shot[i]
            raw_answer = shot[1]
            answer = self._dataset_processor.match_answer(raw_answer)
            if answer is not None:
                answer_list.append(answer)
                if answer not in answer_shot:
                    answer_shot[answer] = [i]
                else:
                    answer_shot[answer].append(i)
        from collections import Counter
        counter = Counter(answer_list)
        most_answer, _ = counter.most_common(1)[0]
        shots_index = answer_shot[most_answer]
        token_list = []
        for i in shots_index:
            token_list.append(len(n_shot[i][1]))
        selected_token = sorted(token_list)[(len(token_list)-1)//2]
        selected_index = shots_index[token_list.index(selected_token)]
        selected_shot = n_shot[selected_index]
        self._log("selected_index:{}".format(selected_index))
        self._log("selected_shot:{}".format(selected_shot))
        return selected_shot

    def _rephrase_aggregate(self, case_batch):
        for case in case_batch:
            n_shot_list = self._rephrase_icl_shots(case)
            selected_shots = []
            for n_shot in n_shot_list:
                selected_shot = self._select_n_shot(n_shot)
                selected_shots.append(selected_shot)
            for i in range(len(case['in-context'])):
                case['in-context'][i] = selected_shots[i]
        return case_batch

    def COT_SC_correct_rate(self, answers_list):
        from collections import Counter
        valid_count = 0
        SC_right_count = 0
        for answers in answers_list:
            answers = [sublist for sublist in answers if isinstance(sublist, list)]  # clean answers
            if (len(answers) == 0):
                continue
            else:
                valid_count += 1

            second_elements_are_1 = [sublist[1] == 1 for sublist in answers]
            any_second_element_is_1 = any(second_elements_are_1)
            if not any_second_element_is_1:
                continue
            true_answer = next((sublist[0] for sublist in answers if sublist[1] == 1), None)
            counter = Counter(sublist[0] for sublist in answers)
            guess_value, _ = counter.most_common(1)[0]
            if guess_value == true_answer:
                SC_right_count += 1

        self._log("SC_correct_num:{}, vaild_num:{}, SC_correct_rate:{}".format(SC_right_count, valid_count,
                                                                       SC_right_count / valid_count))
        return SC_right_count, valid_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', type=str, default='config.yml', help='Path to the config file')
    with open('config.yml', 'r') as f:
        config = yaml.safe_load(f)
    test = noise_test(args=config)
    [correct_num, error_num, answer_list, answer_cotents] = test.run()
    
    # with open('./result/base_math/gpt-3.5-turbo-0613/temperature1/rephrase/log_ICL_0_noise_3irrelative_level3.pkl', 'rb') as f:
    #     lists = pickle.load(f)

    # [correct_num, error_num, answer_list, answer_cotents]  = lists

    test.COT_SC_correct_rate(answer_list)
