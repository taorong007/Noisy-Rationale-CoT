import yaml
import os
from typing import List, Optional
import json
import re
import pickle
import data_process.base_math.base_math as base_math
import data_process.family_relation.family_relation as family_relation
import data_process.GSM.GSM as GSM
import data_process.SCAN.scan_master as scan_master
import data_process.tracking_shuffled_objects.tracking_shuffled_objects as shuffled_obj
import data_process.BBH.bbh as bbh
import pandas as pd
import nltk
import random
import time
from datetime import datetime
import copy
import string
import argparse
import zipfile

def wr_log(obj, log_file):
    print(obj)
    log_file.write(str(obj) + "\n")
    log_file.flush()


class noise_test:
    def __init__(self, args) -> None:
        self.config = args
        self._model_name = args["model"]
        self._dataset_name = args["dataset"]
        self._start_num = args["start_num"]
        self._test_num = args["test_num"]
        self._batch_size = args["batch_size"]

        assert self._test_num / self._batch_size == int(
            self._test_num / self._batch_size), "test_num / batch_size should be a positive integer"

        self.use_processed_dataset =  args["use_processed_dataset"]
        if(self.use_processed_dataset):
            processed_dataset_options = args["processed_dataset_options"]
            processed_dataset_path = processed_dataset_options["processed_dataset_path"]
            if processed_dataset_path.startswith("default-"):
                dataset_label = processed_dataset_path.split("-")
                self.processed_dataset_path = self._get_default_processed_dataset_name(dataset_label)
            else:
                self.processed_dataset_path = processed_dataset_path
            with open(self.processed_dataset_path, "r", encoding="utf-8") as f:
                config = json.load(f)["config"]
            if dataset_label[1] == "zeroshot":
                config["if_in_context"] = False
            else:
                config["if_in_context"] = True
                assert processed_dataset_options["n_shots"] <= config["n_max_shots"]
                if config["if_noise"] == True:
                    config["n_shots"] = 0
                    config["n_noisy_shots"] = processed_dataset_options["n_shots"]
                else:
                    config["n_shots"] = processed_dataset_options["n_shots"]
                    config["n_noisy_shots"] = 0
                
        else:
            config = args["raw_dataset_options"]
        
        self._if_in_context = config["if_in_context"] if "if_in_context" in config else False
        if self._if_in_context:
            self._if_noise = config["if_noise"] if "if_noise" in config else False
            self._n_shots = config["n_shots"] if "n_shots" in config else 1
            self._n_weak_shots = config["n_weak_shots"] if "n_weak_shots" in config else 0
        else:
            self._if_noise = False
            self._n_shots = 0
            self._n_weak_shots = 0

        if self._if_noise:
            self._n_noisy_shots = config["n_noisy_shots"] if "n_noisy_shots" in config else 0
            if self._n_noisy_shots == 0:
                self._if_noise = False
                self._noise_type = None
                self._noise_ratio = 0
                self._noise_distribution = None
            else:
                self._noise_type = config["noise_type"]
                self._noise_ratio = config["noise_ratio"]
                self._noise_distribution = config["noise_distribution"]
        else:
            self._n_noisy_shots = 0
            self._noise_type = None
            self._noise_ratio = 0
            self._noise_distribution = None
        
        self._prefix_context = args["prefix_context"] if "prefix_context" in args else False

        
        random.seed(time.time())

        self._init_model()
        self._init_dataset()
        self._init_method()
        log_name = args["log_name"] if "log_name" in args else self._get_log_file_name()
        print(f"test result is in {log_name}")
        self._log_file = open(log_name, 'w', encoding='utf-8')
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

        return

    def _unzip_default_processed_dataset(self, file_dir):
        file_path = os.path.join(file_dir, "processed.zip")
        # if os.path.exists(file_path):
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(file_dir)
        print(f"processed_dataset has been extracted to {file_dir}")
            
    def _get_default_processed_dataset_name(self, dataset_label):
        args = self.config
        noise_type = ["zeroshot","clean", "irrelevant", "inaccurate"]
        noise_difficulty = ["easy", "medium", "hard"]
        type = dataset_label[1]
        assert type in noise_type
        if type in ["irrelevant", "inaccurate"]:
            file_name =  f"{type}"
            difficulty = dataset_label[2]
            distribution = dataset_label[3]
            assert difficulty in noise_difficulty
            file_name += f"_{difficulty}_{distribution}.json"
        else:
            file_name = "clean.json"
        if self._dataset_name == "base_math":
            reasoning_type = args[self._dataset_name]["reasoning_type"]
            dataset_dir = os.path.join("data", "base_math") 
            processed_dataset_dir = os.path.join("data", "base_math", "processed",  reasoning_type) 
        elif self._dataset_name == "family_relation":
            dataset_dir = os.path.join("data", "data_emnlp_final")
            processed_dataset_dir = os.path.join("data", "data_emnlp_final", "processed") 
        elif self._dataset_name == "SCAN":
            reasoning_type = args[self._dataset_name]["reasoning_type"]
            dataset_dir = os.path.join("data", "SCAN-master")
            processed_dataset_dir = os.path.join("data", "SCAN-master", "processed", reasoning_type) 
        else:
            raise ValueError(f"dataset {self._dataset_name} are not supported in default")
        if not os.path.exists(os.path.join(processed_dataset_dir, file_name)):
            self._unzip_default_processed_dataset(dataset_dir)
        if not os.path.exists(os.path.join(processed_dataset_dir, file_name)):
            raise ValueError(f"default file {os.path.join(processed_dataset_dir, file_name)} not exist")
        return os.path.join(processed_dataset_dir, file_name)

    def _init_model(self):
        if self._model_name == "llama2":
            from llm_model.llama.my_llama import my_llama
            model_config = self.config["llama2"] if "llama2" in self.config else None
            self._model = my_llama(config=model_config)
        elif self._model_name.split("-")[0] == "gpt":
            from llm_model.my_gpt.my_gpt import my_gpt
            model_config = self.config["gpt"] if "gpt" in self.config else None
            self._model = my_gpt(model=self._model_name, config=model_config)
        else:
            raise ValueError("Unsupported model {}".format(self._model_name))

    def _load_processed_dataset(self):
        with open(self.processed_dataset_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
            dataset_content = dataset["content"]
            if "system-prompt" in dataset:
                self._dataset_system_prompt = dataset["system-prompt"]
            else:
                self._dataset_system_prompt = None
        return dataset_content

    def _init_dataset(self):
        processor_config = self.config[self._dataset_name] if self._dataset_name in self.config else None
        if self._dataset_name == "base_math":
            self._dataset_processor = base_math.base_math(n_shots=self._n_shots,
                                                            n_noisy_shots=self._n_noisy_shots,
                                                            noise_type=self._noise_type, noise_ratio=self._noise_ratio, noise_distribution=self._noise_distribution,
                                                            prefix_context=self._prefix_context, config=processor_config)
        elif self._dataset_name == "family_relation":
            self._dataset_processor = family_relation.family_relation(if_in_context=self._if_in_context,
                                                                      n_shots=self._n_shots,
                                                                      n_noisy_shots=self._n_noisy_shots,
                                                                      noise_type=self._noise_type,
                                                                      noise_ratio=self._noise_ratio,
                                                                      prefix_context=self._prefix_context,
                                                                      config=processor_config)
            self._dataset_config = self._dataset_processor.get_config()
        elif self._dataset_name == "GSM":
            self._dataset_processor = GSM.GSM(n_shots=self._n_shots, n_noisy_shots=self._n_noisy_shots, noise_type=self._noise_type,  noisy_level=self._noisy_level, prefix_context=self._prefix_context)
        elif self._dataset_name == "SCAN":
            self._dataset_processor = scan_master.scan_master(n_shots=self._n_shots, n_noisy_shots=self._n_noisy_shots, noise_type=self._noise_type,  noise_ratio=self._noise_ratio, noise_distribution=self._noise_distribution, prefix_context=self._prefix_context, config = processor_config)
        elif self._dataset_name == "BBH":
            self._dataset_processor = bbh.bbh(n_shots=self._n_shots, n_noisy_shots=self._n_noisy_shots, noise_type=self._noise_type,  noise_ratio=self._noise_ratio, noise_distribution=self._noise_distribution, prefix_context=self._prefix_context, config = processor_config)
        elif self._dataset_name == "shuffled_obj":
            self._dataset_processor = shuffled_obj.tracking_shuffled_objects(n_shots=self._n_shots, n_noisy_shots=self._n_noisy_shots, noise_type=self._noise_type,  noise_ratio=self._noise_ratio, noise_distribution=self._noise_distribution, prefix_context=self._prefix_context, config = processor_config)
        else:
            raise ValueError("Unsupported dataset {}".format(self._dataset_name))
        if not self.use_processed_dataset:
            self._dataset = self._dataset_processor.load_data()
        else:
            self._dataset_processor.load_data()
            self._dataset =  self._load_processed_dataset()
        assert len(self._dataset) >= self._test_num

    def _init_method(self):
        self.method = self.config["method"]
        args = self.config
        if self.method == "baseline":
            self.temperature_reason = args["temperature_reason"] if "temperature_reason" in args else 1
            self.n_reason = args["n_reason"] if "n_reason" in args else 1
        elif self.method == "RV":
            self.temperature_rephrase = args["temperature_rephrase"] if "temperature_rephrase" in args else 1
            self.n_rephrase = args["n_rephrase"] if "n_rephrase" in args else 5
            self.RV_n_reason = args["RV_n_reason"] if "RV_n_reason" in args else 1
            self.RV_temp_reason = args["RV_temp_reason"] if "RV_temp_reason" in args else 0.1
            self.RV_topp_reason = args["RV_topp_reason"] if "RV_topp_reason" in args else 1
        elif self.method == "RAV":
            self.temperature_rephrase = args["temperature_rephrase"] if "temperature_rephrase" in args else 1
            self.n_rephrase = args["n_rephrase"] if "n_rephrase" in args else 5
            self.RAV_n_reason = args["RAV_n_reason"] if "RAV_n_reason" in args else 5
            self.RAV_temp_reason = args["RAV_temp_reason"] if "RAV_temp_reason" in args else 1
            self.RAV_topp_reason = args["RAV_topp_reason"] if "RAV_topp_reason" in args else 0.9
        elif self.method == "both":
            self.temperature_rephrase = args["temperature_rephrase"] if "temperature_rephrase" in args else 1
            self.n_rephrase = args["n_rephrase"] if "n_rephrase" in args else 5
            self.RV_n_reason = args["RV_n_reason"] if "RV_n_reason" in args else 1
            self.RV_temp_reason = args["RV_temp_reason"] if "RV_temp_reason" in args else 0.1
            self.RV_topp_reason = args["RV_topp_reason"] if "RV_topp_reason" in args else 1
            self.RAV_n_reason = args["RAV_n_reason"] if "RAV_n_reason" in args else 5
            self.RAV_temp_reason = args["RAV_temp_reason"] if "RAV_temp_reason" in args else 1
            self.RAV_topp_reason = args["RAV_topp_reason"] if "RAV_topp_reason" in args else 0.9
            self.RV_weight = args["RV_weight"] if "RV_weight" in args else 0.5
            self.RAV_weight = args["RAV_weight"] if "RAV_weight" in args else 0.5
        elif self.method == "smoothllm":
            from method.smooth_llm_main.lib.defenses import SmoothLLM
            self.temperature_reason = args["temperature_reason"] if "temperature_reason" in args else 1
            self.n_reason = args["n_reason"] if "n_reason" in args else 1
            self.smoothllm = SmoothLLM(self._model, self._dataset_processor, "RandomSwapPerturbation", 10, self.n_reason)
        elif self.method == "selfdenoise":
            from method.SelfDenoise_main.baseline_test import SelfDenoise
            self.temperature_reason = args["temperature_reason"] if "temperature_reason" in args else 1
            self.n_reason = args["n_reason"] if "n_reason" in args else 1
            self.SelfDenoise = SelfDenoise(n_reason=self.n_reason)
        elif self.method == "contrastivecot":
            self.temperature_reason = args["temperature_reason"] if "temperature_reason" in args else 1
            self.n_reason = args["n_reason"] if "n_reason" in args else 1
        elif self.method == "ISC":
            self.temperature_reason = args["temperature_reason"] if "temperature_reason" in args else 1
            self.n_reason = args["n_reason"] if "n_reason" in args else 1
        # if self.method == "smoothllm":
        #     self.smoothllm = SmoothLLM(self._model, self._dataset_processor, "RandomSwapPerturbation", 10, self.n_reason)
        # elif self.method == "selfdenoise":
        #     self.SelfDenoise = SelfDenoise(n_reason=self.n_reason)

    def _get_log_file_name(self):
        log_path = os.path.join("result", self._dataset_name)
        dataset_config = self.config[self._dataset_name] if self._dataset_name in self.config else None
        if dataset_config != None:
            if "reasoning_type" in dataset_config:
                log_path = os.path.join(log_path, dataset_config["reasoning_type"])
                    
        if self._dataset_name == "family_relation":
            if self._dataset_config["reasoning_type"] == "symbolic":
                log_path = os.path.join(log_path, "hop" + str(self._dataset_config["hop"]))
        log_path = os.path.join(log_path, self._model_name)
        log_path = os.path.join(log_path, f"method_{self.method}")
        if "subfolder_suffix_path" in self.config:
            if len(self.config["subfolder_suffix_path"]) > 0:
                log_path = os.path.join(log_path, self.config["subfolder_suffix_path"])        
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        log_file = "log"
        if self._if_in_context:
            if self._prefix_context:
                log_file += "_prefix"
            log_file += "_ICL_{}clean".format(self._n_shots)
            if self._n_weak_shots > 0:
                log_file += "_{}weak".format(self._n_weak_shots)
        if self._if_noise:
            log_file += "_noise_{}{}_{}_ratio{}".format(self._n_noisy_shots, self._noise_type, self._noise_distribution, self._noise_ratio)
        else:
            log_file += "_origin"

        log_file += "_case{}".format(self._test_num)
        if self.method == "baseline":
            log_file += "_temp{}_n{}".format(self.temperature_reason, self.n_reason)
        elif self.method == "RV":
            log_file += "_rephrase_temp{}_n{}".format(self.temperature_rephrase, self.n_rephrase)
            log_file += "_reason_temp{}_n{}".format(self.RV_temp_reason, self.RV_n_reason)
            log_file += "_topp{}_".format(self.RV_topp_reason)
        elif self.method == "RAV":
            log_file += "_rephrase_temp{}_n{}".format(self.temperature_rephrase, self.n_rephrase)
            log_file += "_reason_temp{}_n{}".format(self.RAV_temp_reason, self.RAV_n_reason)
            log_file += "_topp{}_".format(self.RAV_topp_reason)
        elif self.method == "both":
            log_file += "_rephrase_temp{}_n{}".format(self.temperature_rephrase, self.n_rephrase)
            log_file += "_reason_temp_RV{}_RAV{}_n_RV{}_RAV{}".format(self.RV_temp_reason, self.RAV_temp_reason,
                                                                      self.RV_n_reason, self.RAV_n_reason)
            log_file += "_topp_RV{}_RAV{}".format(self.RV_topp_reason, self.RAV_topp_reason)
        else:
            log_file += "_temp{}_n{}".format(self.temperature_reason, self.n_reason)
        log_file += ".log"
        log_file_path = os.path.join(log_path, log_file)
        return log_file_path

    def run(self):
        self._log("Start time: {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        if self._noise_test_result is None:
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
            self._noise_test_result = dict()
            self._noise_test_result["correct_num"] = self._correct_num
            self._noise_test_result["error_num"] = self._error_num
            self._noise_test_result["not_match_num"] = self._not_match_num
            self._noise_test_result["answers_list"] = self._answers_list
            self._noise_test_result["contents_list"] = self._contents_list
            self._noise_test_result["question_list"] = [case["question"] for case in self._case_list]
            self._noise_test_result["label_list"] = [case["label"] for case in self._case_list]
            # self._noise_test_result = [self._correct_num, self._error_num, self._answers_list, self._contents_list]
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
            context = case["messages"]
            label = case["label"]
            self._log(json.dumps(context))
            self._log("\nCorrect answer is {}\n".format(label))
            responses = case["messages"][-1]  # all responses
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
            if self.method == "baseline":
                case_n = self.n_reason
                self._model.query_case_batch(case_batch, self.temperature_reason, self.n_reason)
                self._response_process(case_batch)
            elif self.method == "RV":
                case_n = self.n_rephrase * self.RV_n_reason
                case_batch = self._rephrase(case_batch)
                self._model.query_case_batch(case_batch, self.RV_temp_reason, self.RV_n_reason, self.RV_topp_reason)
                self._response_process(case_batch)
            elif self.method == "RAV":
                case_n = self.RAV_n_reason
                case_batch = self._rephrase_aggregate(case_batch)
                self._model.query_case_batch(case_batch, self.RAV_temp_reason, self.RAV_n_reason, self.RAV_topp_reason)
                self._response_process(case_batch)
            elif self.method == "both":
                RV_case_batch, RAV_case_batch = self._rephrase_both(case_batch)
                self._model.query_case_batch(RV_case_batch, self.RV_temp_reason, self.RV_n_reason, self.RV_topp_reason)
                self._model.query_case_batch(RAV_case_batch, self.RAV_temp_reason, self.RAV_n_reason, self.RAV_topp_reason)
                split_RV_case_batch = [RV_case_batch[i:i + self.n_rephrase] for i in
                                       range(0, len(RV_case_batch), self.n_rephrase)]
                split_RAV_case_batch = [[RAV_case_batch[i]] for i in range(len(RAV_case_batch))]
                merge_split_case_batch = [split_RV_case_batch[i] + split_RAV_case_batch[i] for i in
                                          range(len(split_RV_case_batch))]
                merge_case_batch = []
                for merge_split in merge_split_case_batch:
                    merge_case_batch += merge_split
                self._response_process(merge_case_batch)
                case_n = self.n_rephrase * self.RV_n_reason + self.RAV_n_reason
            elif self.method == "smoothllm":
                case_batch = self.smoothllm(case_batch)
                self._response_process(case_batch)
                case_n = 1
            elif self.method == "selfdenoise":
                case_batch = self.SelfDenoise.certify(case_batch, model= self._model)
                self._response_process(case_batch)
                case_n = self.n_reason
            elif self.method == "contrastivecot":
                if self._dataset_name == "base_math":
                    expr = "47+58"
                elif self._dataset_name == "SCAN":
                    # expr = self._dataset_processor.get_random_demos(1)[0]
                    expr = ["walk around right twice after run opposite left",
                            ["I_TURN_LEFT", "I_TURN_LEFT", "I_RUN", "I_TURN_RIGHT", "I_WALK", "I_TURN_RIGHT", "I_WALK",
                            "I_TURN_RIGHT", "I_WALK", "I_TURN_RIGHT", "I_WALK", "I_TURN_RIGHT", "I_WALK", "I_TURN_RIGHT",
                            "I_WALK", "I_TURN_RIGHT", "I_WALK", "I_TURN_RIGHT", "I_WALK"]]
                elif self._dataset_name == "family_relation":
                    expr = self._dataset_processor.get_random_demos(1).iloc[0]
                postive_QAL = []
                postive_QAL.append(self._dataset_processor.get_question(expr))
                postive_QAL.append(self._dataset_processor.get_correct_answer(expr))
                postive_QAL.append(self._dataset_processor.get_label(expr))
                from method.Contrastive_CoT.Contrastive_CoT import Contrastive_CoT
                case_batch = Contrastive_CoT(postive_QAL=postive_QAL, case_batch=case_batch, model=self._model, dataprocessor=self._dataset_processor, n_reason=self.n_reason)
                self._response_process(case_batch)
                case_n = self.n_reason
            elif self.method == "ISC":
                from method.Intrinsic_Self_Correct.Intrinsic_Self_Correct import Intrinsic_Self_Correct
                case_batch = Intrinsic_Self_Correct(case_batch=case_batch, model=self._model, dataset_name=self._dataset_name, n_reason=self.n_reason)
                self._response_process(case_batch)
                case_n = self.n_reason
            if self._correct_num + self._error_num == 0:
                self._log(
                f"index {index}/{len(case_list) - 1}, correct_num {self._correct_num}, error_num {self._error_num}, "
                f"accuracy NULL")
            else:
                self._log(
                    f"index {index}/{len(case_list) - 1}, correct_num {self._correct_num}, error_num {self._error_num}, "
                    f"accuracy {self._correct_num / (self._correct_num + self._error_num)}, " f"correct_num/total_num  {self._correct_num / (self._correct_num + self._error_num + self._not_match_num)}")
            self._log(self._model.compute_cost())
        self._answers_list = [self._answers_list[i:i + case_n]
                              for i in range(0, len(self._answers_list), case_n)]
        self._contents_list = [self._contents_list[i:i + case_n]
                               for i in range(0, len(self._contents_list), case_n)]
        
    def _question_insert(self, raw_data):
        if not self.use_processed_dataset:
            processed_case = self._dataset_processor.get_case(raw_data)
            self._case_list.append(processed_case)
        else:
            case = dict()
            case["question"] = raw_data["question"]
            case["label"] = raw_data["label"]
            demos = []
            for i in range(self._n_shots + self._n_noisy_shots):
                demo = []
                demo.append(raw_data["CoT_demos"][i]["question"])
                demo.append(raw_data["CoT_demos"][i]["answer"])
                demos.append(demo)
            case["in-context"] = demos
            if self._dataset_system_prompt != None:
                case["system-prompt"] = self._dataset_system_prompt
            self._case_list.append(case)
    def _save_result(self):
        with open(self._pickle_name, 'wb') as f:
            pickle.dump(self._noise_test_result, f)

    def _rephrase_icl_shots(self, case):
        if self._dataset_name == "base_math":
            expr = "47+58"
        elif self._dataset_name == "SCAN":
            expr = ["walk around right twice after run opposite left",
                    ["I_TURN_LEFT", "I_TURN_LEFT", "I_RUN", "I_TURN_RIGHT", "I_WALK", "I_TURN_RIGHT", "I_WALK",
                     "I_TURN_RIGHT", "I_WALK", "I_TURN_RIGHT", "I_WALK", "I_TURN_RIGHT", "I_WALK", "I_TURN_RIGHT",
                     "I_WALK", "I_TURN_RIGHT", "I_WALK", "I_TURN_RIGHT", "I_WALK"]]
        elif self._dataset_name == "family_relation":
            expr = self._dataset_processor.get_random_demos(1).iloc[0]
        else:
            raise ValueError("dataset type {} not support rephrase".format(self._dataset_name))
        temperature_rephrase = self.temperature_rephrase
        n_rephrase = self.n_rephrase
        contrastive_queries = []
        in_context = case["in-context"]
        for shot in in_context:
            contrastive_case = dict()
            if self._dataset_name == "SCAN":
                contrastive_question = self._dataset_processor.get_sys_prompt()
            else:
                contrastive_question = ""
            contrastive_question += "The following are two examples for this same kind of tasks: " \
                                    "there is an excellent response and a distracted response. " \
                                    "Please follow the good response and provide a corrected version of the distracted response, " \
                                    "which must be logically consistent with the excellent one."
            contrastive_question += "Good Example:\nQ:"
            contrastive_question += self._dataset_processor.get_question(expr)
            contrastive_question += "\nA:"
            # if self._dataset_name == "base_math":
            #     standard_answer = self._dataset_processor.answer(expr)
                
            # elif self._dataset_name == "SCAN":
            #     standard_answer = self._dataset_processor.get_correct_answer(expr)
            # elif self._dataset_name == "family_relation":
            standard_answer = self._dataset_processor.get_correct_answer(expr)
            contrastive_question += standard_answer
            contrastive_question += "\n"
            contrastive_question += "Bad Example:\nQ:"
            contrastive_question += shot[0]
            contrastive_question += "\nA:"
            contrastive_question += shot[1]
            contrastive_question += "\n You must answer in the format of \"correct version is:{only the correct " \
                                    "version of response in the bad example with reaoning step like in good " \
                                    "example}\". "
            contrastive_question += "Don't offer anything else."
            contrastive_case["question"] = contrastive_question
            contrastive_queries.append(contrastive_case)
        self._model.query_case_batch(contrastive_queries, temperature_rephrase, n_rephrase)
        n_shot_list = []
        for shot, query in zip(in_context, contrastive_queries):
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
                    self._log("rephrased response not match")
                # print("noisy answer:{}".format(shot[1]))
                # print("rephrased answer:{}".format(new_shot[1]))
                n_shot.append(new_shot)
            n_shot_list.append(n_shot)
        return n_shot_list, standard_answer

    def _rephrase(self, case_batch):
        n_case_batch = []
        for case in case_batch:
            n_shot_list, standard_answer = self._rephrase_icl_shots(case)
            for context in zip(*n_shot_list):
                new_case = copy.deepcopy(case)
                new_case['in-context'] = list(context)
                n_case_batch.append(new_case)
        return n_case_batch

    def _select_n_shot(self, n_shot, standard_answer_embedding):
        answer_list = []
        answer_shot = dict()
        # self._log("rephrased_question:{}".format(n_shot[0][0]))
        # self._log("rephrased_response:\n")
        for i in range(len(n_shot)):
            shot = n_shot[i]
            raw_answer = shot[1]
            # self._log("index_{}: {}\n".format(i, raw_answer))
            answer = self._dataset_processor.match_answer(raw_answer)
            if answer is not None:
                answer_list.append(answer)
                if answer not in answer_shot:
                    answer_shot[answer] = [i]
                else:
                    answer_shot[answer].append(i)
        # 1. get the most confident answers
        from collections import Counter
        counter = Counter(answer_list)
        count_pairs = counter.most_common()
        _, most_count = count_pairs[0]
        most_answer_list = [pair[0] for index, pair in enumerate(count_pairs) if pair[1] == most_count]
        most_consistent_index = []
        for answer in most_answer_list:
            most_consistent_index += answer_shot[answer]

        # 2. heuristic removal of unreasonable responses
        heuristic_selected_index = copy.deepcopy(most_consistent_index)
        for i in most_consistent_index:
            raw_answer = n_shot[i][1]
            # task-specific
            if self._dataset_name == "base_math":
                token_limit = 5
                phrase_limit = 20
            elif self._dataset_name == "SCAN":
                token_limit = 20
                phrase_limit = 30  # need to be changed
            # remove over-short response
            if len(heuristic_selected_index) > 1 and len(raw_answer) < token_limit:
                heuristic_selected_index.remove(i)
            # remove over-long response
            import re
            pattern = r'[.,?!]'
            phrases = re.split(pattern, raw_answer)
            if len(heuristic_selected_index) > 1 and len(phrases) > phrase_limit:
                heuristic_selected_index.remove(i)

        # 3. rank responses by similarity
        if len(heuristic_selected_index) == 1:
            selected_index = heuristic_selected_index[0]
            selected_shot = n_shot[selected_index]
        else:
            answer_embeddings = []
            for j in heuristic_selected_index:
                shot = n_shot[j]
                raw_answer = shot[1]
                answer_embeddings.append(self._model.get_embedding(raw_answer))
            from sklearn.metrics.pairwise import cosine_similarity
            import numpy as np
            similarity_score = cosine_similarity([standard_answer_embedding],
                                                 answer_embeddings)[0]
            max_index = int(np.argmax(similarity_score))
            selected_index = heuristic_selected_index[max_index]
            selected_shot = n_shot[selected_index]
        self._log("selected_index:{}".format(selected_index))
        self._log("selected_response:{}".format(selected_shot[1]))

        return selected_shot

    def _rephrase_aggregate(self, case_batch):
        standard_answer_embedding = None
        for case in case_batch:
            n_shot_list, standard_answer = self._rephrase_icl_shots(case)
            if standard_answer_embedding is None:
                standard_answer_embedding = self._model.get_embedding(standard_answer)
            selected_shots = []
            for n_shot in n_shot_list:
                selected_shot = self._select_n_shot(n_shot, standard_answer_embedding)
                selected_shots.append(selected_shot)
            for i in range(len(case['in-context'])):
                case['in-context'][i] = selected_shots[i]
        return case_batch

    def _rephrase_both(self, case_batch):
        RV_case_batch = []
        RAV_case_batch = []
        standard_answer_embedding = None
        for case in case_batch:
            n_shot_list, standard_answer = self._rephrase_icl_shots(case)
            # RV
            for context in zip(*n_shot_list):
                new_case = copy.deepcopy(case)
                new_case['in-context'] = list(context)
                RV_case_batch.append(new_case)
            # RAV
            if standard_answer_embedding is None:
                standard_answer_embedding = self._model.get_embedding(standard_answer)
            selected_shots = []
            for n_shot in n_shot_list:
                selected_shot = self._select_n_shot(n_shot, standard_answer_embedding)
                selected_shots.append(selected_shot)
            new_case = copy.deepcopy(case)
            for i in range(len(new_case['in-context'])):
                new_case['in-context'][i] = selected_shots[i]
            RAV_case_batch.append(new_case)
        return RV_case_batch, RAV_case_batch

    def COT_SC_correct_rate(self, answers_list):
        from collections import Counter
        valid_count = 0
        SC_right_count = 0
        for answers in answers_list:
            answers = [sublist for sublist in answers if isinstance(sublist, list)]  # clean answers
            if len(answers) == 0:
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

    def weighted_SC_correct_rate(self, answers_list):
        from collections import Counter
        valid_count = 0
        SC_right_count = 0
        RV_weight = self.RV_weight
        RAV_weight = self.RAV_weight
        RV_length = self.RV_n_reason * self.n_rephrase
        for answers in answers_list:
            RV_answers = [sublist for sublist in answers[:RV_length] if isinstance(sublist, list)]  # clean answers
            RAV_answers = [sublist for sublist in answers[RV_length:] if isinstance(sublist, list)]
            if len(RV_answers) + len(RAV_answers) == 0:
                continue
            else:
                valid_count += 1

            any_prediction_is_right = any(
                [sublist[1] == 1 for sublist in RV_answers] + [sublist[1] == 1 for sublist in RAV_answers])
            if not any_prediction_is_right:
                continue
            true_answer = next((sublist[0] for sublist in RV_answers + RAV_answers if sublist[1] == 1), None)
            RV_counter = Counter(sublist[0] for sublist in RV_answers)
            RAV_counter = Counter(sublist[0] for sublist in RAV_answers)
            RV_weighted_counter = dict()
            RAV_weighted_counter = dict()
            for answer, count in RV_counter.most_common():
                RV_weighted_counter[answer] = count * RV_weight
            for answer, count in RAV_counter.most_common():
                RAV_weighted_counter[answer] = count * RAV_weight
            merged_weighted_counter = copy.deepcopy(RV_weighted_counter)
            for k, v in RAV_weighted_counter.items():
                if k in RV_weighted_counter:
                    merged_weighted_counter[k] += RAV_weighted_counter[k]
                else:
                    merged_weighted_counter[k] = RAV_weighted_counter[k]
            guess_answer = None
            most_counter = 0
            for k,v in merged_weighted_counter.items():
                if v > most_counter:
                    most_counter = v
                    guess_answer = k
            if guess_answer == true_answer:
                SC_right_count += 1

        self._log("RV_weight:{}, RAV_weight:{}, weighted_SC_correct_num:{}, vaild_num:{}, SC_correct_rate:{}".format(
            self.RV_weight, self.RAV_weight, SC_right_count, valid_count, SC_right_count / valid_count))
        return SC_right_count, valid_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', type=str, default='config.yml', help='Path to the config file')
    args = parser.parse_args()
    config_path = args.config
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    test = noise_test(args=config)
    result = test.run()
    answers_list = result["answers_list"]
    # with open('./result/base_math/gpt-3.5-turbo-0613/temperature1/rephrase/log_ICL_0_noise_3irrelevant_level3.pkl', 'rb') as f:
    #     lists = pickle.load(f)

    # [correct_num, error_num, answer_list, answer_cotents]  = lists
    if test.method == "both":
        test.weighted_SC_correct_rate(answers_list)
    else:
        test.COT_SC_correct_rate(answers_list)
