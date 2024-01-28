import time
import yaml
import openai
from ..multiple_key import init_api_key_handling
import concurrent
import os
from dotenv import load_dotenv
import sys
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, TimeoutError

class my_llama:    
    def __init__(self, model='llama', config = None):
        with open('mixtral_key.yml', 'r') as f:
            key_config = yaml.safe_load(f)
        self.prompt_tokens = 0 
        self.completion_tokens = 0 
        self.embedding_tokens = 0
        self.total_tokens = 0
        self.max_response_tokens = 800
        
        os.environ['https_proxy'] = 'http://127.0.0.1:10809'
        os.environ['http_proxy'] = 'http://127.0.0.1:10809'
        if isinstance(key_config["key"], list):   
                key_list = key_config["key"]
                self.key = init_api_key_handling(key_list, "mixtral_apikey_manager.json")
                self.client = openai.OpenAI(
                    base_url="https://llama2-70b.lepton.run/api/v1/",
                    api_key=self.key
                )
        else:
            self.key = key_config["key"]
            self.client = openai.OpenAI(
                base_url="https://llama2-70b.lepton.run/api/v1/",
                api_key=key_config["key"]
            )
        pass
        

    def compute_cost(self):
        price = 0.8
        # output_price = 0.002
        # embedding_price = 0.0001
        rate = 7.18
        # return "cost: not know"
        cost = float(self.total_tokens) / 1000000 * price * rate
        return "input tokens:{}, output tokens:{}, embedding tokens:{}, total tokens:{}, cost:{}".format(
            self.prompt_tokens, self.completion_tokens, self.embedding_tokens, self.total_tokens, cost)

    def completions_process(self, responses, messages, temperature=1, n=1, top_p=1):
        this_responses = []
        start_time = time.time()
        for _ in range(n):
            this_responses.append("")
        completion = self.client.chat.completions.create(
            model="llama2-70b",
            messages=messages,
            stream=True,
            n=n,
            temperature=temperature,
            top_p=top_p,
            max_tokens=self.max_response_tokens
        )
        
        for chunk in completion:
            last_chunk = chunk
            if not chunk.choices:
                continue
            for choice in chunk.choices:
                content = choice.delta.content
                if content != None:
                    this_responses[choice.index] += content
        end_time = time.time()
        consume_time = end_time - start_time
        print(f"{consume_time} ", end="")
        self.completion_tokens += last_chunk.model_extra["usage"]["completion_tokens"]
        self.prompt_tokens += last_chunk.model_extra["usage"]["prompt_tokens"]
        self.total_tokens += last_chunk.model_extra["usage"]["total_tokens"]
        responses += this_responses
        return responses

    def query(self, messages, temperature=1, n=1, top_p=1):
        try:
            all_n = n
            single_n = 1
            responses = []
            print("Time Now: {} ".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), end="")
            while all_n > 0:
                this_n = single_n if all_n > single_n else all_n
                all_n -= this_n
                print(f"{all_n} ", end="")
                sys.stdout.flush() 
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(self.completions_process, responses, messages, temperature, this_n, top_p)
                    try:
                        responses = future.result(timeout=180) 
                    except TimeoutError:
                        err = "chat.completions timeout err"
                        print(err)
                        time.sleep(this_n * 6.1)
                        return (False, f'Llama2 API Err: {err}'), err
                time.sleep(this_n * 6.1)
            print("")
            return (True, f''), responses
        except Exception as err:
            print(err)
            return (False, f'Llama2 API Err: {err}'), err

    
    def get_config(self):
        config = dict()
        config["max_seq_len"]  = self.max_seq_len
        config["temperature"] = self.temperature
        config["top_p"] = self.top_p
        return config
    
    def query_case(self, case, temperature, n, top_p):
        messages = []
        if "system-prompt" in case:
            system_prompt = case["system-prompt"]
            messages.append({'role': "system", 'content': system_prompt})
        if "in-context" in case:
            IC_list = case["in-context"]
            for shot in IC_list:
                shot_q = shot[0]
                shot_a = shot[1]
                messages.append({'role': "user", 'content': shot_q})
                messages.append({'role': "assistant", 'content': shot_a})
        question = case["question"]
        messages.append({'role': "user", 'content': question})
        
        retval, responses =  self.query(messages, temperature, n, top_p)
        response_content = []
        if retval[0]:
            for response in responses:
                response_content.append({"role":"assistent", "content": response})
            messages.append(response_content)  # the gemini format is "model" and "parts". This aims to use unique format in our program
        case["messages"] = messages
        return retval, messages

    def _query_and_append(self, single_query, temperature, n, top_p):
        err_count = 0
        while True:
            if isinstance(single_query, dict):  # case
                retval, messages = self.query_case(single_query, temperature, n, top_p)
            else:  # messages
                retval, responses = self.query(single_query, temperature, n, top_p)
                if retval[0]:
                    response_content = []
                    for response in responses:
                        response_content.append({"role":"assistent", "content": response})
                    single_query.append(response_content)  # the gemini format is "model" and "parts". This aims to use unique format in our program
                    return
            if retval[0]:
                return
            # tokens = self.num_tokens_from_messages(messages)
            # if tokens >= self.max_prompt_tokens:
            #     messages.append([{'role': "assistant", 'content': f"error:{retval}"}])
            #     break
            if "This model's maximum context length is 4097 tokens. However, your messages resulted" in retval[1]:
                messages.append([{'role': "assistant", 'content': f"error:{retval}"}])
                break
            err_count += 1
            if err_count == 20:
                messages.append([{'role': "assistant", 'content': f"error:{retval}"}])
                break
    
    def query_case_batch(self, cases, temperature=1, n=1, top_p=1):
        # with concurrent.futures.ThreadPoolExecutor() as executor:
        #     future_to_case = {executor.submit(self._query_and_append, case, temperature, n, top_p): case for case in
        #                       cases}
        #     for future in concurrent.futures.as_completed(future_to_case):
        #         future.result()
        for case in cases:
            self._query_and_append(case, temperature, n, top_p)
        return

    
    def query_n_case(self, n_case, c_reason, temperature=1, top_p=1):
    
        for i in range(len(n_case)):
            self._query_and_append(n_case[i], temperature, c_reason[i], top_p)
        return
        # with concurrent.futures.ThreadPoolExecutor() as executor:
        #     future_to_case = {executor.submit(self._query_and_append, n_case[i], temperature, c_reason[i], top_p): n_case[i]
        #                       for i in range(len(n_case))}
        #     for future in concurrent.futures.as_completed(future_to_case):
        #         future.result()
        # return

    def query_messages_batch(self, messages_batch, temperature=1, n=1, top_p=1):
        # with concurrent.futures.ThreadPoolExecutor() as executor:
        for messages in messages_batch:
            self._query_and_append(messages, temperature, n, top_p)
        return
