import openai
import requests
import yaml
import concurrent.futures
import time
import tiktoken
from ..multiple_key import init_api_key_handling


class my_gpt:
    def __init__(self, model='gpt-3.5-turbo-0613', config: dict = None, api="openai") -> None:
        if config != None:
            api = config["api"] if "api" in config else api
            # temperature = config["temperature"] if "temperature" in config else temperature
            # run_times = config["run_times"] if "run_times" in config else run_times
        self.api = api
        self.model = model
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.embedding_tokens = 0
        self.total_tokens = 0
        self.max_prompt_tokens = 4096
        self.max_response_tokens = 1000
        # self.temperature = temperature
        # self.run_times = run_times

        if api == 'openai':
            with open('openai_key.yml', 'r') as f:
                openai_config = yaml.safe_load(f)
              
            if isinstance(openai_config["key"], list):   
                key_list = openai_config["key"]
                openai.api_key = init_api_key_handling(key_list) 
            else:
                openai.api_key = openai_config["key"]
            if "api_base" in openai_config:
                openai.api_base = openai_config["api_base"]
            # openai.api_base = "https://openkey.cloud/v1"
        elif api == 'hkbu':
            with open('hkbu_key.yml', 'r') as f:
                key_config = yaml.safe_load(f)
            apiKey = key_config["key"]
            basicUrl = "https://chatgpt.hkbu.edu.hk/general/rest"
            modelName = "gpt-35-turbo-16k"
            apiVersion = "2023-08-01-preview"
            self.url = basicUrl + "/deployments/" + modelName + "/chat/completions/?api-version=" + apiVersion
            self.headers = {'Content-Type': 'application/json', 'api-key': apiKey}
        else:
            raise "Api not support: {}".format(api)
        pass

    # def get_config(self):
    #     config = dict()
    #     config["api"] = self.api
    #     config["temperature"] = self.temperature
    #     config["model"] = self.model
    #     config["n"] = self.run_times
    #     return config

    def chat(self, single_chat):
        messages = []
        messages.append({'role': "user", 'content': single_chat})
        retval, error = self.query(messages)
        if retval:
            return messages[-1]["content"]
        else:
            return f"error:{error}"

    def get_embedding(self, text, model="text-embedding-ada-002"):
        text = text.replace("\n", " ")
        text = text.replace("\\", "")
        response = openai.Embedding.create(
            input=text,
            model=model
        )
        embedding = response['data'][0]['embedding']
        self.embedding_tokens += response['usage']['prompt_tokens']
        return embedding

    def query(self, messages, temperature=1, n=1, top_p=1):
        if self.api == 'openai':
            try:
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    n=n,
                    top_p=top_p,
                    max_tokens=self.max_response_tokens
                )
                self.completion_tokens += response["usage"]["completion_tokens"]
                self.prompt_tokens += response["usage"]["prompt_tokens"]
                self.total_tokens += response["usage"]["total_tokens"]
                completions = []
                for choice in response['choices']:
                    message = choice['message']
                    completion = dict()
                    completion['role'] = message['role']
                    completion['content'] = message['content']
                    completions.append(completion)
                messages.append(completions)
                return (True, ''), messages
            except Exception as err:
                print(err)
                return (False, f'OpenAI API Err: {err}'), messages
        else:
            payload = {'messages': messages}
            response = requests.post(self.url, json=payload, headers=self.headers)
            if response.status_code == 200:
                data = response.json()
                # if data['choices'][0]['finish_reason'] == 'stop':
                #     pass
                messages.append([data['choices'][0]["message"]])
                return (True, ''), messages
            else:
                return (False, f'Error: {response}'), messages

    def compute_cost(self):
        input_price = 0.0015
        output_price = 0.002
        embedding_price = 0.0001
        rate = 7.18
        cost = float(self.prompt_tokens) / 1000 * input_price * rate + \
               float(self.completion_tokens) / 1000 * output_price * rate + \
               float(self.embedding_tokens) / 1000 * embedding_price * rate
        return "input tokens:{}, output tokens:{}, embedding tokens:{}, total tokens:{}, total cost:{:.2f}".format(
            self.prompt_tokens, self.completion_tokens, self.embedding_tokens, self.total_tokens, cost)

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
        case["messages"] = messages
        return self.query(messages, temperature, n, top_p)

    def _query_and_append(self, single_query, temperature, n, top_p):
        err_count = 0
        while True:
            if isinstance(single_query, dict):  # case
                retval, messages = self.query_case(single_query, temperature, n, top_p)
            else:  # messages
                retval, messages = self.query(single_query, temperature, n, top_p)
            if retval[0]:
                return
            tokens = self.num_tokens_from_messages(messages)
            if tokens >= self.max_prompt_tokens:
                messages.append([{'role': "assistant", 'content': f"error:{retval}"}])
                break
            if "This model's maximum context length is 4097 tokens. However, your messages resulted" in retval[1]:
                messages.append([{'role': "assistant", 'content': f"error:{retval}"}])
                break
            err_count += 1
            if err_count == 10:
                messages.append([{'role': "assistant", 'content': f"error:{retval}"}])
                break
            time.sleep(1)

    def query_case_batch(self, cases, temperature=1, n=1, top_p=1):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_case = {executor.submit(self._query_and_append, case, temperature, n, top_p): case for case in
                              cases}
            for future in concurrent.futures.as_completed(future_to_case):
                future.result()
        return

    def query_n_case(self, n_case, c_reason, temperature=1, top_p=1):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_case = {executor.submit(self._query_and_append, n_case[i], temperature, c_reason[i], top_p): n_case[i]
                              for i in range(len(n_case))}
            for future in concurrent.futures.as_completed(future_to_case):
                future.result()
        return

    def query_messages_batch(self, messages_batch, temperature=1, n=1, top_p=1):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_case = {executor.submit(self._query_and_append, messages, temperature, n, top_p): messages for
                              messages in messages_batch}
            for future in concurrent.futures.as_completed(future_to_case):
                future.result()
        return

    def compute_prompt_token_by_case(self, case):
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
        return self.num_tokens_from_messages(messages)

    def num_tokens_from_messages(self, messages):
        """Return the number of tokens used by a list of messages."""
        model = self.model
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            print("Warning: model not found. Using cl100k_base encoding.")
            encoding = tiktoken.get_encoding("cl100k_base")
        if model in {
            "gpt-3.5-turbo-0613",
            "gpt-3.5-turbo-16k-0613",
            "gpt-4-0314",
            "gpt-4-32k-0314",
            "gpt-4-0613",
            "gpt-4-32k-0613",
        }:
            tokens_per_message = 3
            tokens_per_name = 1
        elif model == "gpt-3.5-turbo-0301":
            tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
            tokens_per_name = -1  # if there's a name, the role is omitted
        elif "gpt-3.5-turbo" in model:
            # print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
            return self.num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
        elif "gpt-4" in model:
            # print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
            return self.num_tokens_from_messages(messages, model="gpt-4-0613")
        else:
            raise NotImplementedError(
                f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
            )
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        return num_tokens
