import openai
import requests
import yaml
import concurrent.futures
import time


class my_gpt:
    def __init__(self, model='gpt-3.5-turbo', config: dict = None, api="openai", temperature=1, run_times=1) -> None:
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
        # self.temperature = temperature
        # self.run_times = run_times

        if api == 'openai':
            with open('openai_key.yml', 'r') as f:
                openai_config = yaml.safe_load(f)
            openai.api_key = openai_config["key"]
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

    def query(self, messages, temperature, n, top_p):
        if self.api == 'openai':
            try:
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    n=n,
                    top_p=top_p,
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
                return (True, '')
            except Exception as err:
                print(err)
                return (False, f'OpenAI API 异常: {err}')
        else:
            payload = {'messages': messages}
            response = requests.post(self.url, json=payload, headers=self.headers)
            if response.status_code == 200:
                data = response.json()
                # if data['choices'][0]['finish_reason'] == 'stop':
                #     pass
                messages.append(data['choices'][0]["message"])
                return (True, '')
            else:
                return (False, f'Error: {response}')

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
        return self.query(messages, temperature, n, top_p), messages

    def query_and_append(self, case, temperature, n, top_p):
        while True:
            retval, messages = self.query_case(case, temperature, n, top_p)
            err_count = 0
            if retval[0]:
                return
            err_count += 1
            if err_count == 10:
                messages.append({'role': "assistant", 'content': f"error:{retval}"})
                break
            time.sleep(1)

    def query_batch(self, cases, temperature, n, top_p = 1):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_case = {executor.submit(self.query_and_append, case, temperature, n, top_p): case for case in cases}
            for future in concurrent.futures.as_completed(future_to_case):
                future.result()
        return
