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

    def query(self, messages, temperature, n):
        if self.api == 'openai':
            try:
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=messages,
                    # stream=True,
                    temperature=temperature,
                    n=n,
                )
                self.completion_tokens += response["usage"]["completion_tokens"]
                self.prompt_tokens += response["usage"]["prompt_tokens"]
                self.total_tokens += response["usage"]["total_tokens"]
                # completion = {'role': {}, 'content': {}}
                # for event in response:
                #     if event['choices'][0]['finish_reason'] == 'stop':
                #         break
                #     for delta_k, delta_v in event['choices'][0]['delta'].items():
                #         completion[delta_k] += delta_v
                # 获取同个prompt的多次回答
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
        rate = 7.18
        cost = float(self.prompt_tokens) / 1000 * input_price * rate + \
               float(self.completion_tokens) / 1000 * output_price * rate
        return "input tokens:{}, output tokens:{}, total tokens:{}, total cost:{:.2f}".format(
            self.prompt_tokens, self.completion_tokens, self.total_tokens, cost)

    def query_case(self, case, temperature, n):
        messages = []
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
        return self.query(messages, temperature, n), messages

    def query_and_append(self, case, temperature, n):
        while True:
            retval, _ = self.query_case(case, temperature, n)
            if retval[0]:
                return
            time.sleep(1)

    def query_batch(self, cases, temperature, n):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_case = {executor.submit(self.query_and_append, case, temperature, n): case for case in cases}
            for future in concurrent.futures.as_completed(future_to_case):
                future.result()
        return
