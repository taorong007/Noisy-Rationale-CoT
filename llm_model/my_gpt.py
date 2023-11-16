import openai
import requests
import yaml
import concurrent.futures
import time



class my_gpt:
    def __init__(self, model = 'gpt-3.5-turbo', config : dict =  None, api = "openai") -> None:
        if config != None:
            api = config["api"] if "api" in config else api
        self.api = api
        self.model = model
        if api == 'openai':
            with open('openai_key.yml', 'r') as f:
                key_config = yaml.safe_load(f)
            openai.api_key = key_config["key"]
            openai.api_base = "https://openkey.cloud/v1"
        elif api == 'hkbu':
            with open('hkbu_key.yml', 'r') as f:
                key_config = yaml.safe_load(f)
            apiKey = key_config["key"]
            basicUrl = "https://chatgpt.hkbu.edu.hk/general/rest"
            modelName = "gpt-35-turbo-16k"
            apiVersion = "2023-08-01-preview"
            self.url = basicUrl + "/deployments/" + modelName + "/chat/completions/?api-version=" + apiVersion
            self.headers = { 'Content-Type': 'application/json', 'api-key': apiKey }
        else:
            raise "Api not support: {}".format(api)
        pass
    
    def query(self, messages):
        if self.api == 'openai':
            try:
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=messages,
                    stream=True,
                )
                completion = {'role': '', 'content': ''}
                for event in response:
                    if event['choices'][0]['finish_reason'] == 'stop':
                        break
                    for delta_k, delta_v in event['choices'][0]['delta'].items():
                        completion[delta_k] += delta_v
                messages.append(completion)
                return (True, '')
            except Exception as err:
                return (False, f'OpenAI API 异常: {err}')
        else:
            payload = { 'messages': messages }
            response = requests.post(self.url, json=payload, headers=self.headers)
            if response.status_code == 200:
                data = response.json()
                # if data['choices'][0]['finish_reason'] == 'stop':
                #     pass
                messages.append(data['choices'][0]["message"])
                return (True, '')
            else:
                return (False, f'Error: {response}')
        
        
    def query_case(self, case):
        messages = []
        if "COT" in case:
            COT_list = case["COT"]
            for shot in COT_list:
                cot_q = shot[0]
                cot_a = shot[1]
                messages.append({'role':"user", 'content':cot_q})
                messages.append({'role':"assistant", 'content':cot_a})
        question = case["question"]
        messages.append({'role':"user", 'content': question})
        case["response"] = messages
        return self.query(messages), messages
    
    def query_and_append(self, case):
        while True:
            retval, response = self.query_case(case)
            if retval[0]:
                time.sleep(1)
                return response, case["label"]
    
    def query_batch(self, cases):
        responses = []
        labels = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_case = {executor.submit(self.query_and_append, case): case for case in cases}
            for future in concurrent.futures.as_completed(future_to_case):
                future.result()
            
        return 
    