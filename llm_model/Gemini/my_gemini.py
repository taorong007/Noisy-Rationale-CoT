import os
import yaml
from ..multiple_key import init_api_key_handling
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
import concurrent
import time


class my_gemini:
    def __init__(self, model='gemini-pro', config: dict = None) -> None:
        if model == 'gemini-pro':
            self.model = genai.GenerativeModel(model_name = "gemini-pro")
        else:
            raise ValueError(f"{model} is not supported")
        with open('gemini_key.yml', 'r') as f:
            gemini_config = yaml.safe_load(f)
            
        # os.environ['https_proxy'] = 'http://127.0.0.1:10809'
        # os.environ['http_proxy'] = 'http://127.0.0.1:10809'
        
        if isinstance(gemini_config["key"], list):   
            key_list = gemini_config["key"]
            genai.configure(api_key=init_api_key_handling(key_list, "gemini_apikey_manager.json") )
        else:
            genai.configure(api_key=gemini_config["key"])
        
        
        
    def generate_content(self, prompt_str, temperature, n, top_p):
        try:
            generation_config = GenerationConfig(
                candidate_count=1,  # So far, Only one candidate can be specified (Gemini)
                temperature=temperature,
                top_p=top_p
            )
            # response = self.model.generate_content(prompt_str, generation_config=generation_config)
            
            responses = []
            for _ in range(n):
                response = self.model.generate_content(prompt_str, generation_config=generation_config)
                responses.append(response.text)
                time.sleep(1)
            return (True, f''), responses
        except Exception as err:
            print(err)
            time.sleep(1)
            return (False, f'Gemini API Err: {err}'), err
        

    def query_case(self, case, temperature=1, n=1, top_p=1):
        prompt = ""
        if "system-prompt" in case:
            prompt += case["system-prompt"] + "\n"
        if "in-context" in case:
            shots = case["in-context"]
            for shot in shots:
                prompt += f"user: {shot[0]}\n"
                prompt += f"model: {shot[1]}\n"
        prompt += "user: {}\n".format(case["question"])
        retval, responses = self.generate_content(prompt_str=prompt, temperature=temperature, n=n, top_p=top_p)
        
        response_content = []
        if retval[0]:
            for response in responses:
                response_content.append({"role":"assistent", "content": response})
            messages = [{"role":"user", "content":prompt}, response_content]  # the gemini format is "model" and "parts". This aims to use unique format in our program
            case["messages"] = messages
        else:
            messages = []
        return retval, messages

    def _query_and_append(self, single_query, temperature=1, n=1, top_p=1):
        err_count = 0
        while True:
            # if isinstance(single_query, dict):  # case
            retval, messages = self.query_case(single_query, temperature, n, top_p)
            # else:  # messages
            #     retval, messages = self.query(single_query)
            if retval[0]:
                return
            if err_count > 30:
                return
        
    def query_case_batch(self, cases, temperature=1, n=1, top_p=1):
        if len(cases)> 1:
            raise ValueError("gemini test now does not support batch > 1")
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_case = {executor.submit(self._query_and_append, case, temperature, n, top_p): case for case in
                              cases}
            for future in concurrent.futures.as_completed(future_to_case):
                future.result()
        return