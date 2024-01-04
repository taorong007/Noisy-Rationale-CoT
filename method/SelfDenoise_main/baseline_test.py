import numpy as np
from .code.utils.mask import mask_sentence, mask_forbidden_index
import copy
from .code.old_code.denoiser import chatgpt_cli
import re

class SimpleArgs:
    def __init__(self,  denoise_method, mask_word, sparse_mask_rate):
        self.denoise_method = denoise_method
        self.sparse_mask_rate = sparse_mask_rate
        self.mask_word = mask_word

class SelfDenoise:
    def __init__(self, n_reason) -> None:
        self.args = SimpleArgs(denoise_method="chatgpt_single_by_model", mask_word ="###", sparse_mask_rate=0.1)
        self.mask_token ="<mask>"
        self.n_reason = n_reason
        
    def certify(self, case_batch, model, **kwargs):
        args = self.args
        # log_file = open(os.path.join(args.save_path,'log.txt'),'w+')
        chatgpt_cli.set_mask_word(args.mask_word)
        all_cases = []
        for case in case_batch:
            shots = case["in-context"]
            all_shots = []
            qa_prompts = []
            for shot in shots:
                # question = shot[0]
                answer = copy.deepcopy(shot[1])
                # modifiable_q = question[0:question.find("Please reason it step by step")]
                # tail_q = question[question.find("Please reason it step by step"):]
                # tmp_sentence = mask_sentence(modifiable_q, args.sparse_mask_rate, self.mask_token, 1, False, random_probs=None)
                # shot[0] = tmp_sentence[0] + tail_q
                
                modifiable_a = answer[0:answer.rfind(".")+1]
                tail_a = answer[answer.rfind(".")+1:]
                tmp_sentence = mask_sentence(modifiable_a, args.sparse_mask_rate, self.mask_token, 1, False, random_probs=None)
                answer = tmp_sentence[0] + tail_a
                qa_prompts.append(f"User: {shot[0].replace(self.mask_token, args.mask_word)}\n" + f"Assistant: {answer.replace(self.mask_token, args.mask_word)}")
            
            
            sentences_list = chatgpt_cli.get_batch_response_by_model(qa_prompts, model, self.n_reason)
            n_shot_list = []
            for shot, sentences in zip(shots, sentences_list):
                n_shot = []
                for sentence in sentences:
                    new_shot = []
                    user_text_match = re.search(r'[Uu]ser:(.*?)\n', sentence)
                    assistant_text_match = re.search(r'[Aa]ssistant:(.*)', sentence)
                    user_text = user_text_match.group(1) if user_text_match else shot[0]
                    assistant_text = assistant_text_match.group(1) if assistant_text_match else sentence
                    new_shot.append(user_text)
                    new_shot.append(assistant_text)
                    n_shot.append(new_shot)
                n_shot_list.append(n_shot)
            for context in zip(*n_shot_list):
                new_case = copy.deepcopy(case)
                new_case['in-context'] = list(context)
                all_cases.append(new_case)
        model.query_batch(cases = all_cases, temperature = 1, n = 1)
        return all_cases
          