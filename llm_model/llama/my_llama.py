from typing import List, Optional
from . import Dialog, Llama
class my_llama:
    llama_generator = None
    def __init__(self,
        config: dict = None,
        ckpt_dir: str = None,
        tokenizer_path: str = None,
        max_seq_len: int = 512,
        max_batch_size: int = 8,
        temperature: float = 0.6, 
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None
    ):
        """
        Entry point of the program for generating text using a pretrained model.

        Args:
            ckpt_dir (str): The directory containing checkpoint files for the pretrained model.
            tokenizer_path (str): The path to the tokenizer model used for text encoding/decoding.
            temperature (float, optional): The temperature value for controlling randomness in generation.
                Defaults to 0.6.
            top_p (float, optional): The top-p sampling parameter for controlling diversity in generation.
                Defaults to 0.9.
            max_seq_len (int, optional): The maximum sequence length for input prompts. Defaults to 512.
            max_batch_size (int, optional): The maximum batch size for generating sequences. Defaults to 8.
            max_gen_len (int, optional): The maximum length of generated sequences. If None, it will be
                set to the model's max sequence length. Defaults to None.
        """
        if config:
            if "ckpt_dir" in config:
                ckpt_dir = config["ckpt_dir"]
            if "tokenizer_path" in config:
                tokenizer_path = config["tokenizer_path"]
            if "max_seq_len" in config:
                max_seq_len = config["max_seq_len"]
            if "max_batch_size" in config:
                max_batch_size = config["max_batch_size"]
        self.llama_generator = Llama.build(
            ckpt_dir=ckpt_dir,
            tokenizer_path=tokenizer_path,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
        )
        self.temperature = temperature
        self.top_p =top_p
        self.max_gen_len = max_gen_len
        return
    # def _chat_with_llama(self):
    #     batch_size = self._batch_size
    #     run_times = self._run_times
    #     qa_list = [self._case_list[i:i+batch_size] for i in range(0, len(self._case_list), batch_size)]
    #     for qa_batch in qa_list:
    #         messages_batch = []
    #         label_batch = []
    #         for qa in qa_batch:
    #             question = qa["question"]
    #             label = qa["label"]
    #             messages = [{'role': 'user','content': question}]
    #             if self._if_in_context:
    #                 messages = in_context + messages
    #             messages_batch.append(messages)
    #             label_batch.append(label)
    #         responses_batch = self._GMS_llama.llama_chat(messages_batch)
    #         self._response_process(messages_batch, responses_batch, label_batch)
    #     self._answers_list = [self._answers_list[i:i+run_times] for i in range(0, len(self._answers_list), run_times)]
    #     self._contents_list = [self._contents_list[i:i+run_times] for i in range(0, len(self._contents_list), run_times)]
    
    def get_config(self):
        config = dict()
        config["max_seq_len"]  = self.max_seq_len
        config["temperature"] = self.temperature
        config["top_p"] = self.top_p
        return config
    
    def query(self, dialogs : list = [[{"role": "user", "content":"hello"}]]):
        responses = self.llama_generator.chat_completion(
            dialogs,  # type: ignore
            max_gen_len=self.max_gen_len,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        return responses
    
    def query_case(self, case):
        messages = []
        if "in-context" in case:
            IC_list = case["in-context"]
            for shot in IC_list:
                shot_q = shot[0]
                shot_a = shot[1]
                messages.append({'role':"user", 'content':shot_q})
                messages.append({'role':"assistant", 'content':shot_a})
        question = case["question"]
        messages.append({'role':"user", 'content': question})
        case["messages"] = messages
        responses = self.query([messages])
        messages.append({'role':"assistant", 'content':responses[0]["generation"]["content"]})
        return
    
    def query_case_batch(self, cases):
        messages_list = []
        for case in cases:
            messages = []
            if "in-context" in case:
                IC_list = case["in-context"]
                for shot in IC_list:
                    shot_q = shot[0]
                    shot_a = shot[1]
                    messages.append({'role':"user", 'content':shot_q})
                    messages.append({'role':"assistant", 'content':shot_a})
            question = case["question"]
            messages.append({'role':"user", 'content': question})
            messages_list.append(messages)
            case["messages"] = messages
        responses = self.query(messages_list)
        assert len(cases) == len(responses)
        for case, response in zip(cases, responses):
            case["messages"].append({'role':"assistant", 'content':response["generation"]["content"]})
        return 