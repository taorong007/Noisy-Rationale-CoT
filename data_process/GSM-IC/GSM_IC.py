class GSM_IC:
    def __init__(self) -> None:
        pass
    def get_case(self, case):
        original_question = case["original_question"]
        label = case["answer"]
        new_qustion = case["new_question"]
    
        if not self._if_noise:
            question = original_question + self._suffix_prompt
        else:
            question = new_qustion + self._suffix_prompt
        # self._case_list.append({"question": question, "label": label})