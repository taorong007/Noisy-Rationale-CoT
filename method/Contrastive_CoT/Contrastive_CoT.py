
def contrastive_CoT(postive_QA, case):
    
    prompt = ""
    
    if "in-context" in case and case["in-context"] != None:
        shots = case["in-context"]
    new_case = dict()
    new_case["question"] = case["question"]
    new_case["label"] = case["label"]
    return new_case