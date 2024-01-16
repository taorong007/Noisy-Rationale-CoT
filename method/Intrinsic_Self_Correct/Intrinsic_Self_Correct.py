import copy
# from llm_model.my_gpt.my_gpt import my_gpt

def Intrinsic_Self_Correct(case_batch, model, dataset_name, n_reason):
    # model.query_case_batch(case_batch, n = n_reason)
    model.query_case_batch(case_batch, n = 1)
    
    prompt1 = "Review your previous answer and find problems with your answer."
    prompt2 = "Based on the problems you found, improve vour answer. Please reiterate your answer, with your final answer "
    if dataset_name == "base_math":
        prompt2 += "in the format of \"Answer:\\boxed{{result}}\""
    elif dataset_name == "SCAN":
        prompt2 += "in the format of \"So, final answer is OUT: <action sequence>\""
    elif dataset_name == "family_relation":
        prompt2 += "in the format of \"Answer: {{relation}}\""
    else:
        prompt2 += "a single numerical number, in the form \\boxed{{answer}}"
    
    messages_list = []
    for case in case_batch:
        responses = case["messages"][-1]
        for response in responses:
            messages = copy.deepcopy(case["messages"][:-1])
            messages.append(response)
            messages.append({'role': "user", 'content': prompt1})
            messages_list.append(messages)
    model.query_messages_batch(messages_list, n = 1)
    
    for messages in messages_list:
        messages[-1] = messages[-1][0]
        messages.append({'role': "user", 'content': prompt2})
    model.query_messages_batch(messages_list, n = n_reason)
    
    
    index = 0
    for case in case_batch:
        # for response in case["messages"][-1]:
        #     response["content"] =  messages_list[index][-1][0]["content"]
        #     index += 1
        case["messages"][-1] = messages_list[index][-1]
        index += 1
        
    return case_batch