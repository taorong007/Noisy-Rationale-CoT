import numpy as np
import random

class base_math:
    """
    Processor of a base caculation dataset.
    
    Attributes:
        if_COT: A boolean indicating if Chain-of-Thought (COT) or in-context learning is used. Default is False.
        n_shots: The number of shots. Default is 0.
        ex_shots: The number of excellent shots. Default is 0.
        error_shots: The number of error shots. Default is 0.
        error_type: The type of error, can be "misunderstanding_error" or "miscalculation_error". Default is "miscalculation".
        prefix_context: A boolean indicating if prefix context is used. Default is False.
        config: A dictionary containing all the attribute values. If provided, values in the config dictionary will be used to overwrite the following parameters.
        base: The base number for calculations. Default is 9.
        
    """
    def __init__(self, if_COT = False, n_shots=0, ex_shots = 0, error_shots=0, error_type="miscalculation", prefix_context = False, config: dict = None, base=9) -> None:
        if config is not None:
            self.base = config["base"]
        else:
            self.base = base
            
        self.if_COT = if_COT
        self.n_shots = n_shots
        self.ex_shots = ex_shots
        self.error_shots = error_shots
        self.prefix_context = prefix_context
        self.error_type = error_type


    def get_label(self, expr):
        base = self.base
        lhs, rhs = expr.split("+")
        lhs_base10 = int(lhs, base)
        rhs_base10 = int(rhs, base)
        sum_base10 = lhs_base10 + rhs_base10
        return np.base_repr(sum_base10, base)

    def answer(self,expr):
        base = self.base
        lhs, rhs = expr.split("+")
        lt, lo = lhs  # tens, ones
        rt, ro = rhs
        ones_sum = self.get_label(f"{lo}+{ro}")
        carry_over = len(ones_sum) > 1
        tens_sum_wo_carry = self.get_label(f"{lt}+{rt}")
        if carry_over:
            assert ones_sum[0] == "1"
            tens_sum_w_carry = self.get_label(f"{tens_sum_wo_carry}+1")
        else:
            tens_sum_w_carry = tens_sum_wo_carry
        assert self.get_label(expr) == tens_sum_w_carry + ones_sum[-1:]

        ret = f"We add the ones digits first. In base-{base}, {lo}+{ro}={ones_sum}. So the ones digit of the final sum is {ones_sum[-1:]}. "
        if carry_over:
            ret += f"We need to carry over the 1 to the tens place. "
        else:
            ret += f"We do not need to carry any digits over. "
        ret += f"Then we add the tens digits. In base-{base}, {lt}+{rt}={tens_sum_wo_carry}. "
        if carry_over:
            ret += f"Since we carried over the 1, {tens_sum_wo_carry}+1={tens_sum_w_carry}. "
        if len(tens_sum_w_carry) == 1:
            ret += f"So the tens digit of the final sum is {tens_sum_w_carry}. "
        else:
            ret += f"So the hundreds and tens digits of the final sum are {tens_sum_w_carry}. "
        ret += f"Putting the digits of the final sum together, we get\n Answer:\\boxed{{{tens_sum_w_carry}{ones_sum[-1:]}}}."
        return ret
    
    
    def misunderstanding_error_answer(self):
        error_answer = [["You are a mathematician. Assuming that all numbers are in base-9 where the digits are \"012345678\", what is 33+68? End the response with the result in \"Answer:\\boxe {result}\".", "To add 33 and 68 in base-9, we can perform the addition as follows:\n\n        33\n      + 68\n    ------\n       101\n\nTherefore, the result is 101 in base-9.\n\nAnswer: $\\boxed{101}$."],
                        ["You are a mathematician. Assuming that all numbers are in base-9 where the digits are \"012345678\", what is 67+54? End the response with the result in \"Answer:\\boxed{result}\".", "In base-9, we have the following digit values:\n$6\\cdot9^1+7\\cdot9^0=54+7=61$ and $5\\cdot9^1+4\\cdot9^0=45+4=49$.\n\nAdding them together, we have $61+49=110$.\n\nTherefore, $67+54=110$.\n\nAnswer: $\\boxed{110}$."],
                        ["You are a mathematician. Assuming that all numbers are in base-9 where the digits are \"012345678\", what is 65+30? End the response with the result in \"Answer:\\boxed{result}\".","To add 65 and 30 in base-9, we can follow the same procedure as in base-10. Starting from the rightmost digit, we have:\n\n      5\n    + 0\n    -----\n      5\n\nThen, we move to the leftmost digit:\n\n      6\n    + 3\n    -----\n    1(10)\n\nSince 6 + 3 = 9, which is equivalent to 1 in base-9, we write down the 0 and carry over 1 to the left. So, the final result is 10 in base-9.\n\nAnswer: $\\boxed{10}$."]
                        ]
        
        return error_answer[:self.error_shots]

    def miscalculation_error_answer(self):
        error_answer = [["You are a mathematician. Assuming that all numbers are in base-9 where the digits are \"012345678\", what is 33+68? End the response with the result in \"Answer:\\boxe {result}\".","We have 3 + 8 = 11 in base-10. Since we're in base-9, that exceeds the maximum value of 8 for a single digit. 11 mod 9 = 2, so the digit is 2 and the carry is 1. We have 3 + 6 = 9 in base 10. 9 mod 9 = 0, so the digit is 0 and the carry is 1. A leading digit 1. So the answer is 102. Answer://box{102}"],
                        ["You are a mathematician. Assuming that all numbers are in base-9 where the digits are \"012345678\", what is 67+54? End the response with the result in \"Answer:\\boxed{result}\".", "We have 7 + 4 = 11 in base-10. Since we're in base-9, that exceeds the maximum value of 8 for a single digit. 11 mod 9 = 2, so the digit is 2 and the carry is 1. We have 6 + 5 = 11 in base 10. 11 mod 9 = 2, so the digit is 2 and the carry is 1. A leading digit 1. So the answer is 122. Answer://box{122}"],
                        ["You are a mathematician. Assuming that all numbers are in base-9 where the digits are \"012345678\", what is 65+30? End the response with the result in \"Answer:\\boxed{result}\".","We have 5 + 3 = 8 in base-10. Since we're in base-9, that exceeds the maximum value of 8 for a single digit. 8 mod 9 = 8, so the digit is 5 and the carry is 0. We have 6 + 3 + 0 = 9 in base 10. 9 mod 9 = 0, so the digit is 0 and the carry is 1. A leading digit 1. So the answer is 105. Answer://box{108}"]]
        
        return error_answer[:self.error_shots]
            

    def get_question(self, expr):
        cot = self.if_COT
        base = self.base
        digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        if cot:
            return f"You are a mathematician. Assuming that all numbers are in base-{base} where the digits are \"{digits[:base]}\", what is {expr}? Let's think step by step, and end the response with the result in \"Answer:\\boxed{{result}}\"."
        else:
            return f"You are a mathematician. Assuming that all numbers are in base-{base} where the digits are \"{digits[:base]}\", what is {expr}? End the response with the result in \"Answer:\\boxed{{result}}\"."
        # return "in base-{}, what is {}".format(base, expr)
        
    def ex_answer(self, expr):
        digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        base = self.base
        lhs, rhs = expr.split("+")
        lt, lo = lhs  # tens, ones
        rt, ro = rhs
        ones_sum = self.get_label(f"{lo}+{ro}")
        carry_over = len(ones_sum) > 1
        tens_sum_wo_carry = self.get_label(f"{lt}+{rt}")
        if carry_over:
            ones_carry_digit = 1
            assert ones_sum[0] == "1"
            tens_sum_w_carry = self.get_label(f"{tens_sum_wo_carry}+1")
        else:
            ones_carry_digit = 0
            tens_sum_w_carry = tens_sum_wo_carry
        assert self.get_label(expr) == tens_sum_w_carry + ones_sum[-1:]
        tens_carry_over = len(tens_sum_w_carry) > 1
        tens_carry_digit = 1 if tens_carry_over else 0
        
        ret = f"We have {lo} + {ro} = {int(lo) + int(ro)} in base-10. Since we're in base-{base}, that exceeds the maximum value of {digits[base-1]} for a single digit. {int(lo) + int(ro)} mod {base} = {ones_sum[-1]}, so the digit is {ones_sum[-1]} and the carry is {ones_carry_digit}. We have {lt} + {rt} + {ones_carry_digit} = {int(lt) + int(rt) + ones_carry_digit} in base 10. {int(lt) + int(rt) + ones_carry_digit} mod {base} = {tens_sum_w_carry[-1]}, so the digit is {tens_sum_w_carry[-1]} and the carry is {tens_carry_digit}. A leading digit {tens_carry_digit}. So the answer is {self.get_label(expr)}. Answer:\\box{{{self.get_label(expr)}}}"
        return ret
        

    def get_prompt_case(self, expr):
        n_shots = self.n_shots
        case = dict()
        prefix = ''
        if n_shots > 0:
            COT = []
            expr, demos = expr.split("\t")
            shots = demos.split(",")[:n_shots - self.ex_shots]
            assert len(shots) == n_shots - self.ex_shots
            for shot in shots:
                COT_q = self.get_question(shot)
                COT_a =  self.answer(shot)
                COT.append([COT_q, COT_a])
            ex_shots = demos.split(",")[n_shots - self.ex_shots:n_shots]
            assert len(ex_shots) == self.ex_shots
            for shot in ex_shots:
                COT_q = self.get_question(shot)
                COT_a =  self.ex_answer(shot)
                COT.append([COT_q, COT_a])
            random.shuffle(COT)
            
            if self.error_shots > 0:
                COT = COT[:n_shots - self.error_shots]
                if self.error_type == "misunderstanding":
                    error_shots = self.misunderstanding_error_answer()
                else:
                    error_shots = self.miscalculation_error_answer()
                COT = COT + error_shots
                random.shuffle(COT)
            if self.prefix_context:
                for shot in COT:
                    prefix += "user:{}\nassistant:{}\n".format(shot[0], shot[1])
            else:    
                case["COT"] = COT
        question = self.get_question(expr)
        real_answer = self.get_label(expr)
        case["question"] = prefix + question
        case["label"] = real_answer 
        return case
        
        
    def load_data(self):
        data_file = "./data/base_math/icl/base{}.txt".format(self.base)
        dataset = [line.strip() for line in open(data_file)]
        return dataset


