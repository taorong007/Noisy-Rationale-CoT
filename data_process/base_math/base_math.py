import numpy as np
import random

class base_math:
    """
    Processor of a base caculation dataset.
    
    Attributes:
        if_COT: A boolean indicating if Chain-of-Thought (COT) or in-context learning is used. Default is False.
        n_shots: The number of shots. Default is 0.
        ex_shots: The number of excellent shots. Default is 0.
        noisy_shots: The number of noisy shots. Default is 0.
        noisy_type: The type of noisy, can be "misunderstanding" or "miscalculation". Default is "miscalculation".
        prefix_context: A boolean indicating if prefix context is used. Default is False.
        config: A dictionary containing all the attribute values. If provided, values in the config dictionary will be used to overwrite the following parameters.
        base: The base number for calculations. Default is 9.
        
    """
    def __init__(self, if_COT = False, n_shots=0, ex_shots = 0, noisy_shots=0, noisy_type="miscalculation", prefix_context = False, config: dict = None, base=9) -> None:
        if config is not None:
            self.base = config["base"]
        else:
            self.base = base
            
        self.if_COT = if_COT
        self.n_shots = n_shots
        self.ex_shots = ex_shots
        self.noisy_shots = noisy_shots
        self.prefix_context = prefix_context
        self.noisy_type = noisy_type
        self.distracting_index = 0


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
    
    
    def misunderstanding_noisy_answer(self):
        noisy_answer = [["You are a mathematician. Assuming that all numbers are in base-9 where the digits are \"012345678\", what is 33+68? End the response with the result in \"Answer:\\boxe {result}\".", "To add 33 and 68 in base-9, we can perform the addition as follows:\n\n        33\n      + 68\n    ------\n       101\n\nTherefore, the result is 101 in base-9.\n\nAnswer: $\\boxed{101}$."],
                        ["You are a mathematician. Assuming that all numbers are in base-9 where the digits are \"012345678\", what is 67+54? End the response with the result in \"Answer:\\boxed{result}\".", "In base-9, we have the following digit values:\n$6\\cdot9^1+7\\cdot9^0=54+7=61$ and $5\\cdot9^1+4\\cdot9^0=45+4=49$.\n\nAdding them together, we have $61+49=110$.\n\nTherefore, $67+54=110$.\n\nAnswer: $\\boxed{110}$."],
                        ["You are a mathematician. Assuming that all numbers are in base-9 where the digits are \"012345678\", what is 65+30? End the response with the result in \"Answer:\\boxed{result}\".","To add 65 and 30 in base-9, we can follow the same procedure as in base-10. Starting from the rightmost digit, we have:\n\n      5\n    + 0\n    -----\n      5\n\nThen, we move to the leftmost digit:\n\n      6\n    + 3\n    -----\n    1(10)\n\nSince 6 + 3 = 9, which is equivalent to 1 in base-9, we write down the 0 and carry over 1 to the left. So, the final result is 10 in base-9.\n\nAnswer: $\\boxed{10}$."]
                        ]
        
        return noisy_answer[:self.noisy_shots]

    def miscalculation_noisy_answer(self):
        noisy_answer = [["You are a mathematician. Assuming that all numbers are in base-9 where the digits are \"012345678\", what is 33+68? End the response with the result in \"Answer:\\boxe {result}\".","We have 3 + 8 = 11 in base-10. Since we're in base-9, that exceeds the maximum value of 8 for a single digit. 11 mod 9 = 2, so the digit is 2 and the carry is 1. We have 3 + 6 = 9 in base 10. 9 mod 9 = 0, so the digit is 0 and the carry is 1. A leading digit 1. So the answer is 102. Answer://box{102}"],
                        ["You are a mathematician. Assuming that all numbers are in base-9 where the digits are \"012345678\", what is 67+54? End the response with the result in \"Answer:\\boxed{result}\".", "We have 7 + 4 = 11 in base-10. Since we're in base-9, that exceeds the maximum value of 8 for a single digit. 11 mod 9 = 2, so the digit is 2 and the carry is 1. We have 6 + 5 = 11 in base 10. 11 mod 9 = 2, so the digit is 2 and the carry is 1. A leading digit 1. So the answer is 122. Answer://box{122}"],
                        ["You are a mathematician. Assuming that all numbers are in base-9 where the digits are \"012345678\", what is 65+30? End the response with the result in \"Answer:\\boxed{result}\".","We have 5 + 3 = 8 in base-10. Since we're in base-9, that exceeds the maximum value of 8 for a single digit. 8 mod 9 = 8, so the digit is 5 and the carry is 0. We have 6 + 3 + 0 = 9 in base 10. 9 mod 9 = 0, so the digit is 0 and the carry is 1. A leading digit 1. So the answer is 105. Answer://box{108}"]]
        
        return noisy_answer[:self.noisy_shots]
    
    def distracting_answer(self, expr):
        self.distracting_index = self.distracting_index % 6
        irrelative_content = ["In base-9, the digits run from 0 to 8. If you're a fan of unique number systems, you might enjoy the Mayan numeral system, which is base-20 and uses a combination of dots, lines, and symbols to represent numbers, quite different from our familiar base-10 system. Fascinating, isn't it?",
                              "By the way, did you know that the word 'digit' comes from the Latin word 'digitus', which means finger? It's a hint at how our ten fingers shaped the base-10 system.",
                              "In base-9, digits run from 0 to 8. As you stroll through nature, you might notice that many natural phenomena have their own 'numbers' too - like the hexagonal cells of a beehive, the six-fold symmetry of a snowflake, or even the veins of a spinach leaf all following certain mathematical patterns. Mathematics is truly everywhere!",
                              "By the way, did you know that Beethoven's Fifth Symphony is known as the 'Symphony of Fate' in the world of music? Its opening four notes - short, short, short, long - are seen as a symbol of fate knocking at the door. See, that's another kind of power of numbers!",
                              "In base-9, digits go from 0 to 8. If you're a sports enthusiast, you might know that in baseball, a team can have up to 9 players on the field at once. This number is quite significant in baseball, much like our dealings with base-9 numbers in mathematics.",
                              "By the way, did you know that Mars is the fourth planet from the Sun in our solar system? It's known as the Red Planet because its surface is covered in iron oxide (commonly known as rust), giving it its distinctive red appearance. Space and mathematics have a close connection."]
        
        
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
        
        ret = ""
        ret +=  irrelative_content[self.distracting_index]
        self.distracting_index += 1  
        
        ret += f"We have {lo} + {ro} = {int(lo) + int(ro)} in base-10. Since we're in base-{base}, that exceeds the maximum value of {digits[base-1]} for a single digit. {int(lo) + int(ro)} mod {base} = {ones_sum[-1]}, so the digit is {ones_sum[-1]} and the carry is {ones_carry_digit}. "
        
        ret += irrelative_content[self.distracting_index]
        self.distracting_index += 1
        
        ret += f"We have {lt} + {rt} + {ones_carry_digit} = {int(lt) + int(rt) + ones_carry_digit} in base 10. {int(lt) + int(rt) + ones_carry_digit} mod {base} = {tens_sum_w_carry[-1]}, so the digit is {tens_sum_w_carry[-1]} and the carry is {tens_carry_digit}. A leading digit {tens_carry_digit}. So the answer is {self.get_label(expr)}. Answer:\\box{{{self.get_label(expr)}}}"
        return ret
            

    def get_question(self, expr, if_demo = False):
        base = self.base
        digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        if if_demo:
            return f"In base-{base}, what is {expr}?"
        else:
            return f"You are a mathematician. Assuming that all numbers are in base-{base} where the digits are \"{digits[:base]}\", what is {expr}? Let's think step by step, and end the response with the result in \"Answer:\\boxed{{result}}\"."
        # if cot:
        #     return f"You are a mathematician. Assuming that all numbers are in base-{base} where the digits are \"{digits[:base]}\", what is {expr}? Let's think step by step, and end the response with the result in \"Answer:\\boxed{{result}}\"."
        # else:
        #     return f"You are a mathematician. Assuming that all numbers are in base-{base} where the digits are \"{digits[:base]}\", what is {expr}? End the response with the result in \"Answer:\\boxed{{result}}\"."
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
        total_shots = self.n_shots + self.noisy_shots
        n_noisy_shot = self.noisy_shots
        case = dict()
        prefix = ''
        if total_shots > 0:
            COT = []
            expr, demos = expr.split("\t")
            weak_demos = demos.split(",")[:n_shots - self.ex_shots]
            assert n_shots >= self.ex_shots
            assert len(weak_demos) == n_shots - self.ex_shots
            for shot in weak_demos:
                COT_q = self.get_question(shot, if_demo=True)
                COT_a =  self.answer(shot)
                COT.append([COT_q, COT_a])
            ex_demos = demos.split(",")[n_shots - self.ex_shots:n_shots]
            assert len(ex_demos) == self.ex_shots
            for shot in ex_demos:
                COT_q = self.get_question(shot, if_demo=True)
                COT_a =  self.ex_answer(shot)
                COT.append([COT_q, COT_a])
            random.shuffle(COT)
            
            if self.noisy_shots > 0:
                
                # COT = COT[:n_shots - self.noisy_shots]
                if self.noisy_type == "misunderstanding":
                    noisy_shots = self.misunderstanding_noisy_answer()
                elif self.noisy_type == "distracting":
                    noisy_shots = []
                    noisy_demos = demos.split(',')[n_shots:n_shots+n_noisy_shot]
                    for demo in noisy_demos:
                        COT_q = self.get_question(demo, if_demo=True)
                        COT_a =  self.distracting_answer(demo)
                        noisy_shots.append([COT_q, COT_a])
                else:
                    noisy_shots = self.miscalculation_noisy_answer()
                COT = COT + noisy_shots
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


