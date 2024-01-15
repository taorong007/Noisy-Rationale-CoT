import numpy as np
import random
import json
import re
import math
import os
import time

class base_math:
    """
    Processor of a base caculation dataset.
    
    Attributes:
        n_shots: The number of normal shots. Default is 0.
        weak_shots: The number of weak shots. Default is 0.
        n_noisy_shots: The number of noisy shots. Default is 0.
        noise_type: The type of noisy, can be "misunderstanding" or "miscalculation". Default is "miscalculation".
        noise_ratio: The ratio of noise. Each thought has a chance.
        noise_distribution: The method to fill the noise. ( fixed noise num in one shot or random num in one shot )
        config: A dictionary containing all the attribute values. If provided, values in the config dictionary will be used to overwrite the following parameters.
        base: The base number for calculations. Default is 9.
        
    """
    def __init__(self,  n_shots=0, n_noisy_shots=0, noise_type="miscalculation", noise_ratio = 0.5, noise_distribution = "fixed", prefix_context = False, config: dict = None, reasoning_type="base-9") -> None:
        if config is not None:
            self.base = int(config["reasoning_type"].split("base-")[1])
        else:
            self.base = int(reasoning_type.split("base-"))

        self.n_shots = n_shots
        self.n_noisy_shots = n_noisy_shots
        self.total_shot = self.n_shots + self.n_noisy_shots 
        if self.total_shot > 0:
            self.if_in_context = True
        else:
            self.if_in_context = False
        
        # self.n_weak_shots = n_weak_shots
        
        if self.n_noisy_shots > 0:
            self.noise_ratio = noise_ratio
            self.noise_distribution = noise_distribution
            assert noise_distribution == "fixed" or noise_distribution == "random"
        else:
            self.noise_ratio = 0
            self.noise_distribution = None
        self.prefix_context = prefix_context
        self.noise_type = noise_type            
        self.irrelevant_index = 0
        self.file_path = self.get_file_path()
        

    def get_label(self, expr):
        base = self.base
        lhs, rhs = expr.split("+")
        lhs_base10 = int(lhs, base)
        rhs_base10 = int(rhs, base)
        sum_base10 = lhs_base10 + rhs_base10
        return np.base_repr(sum_base10, base)
    
    def get_answer(self, expr, generate_info = None):
        if generate_info!=None:
            generate_info["total_thought"] = 8
            generate_info["noise_thought"] = 0
            generate_info["sentences_with_noise"] = [0] * 9
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
        
        explaination = f"Since we're in base-{base}, that exceeds the maximum value of {digits[base-1]} for a single digit. " if carry_over == 1 else f"Since we're in base-{base}, that doesn't exceed the maximum value of {digits[base-1]} for a single digit. "
        
        #In base-{base} where the digits are \"{digits[:base]}\".
        ret = f"In base-{base}, the digits are \"{digits[:base]}\". We have {lo} + {ro} = {int(lo, base) + int(ro, base)} in base-10. "+ explaination + f"{int(lo, base) + int(ro, base)} mod {base} = {ones_sum[-1]}, so the digit is {ones_sum[-1]} and the carry is {ones_carry_digit}. We have {lt} + {rt} + {ones_carry_digit} = {int(lt, base) + int(rt, base) + ones_carry_digit} in base 10. {int(lt, base) + int(rt, base) + ones_carry_digit} mod {base} = {tens_sum_w_carry[-1]}, so the digit is {tens_sum_w_carry[-1]} and the carry is {tens_carry_digit}. A leading digit is {tens_carry_digit}. So the answer is {self.get_label(expr)}. Answer:\\box{{{self.get_label(expr)}}}"
        return ret
    
    def _generate_noise_distribution_list(self, n_thought, noise_ratio, noise_distribution):
        noise_distribution_list = [0] * n_thought
        if noise_distribution == "fixed":
            noise_count = round(n_thought * noise_ratio)
            noise_positions = random.sample(range(n_thought), noise_count)
            for pos in noise_positions:
                noise_distribution_list[pos] = 1
        else:
            for pos in range(len(noise_distribution_list)):
                if random.random() < noise_ratio:
                    noise_distribution_list[pos] = 1
        self.noise_pos = 0
        return noise_distribution_list
        
    def _should_add_noise(self, noise_distribution_list):
        if_noise = noise_distribution_list[self.noise_pos]
        self.noise_pos += 1
        return if_noise

    def get_generation_config(self, noise_distribution_list, generate_info):
        generate_info["total_thought"] = len(noise_distribution_list) + noise_distribution_list.count(1)
        generate_info["noise_thought"] = noise_distribution_list.count(1)
        generate_info["sentences_with_noise"] = []
        for if_noise in noise_distribution_list:
            generate_info["sentences_with_noise"].append(0)
            if if_noise:
                generate_info["sentences_with_noise"].append(1)
        generate_info["sentences_with_noise"].append(0)

    def irrelevant_answer(self, expr, generate_info = None):
        digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        base = self.base
        lhs, rhs = expr.split("+")
        lt, lo = lhs  # tens, ones
        rt, ro = rhs
        ones_sum = self.get_label(f"{lo}+{ro}")
        carry_over = len(ones_sum) > 1
        tens_sum_wo_carry = self.get_label(f"{lt}+{rt}")
        
        noise_distribution_list = self._generate_noise_distribution_list(n_thought=8, noise_ratio=self.noise_ratio, noise_distribution=self.noise_distribution)
        if generate_info!=None:
            self.get_generation_config(noise_distribution_list, generate_info)
        
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
        explaination = f"Since we're in base-{base}, that exceeds the maximum value of {digits[base-1]} for a single digit." if carry_over == 1 else f"Since we're in base-{base}, that doesn't exceed the maximum value of {digits[base-1]} for a single digit. "
        
        selected_noise_set = set()
    
        
        ret = f"In base-{base}, the digits are \"{digits[:base]}\". "
        if self._should_add_noise(noise_distribution_list):
            fact = self._random_choose_fact(base, selected_noise_set)    
            ret += f"{fact}. "
        
        ret += f" We have {lo} + {ro} = {int(lo, base) + int(ro, base)} in base-10. "
        if self._should_add_noise(noise_distribution_list):
            number = int(lo, base) + int(ro, base)
            fact = self._random_choose_fact(number, selected_noise_set)    
            ret += f"{fact}. "
         
        ret += explaination
        if self._should_add_noise(noise_distribution_list):
            number = int(digits[base-1], base)
            fact = self._random_choose_fact(number, selected_noise_set)    
            ret += f"{fact}. " 
        
        ret += f"{int(lo, base) + int(ro, base)} mod {base} = {ones_sum[-1]}, so the digit is {ones_sum[-1]} and the carry is {ones_carry_digit}. "
        if self._should_add_noise(noise_distribution_list):
            number = int(ones_sum[-1], base)
            fact = self._random_choose_fact(number, selected_noise_set)    
            ret += f"{fact}. "
        
        ret += f"We have {lt} + {rt} + {ones_carry_digit} = {int(lt, base) + int(rt, base) + ones_carry_digit} in base 10. " 
        number = int(lt, base) + int(rt, base) + ones_carry_digit
        if self._should_add_noise(noise_distribution_list):
            fact = self._random_choose_fact(number, selected_noise_set)    
            ret += f"{fact}. "
        
        ret += f"{int(lt, base) + int(rt, base) + ones_carry_digit} mod {base} = {tens_sum_w_carry[-1]}, so the digit is {tens_sum_w_carry[-1]} and the carry is {tens_carry_digit}. "
        if self._should_add_noise(noise_distribution_list):
            number = int(tens_sum_w_carry[-1], base)
            fact = self._random_choose_fact(number, selected_noise_set)    
            ret += f"{fact}. "

        ret += f"A leading digit is {tens_carry_digit}. "
        if self._should_add_noise(noise_distribution_list):
            number = tens_carry_digit
            fact = self._random_choose_fact(number, selected_noise_set)    
            ret += f"{fact}. "
        
        ret += f"So the answer is {self.get_label(expr)}. "
        if self._should_add_noise(noise_distribution_list):
            number = int(self.get_label(expr)[-1], base)
            fact = self._random_choose_fact(number, selected_noise_set)    
            ret += f"{fact}. "
        
        ret += f"Answer:\\box{{{self.get_label(expr)}}}"
        return ret
            
    def _get_random_number(self):
        randomnum = 0
        while randomnum == 0:
            randomnum = random.randrange(1, 9, 1)
        return randomnum
    
    def inaccurate_answer(self, expr, generate_info = None):
        digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        base = self.base
        lhs, rhs = expr.split("+")
        lt, lo = lhs  # tens, ones
        rt, ro = rhs
        ones_sum = self.get_label(f"{lo}+{ro}")
        carry_over = len(ones_sum) > 1
        tens_sum_wo_carry = self.get_label(f"{lt}+{rt}")
        
        noise_distribution_list = self._generate_noise_distribution_list(n_thought=8, noise_ratio=self.noise_ratio, noise_distribution=self.noise_distribution)
        if generate_info!=None:
            self.get_generation_config(noise_distribution_list, generate_info)
        
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
        explaination = f"Since we're in base-{base}, that exceeds the maximum value of {digits[base-1]} for a single digit." if carry_over == 1 else f"Since we're in base-{base}, that doesn't exceed the maximum value of {digits[base-1]} for a single digit. "
    
        
        ret = f"In base-{base}, the digits are \"{digits[:base]}\". "
        if self._should_add_noise(noise_distribution_list):
            fact = ""
            randomnum = 0
            while randomnum == 0: 
                randomnum = random.randrange(1, 9, 1)
            fact += f"{base} + {randomnum} = {base + randomnum}"
            ret += f"{fact}. "
        
        ret += f" We have {lo} + {ro} = {int(lo, base) + int(ro, base)} in base-10. "
        if self._should_add_noise(noise_distribution_list):
            fact = ""
            randomnum = self._get_random_number()
            fact += "{} + {} = {}".format(int(lo, base) + int(ro, base), randomnum, int(lo, base) + int(ro, base) + randomnum)
            ret += f"{fact}. "
         
        ret += explaination 
        if self._should_add_noise(noise_distribution_list):
            fact = ""
            number = int(digits[base-1], base)
            randomnum = self._get_random_number()
            fact += "{} + {} = {}".format(number, randomnum, number + randomnum)
            ret += f"{fact}. "
        
        ret += f"{int(lo, base) + int(ro, base)} mod {base} = {ones_sum[-1]}, so the digit is {ones_sum[-1]} and the carry is {ones_carry_digit}. "
        if self._should_add_noise(noise_distribution_list):
            fact = ""
            fact += f"{ones_sum[-1]} + {base} = {int(ones_sum[-1], base) + base}"
            ret += f"{fact}. "
        
        ret += f"We have {lt} + {rt} + {ones_carry_digit} = {int(lt, base) + int(rt, base) + ones_carry_digit} in base 10. " 
        if self._should_add_noise(noise_distribution_list):
            fact = ""
            randomnum = self._get_random_number()
            fact += f"{int(lt, base) + int(rt, base) + ones_carry_digit} + {randomnum} = {int(lt, base) + int(rt, base) + ones_carry_digit + randomnum}"
            ret += f"{fact}. "

        ret += f"{int(lt, base) + int(rt, base) + ones_carry_digit} mod {base} = {tens_sum_w_carry[-1]}, so the digit is {tens_sum_w_carry[-1]} and the carry is {tens_carry_digit}. "
        if self._should_add_noise(noise_distribution_list):
            fact = ""
            fact += f"{tens_sum_w_carry[-1]} + {base} = {int(tens_sum_w_carry[-1], base) + base}"
            ret += f"{fact}. "
        
        ret += f"A leading digit is {tens_carry_digit}. "    
        if self._should_add_noise(noise_distribution_list):
            fact = ""
            fact += f"{tens_carry_digit} + {base} = {tens_carry_digit + base}"
            ret += f"{fact}. "
        
        ret += f"So the answer is {self.get_label(expr)}. "
        if self._should_add_noise(noise_distribution_list):
            fact = ""
            number = int(self.get_label(expr)[-1], base)
            fact += f"{number} + {base} = {number + base}"
            ret += f"{fact}. "
        
        ret += "Answer:\\box{{{self.get_label(expr)}}}. "
        return ret
    
    # def inaccurate_answer(self, expr):
        
    #     noise_distribution_list = self._generate_noise_distribution_list(n_thought=16, noise_ratio=self.noise_ratio, noise_distribution=self.noise_distribution)
        
    #     digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    #     base = self.base
    #     lhs, rhs = expr.split("+")
    #     lt, lo = lhs  # tens, ones
    #     rt, ro = rhs
    #     ones_sum_base10 = int(lo) + int(ro)
    #     digit_one = np.base_repr(ones_sum_base10 % base, base)
    #     carry_over =  int(ones_sum_base10/base)
    #     tens_sum_base10 = int(lt) + int(rt) + carry_over
    #     digit_ten =  np.base_repr((tens_sum_base10 + self._get_random_error()) % base, base)
    #     tens_carry_over = int(tens_sum_base10 / base)
        
        
    #     if tens_carry_over > 0:
    #         result = str(tens_carry_over) + digit_ten + digit_one
    #     else:
    #         result = digit_ten + digit_one
            
    #     explaination = f"Since we're in base-{base}, that exceeds the maximum value of {digits[base-1]} for a single digit." if carry_over >= 1 else f"Since we're in base-{base}, that doesn't exceed the maximum value of {digits[base-1]} for a single digit. "
        
    #     ret = f"In base-{base}, the digits are \"{digits[:base]}\". "
        
    #     lo_show = int(lo) + self._get_random_error() if self._should_add_noise(noise_distribution_list) else lo
    #     ro_show = int(ro) + self._get_random_error() if self._should_add_noise(noise_distribution_list) else ro
    #     ones_sum_base10_show = ones_sum_base10 + self._get_random_error() if self._should_add_noise(noise_distribution_list) else ones_sum_base10
    #     ret += f"We have {lo_show} + {ro_show} = {ones_sum_base10_show} in base-10. "
        
    #     digit_one_show = int(digit_one) + self._get_random_error() if self._should_add_noise(noise_distribution_list) else digit_one
    #     ones_sum_base10_show = ones_sum_base10 + self._get_random_error() if self._should_add_noise(noise_distribution_list) else ones_sum_base10
    #     ret += explaination + f"{ones_sum_base10_show} mod {base} = {digit_one_show}, "
        
    #     digit_one_show = int(digit_one) + self._get_random_error() if self._should_add_noise(noise_distribution_list) else digit_one
    #     carry_over_show = carry_over + self._get_random_error() if self._should_add_noise(noise_distribution_list) else carry_over
    #     ret += f"so the digit is {digit_one_show} and the carry is {carry_over_show}." 
        
    #     lt_show = int(lt) + self._get_random_error() if self._should_add_noise(noise_distribution_list) else lt
    #     rt_show = int(rt) + self._get_random_error() if self._should_add_noise(noise_distribution_list) else rt
    #     carry_over_show = carry_over + self._get_random_error() if self._should_add_noise(noise_distribution_list) else carry_over
    #     tens_sum_base10_show = tens_sum_base10 + self._get_random_error() if self._should_add_noise(noise_distribution_list) else tens_sum_base10
    #     ret += f"We have {lt_show} + {rt_show} + {carry_over_show} = {tens_sum_base10_show} in base 10. "
        
    #     tens_sum_base10_show = tens_sum_base10 + self._get_random_error() if self._should_add_noise(noise_distribution_list) else tens_sum_base10
    #     digit_ten_show = int(digit_ten) + self._get_random_error() if self._should_add_noise(noise_distribution_list) else digit_ten
    #     ret += f"{tens_sum_base10} mod {base} = {digit_ten_show}, "
        
    #     digit_ten_show = int(digit_ten) + self._get_random_error() if self._should_add_noise(noise_distribution_list) else digit_ten
    #     tens_carry_over_show = tens_carry_over + self._get_random_error() if self._should_add_noise(noise_distribution_list) else tens_carry_over
    #     ret += f"so the digit is {digit_ten_show} and the carry is {tens_carry_over_show}. "
        
    #     tens_carry_over_show = tens_carry_over + self._get_random_error() if self._should_add_noise(noise_distribution_list) else tens_carry_over
    #     ret += f"A leading digit {tens_carry_over}. So the answer is {result}. Answer:\\box{{{result}}}"
        
        # if random.random() < noise_p:
        #     ones_sum_base10 += self._get_random_error()
        # if random.random() < noise_p:
        #     digit_one = np.base_repr((ones_sum_base10 + self._get_random_error()) % base, base)
        # else:
        #     digit_one = np.base_repr(ones_sum_base10 % base, base)
        
        # carry_over_show = carry_over
        # if random.random() < noise_p:
        #     if carry_over == 0:
        #         carry_over = 1
        #     else:
        #         carry_over = 0
        
        # if random.random() < noise_p:
        #     tens_sum_base10 +=  self._get_random_error()
        # if random.random() < noise_p:
        #     digit_ten =  np.base_repr((tens_sum_base10 + self._get_random_error()) % base, base)
        # else:
        #     digit_ten =  np.base_repr(tens_sum_base10 % base, base)
        # tens_carry_over = int(tens_sum_base10 / base)
        # tens_carry_over_show = tens_carry_over
        # if random.random() < noise_p:
        #     if tens_carry_over == 0:
        #         tens_carry_over = 1
        #     else:
        #         tens_carry_over = 0
        # return ret
    
    def _random_choose_fact(self, number, selected_noise_set:set):
        facts = self.noise_data[number]["facts"]
        while(1):
            random_index = random.randrange(0, len(facts))
            selected_fact = facts[random_index]
            fact_index = f"{number}-{random_index}"
            if fact_index not in selected_noise_set:
                selected_noise_set.add(fact_index)
                # print(fact_index)
                break
        # selected_fact = selected_fact[0] + selected_fact[1:]
        if selected_fact[-1] == ".":
            selected_fact = selected_fact[:-1] 
        return selected_fact
        
        
    def get_question(self, expr):
        base = self.base
        digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        if self.if_in_context:
            return f"In base-{base}, what is {expr}? Please reason it step by step. End the response with the result in \"Answer:\\boxed{{result}}\"."
        else:
            return f"You are a mathematician. Assuming that all numbers are in base-{base} where the digits are \"{digits[:base]}\", what is {expr}? Please reason it step by step. End the response with the result in \"Answer:\\boxed{{result}}\"."
        

    def get_case(self, expr, if_generate_info=False):
        n_shots = self.n_shots
        total_shots = self.n_shots + self.n_noisy_shots
        n_noisy_shot = self.n_noisy_shots
        case = dict()
        prefix = ''
        shots = []
        expr, demos = expr.split("\t")
        if total_shots > 0:    
            normal_demos = demos.split(",")[:n_shots]
            assert len(normal_demos) == self.n_shots
            for demo in normal_demos:
                if if_generate_info:
                    generate_info = dict()
                else:
                    generate_info = None
                shot_q = self.get_question(demo)
                shot_a =  self.get_answer(demo, generate_info)
                if not if_generate_info:
                    shots.append([shot_q, shot_a])
                else:
                    shots.append([shot_q, shot_a, generate_info])
            # random.shuffle(shots)
            
            if self.n_noisy_shots > 0:    
                # shots = shots[:n_shots - self.n_noisy_shots]
                noisy_shots = []
                noisy_demos = demos.split(',')[n_shots:n_shots + n_noisy_shot]
                for demo in noisy_demos:
                    if if_generate_info:
                        generate_info = dict()
                    else:
                        generate_info = None
                    shot_q = self.get_question(demo)
                    if self.noise_type == "inaccurate":
                        shot_a =  self.inaccurate_answer(demo, generate_info)
                    elif self.noise_type == "irrelevant":
                        shot_a =  self.irrelevant_answer(demo, generate_info)
                    else:
                        raise ValueError(f"noisy type not support:{self.noise_type}")
                    if not if_generate_info:
                        noisy_shots.append([shot_q, shot_a])
                    else:
                        noisy_shots.append([shot_q, shot_a, generate_info])
                shots = shots + noisy_shots
                # random.shuffle(shots)
            # if self.prefix_context:
            #     for shot in shots:
            #         prefix += "user:{}\nassistant:{}\n".format(shot[0], shot[1])
            #     prefix += "user:"
            # else:    
                case["in-context"] = shots
        question = self.get_question(expr)
        real_answer = self.get_label(expr)
        case["question"] = prefix + question
        case["label"] = real_answer 
        return case
    
    def load_data(self):
        noise_file = "./data/base_math/noise/factsOfNumber.json"
        data_file = "./data/base_math/icl/base{}.txt".format(self.base)
        dataset = [line.strip() for line in open(data_file)]
        with open(noise_file, encoding="utf-8") as f:
            self.noise_data = json.load(f)["noise_info"]
        return dataset
    
    @staticmethod
    def match_answer(answer_str):
        match = re.search(r'[Aa]nswer:.*?(-?[\da-fA-F]+)(?!.*[\da-fA-F])', answer_str)
        if match:
            answer = match.group(1)
        else:
            answer = None
        return answer

    @staticmethod
    def get_file_path():
        return os.path.join("data", "base_math")
        