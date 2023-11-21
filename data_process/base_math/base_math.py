import numpy as np
import random
import json

class base_math:
    """
    Processor of a base caculation dataset.
    
    Attributes:
        if_in_context: A boolean indicating if in-context learning is used. Default is False.
        n_shots: The number of normal shots. Default is 0.
        weak_shots: The number of weak shots. Default is 0.
        n_noisy_shots: The number of noisy shots. Default is 0.
        noisy_type: The type of noisy, can be "misunderstanding" or "miscalculation". Default is "miscalculation".
        noisy_level: The level of noise. 1 - low, 2 - mid, 3 - high.
        prefix_context: A boolean indicating if prefix context is used. Default is False.
        config: A dictionary containing all the attribute values. If provided, values in the config dictionary will be used to overwrite the following parameters.
        base: The base number for calculations. Default is 9.
        
    """
    def __init__(self, if_in_context = False, n_shots=0, n_weak_shots = 0, n_noisy_shots=0, noisy_type="miscalculation", noisy_level = 1, prefix_context = False, config: dict = None, base=9) -> None:
        if config is not None:
            self.base = config["base"]
        else:
            self.base = base
            
        self.if_in_context = if_in_context
        self.n_shots = n_shots
        self.n_weak_shots = n_weak_shots
        self.n_noisy_shots = n_noisy_shots
        self.noisy_level = noisy_level
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


    def weak_answer(self,expr):
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
    
    def answer(self, expr):
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
        
        explaination = f"Since we're in base-{base}, that exceeds the maximum value of {digits[base-1]} for a single digit." if carry_over == 1 else f"Since we're in base-{base}, that doesn't exceed the maximum value of {digits[base-1]} for a single digit. "
        
        #In base-{base} where the digits are \"{digits[:base]}\".
        ret = f"In base-{base}, the digits are \"{digits[:base]}\". We have {lo} + {ro} = {int(lo) + int(ro)} in base-10. "+ explaination + f"{int(lo) + int(ro)} mod {base} = {ones_sum[-1]}, so the digit is {ones_sum[-1]} and the carry is {ones_carry_digit}. We have {lt} + {rt} + {ones_carry_digit} = {int(lt) + int(rt) + ones_carry_digit} in base 10. {int(lt) + int(rt) + ones_carry_digit} mod {base} = {tens_sum_w_carry[-1]}, so the digit is {tens_sum_w_carry[-1]} and the carry is {tens_carry_digit}. A leading digit {tens_carry_digit}. So the answer is {self.get_label(expr)}. Answer:\\box{{{self.get_label(expr)}}}"
        return ret
    
    def distracting_answer(self, expr):
        level = self.noisy_level
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
        explaination = f"Since we're in base-{base}, that exceeds the maximum value of {digits[base-1]} for a single digit." if carry_over == 1 else f"Since we're in base-{base}, that doesn't exceed the maximum value of {digits[base-1]} for a single digit. "
        
        selected_noise_set = set()
        ret = f"In base-{base}, the digits are \"{digits[:base]}\". "
        # if(level>=3):
        #     fact = self._random_choose_fact(-1, selected_noise_set)    
        #     ret += f"{fact}. "
        ret += f" We have {lo} + {ro} = {int(lo) + int(ro)} in base-10. "
        if(level>=1):
            number = int(lo) + int(ro)
            fact = self._random_choose_fact(number, selected_noise_set)    
            ret += f"{fact}. "
            if(level >= 2):
                fact = self._random_choose_fact(number, selected_noise_set)    
                ret += f"{fact}. "
         
        ret += explaination + f"{int(lo) + int(ro)} mod {base} = {ones_sum[-1]}, so the digit is {ones_sum[-1]} and the carry is {ones_carry_digit}. "
        if(level >= 3):
            number = int(ones_sum[-1])
            fact = self._random_choose_fact(number, selected_noise_set)    
            ret += f"{fact}. "
        
        ret += f"We have {lt} + {rt} + {ones_carry_digit} = {int(lt) + int(rt) + ones_carry_digit} in base 10. " 
        
        if(level>=1):
            number = int(lt) + int(rt) + ones_carry_digit
            fact = self._random_choose_fact(number, selected_noise_set)    
            ret += f"{fact}. "
            if(level >= 2):
                fact = self._random_choose_fact(number, selected_noise_set)    
                ret += f"{fact}. "
        
        ret += f"{int(lt) + int(rt) + ones_carry_digit} mod {base} = {tens_sum_w_carry[-1]}, so the digit is {tens_sum_w_carry[-1]} and the carry is {tens_carry_digit}. A leading digit {tens_carry_digit}. "
        if(level >= 3):
            number = int(tens_sum_w_carry[-1])
            fact = self._random_choose_fact(number, selected_noise_set)    
            ret += f"{fact}. "
        ret += f"So the answer is {self.get_label(expr)}. Answer:\\box{{{self.get_label(expr)}}}"
        
        return ret
            
    def _get_random_error(self):
        randomnum = 0
        while randomnum == 0:
            randomnum = random.randrange(-3, 3, 1)
        return randomnum
    
    def arithmetic_error_answer(self, expr):
        digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        noisy_level = self.noisy_level
        base = self.base
        lhs, rhs = expr.split("+")
        lt, lo = lhs  # tens, ones
        rt, ro = rhs
        ones_sum_base10 = int(lo) + int(ro)
        if noisy_level >= 1:
            ones_sum_base10 += self._get_random_error()
        if noisy_level>=2:
            digit_one = np.base_repr((ones_sum_base10 + self._get_random_error()) % base, base)
        else:
            digit_one = np.base_repr(ones_sum_base10 % base, base)
        carry_over =  int(ones_sum_base10/base)
        carry_over_show = carry_over
        if noisy_level >= 3:
            if carry_over == 0:
                carry_over = 1
            else:
                carry_over = 0
        tens_sum_base10 = int(lt) + int(rt) + carry_over
        if noisy_level >= 1:
            tens_sum_base10 +=  self._get_random_error()
        if noisy_level>=2:
            digit_ten =  np.base_repr((tens_sum_base10 + self._get_random_error()) % base, base)
        else:
            digit_ten =  np.base_repr(tens_sum_base10 % base, base)
        tens_carry_over = int(tens_sum_base10 / base)
        tens_carry_over_show = tens_carry_over
        if noisy_level >= 3:
            if tens_carry_over == 0:
                tens_carry_over = 1
            else:
                tens_carry_over = 0
        
        if tens_carry_over > 0:
            result = str(tens_carry_over) + digit_ten + digit_one
        else:
            result = digit_ten + digit_one
        
        explaination = f"Since we're in base-{base}, that exceeds the maximum value of {digits[base-1]} for a single digit." if carry_over >= 1 else f"Since we're in base-{base}, that doesn't exceed the maximum value of {digits[base-1]} for a single digit. "
                
        ret = f"In base-{base}, the digits are \"{digits[:base]}\". We have {lo} + {ro} = {ones_sum_base10} in base-10. "
        
        ret += explaination + f"{ones_sum_base10} mod {base} = {digit_one}, so the digit is {digit_one} and the carry is {carry_over_show}. We have {lt} + {rt} + {carry_over} = {tens_sum_base10} in base 10. {tens_sum_base10} mod {base} = {digit_ten}, so the digit is {digit_ten} and the carry is {tens_carry_over_show}. A leading digit {tens_carry_over}. So the answer is {result}. Answer:\\box{{{result}}}"
        return ret
    
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
        selected_fact = selected_fact[0].lower() + selected_fact[1:]
        if selected_fact[-1] == ".":
            selected_fact = selected_fact[:-1] 
        return selected_fact
        
        
    def get_question(self, expr):
        base = self.base
        digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        if self.if_in_context:
            return f"In base-{base}, what is {expr}? end the response with the result in \"Answer:\\boxed{{result}}\"."
        else:
            return f"You are a mathematician. Assuming that all numbers are in base-{base} where the digits are \"{digits[:base]}\", what is {expr}? End the response with the result in \"Answer:\\boxed{{result}}\"."
        

    def get_prompt_case(self, expr):
        n_shots = self.n_shots
        total_shots = self.n_shots + self.n_weak_shots + self.n_noisy_shots
        n_noisy_shot = self.n_noisy_shots
        case = dict()
        prefix = ''
        if total_shots > 0:
            shots = []
            expr, demos = expr.split("\t")
            normal_demos = demos.split(",")[:n_shots]
            assert len(normal_demos) == self.n_shots
            for demo in normal_demos:
                shot_q = self.get_question(demo)
                shot_a =  self.answer(demo)
                shots.append([shot_q, shot_a])
            if self.n_weak_shots > 0:
                weak_demos = demos.split(",")[n_shots:n_shots + self.n_weak_shots]
                assert len(weak_demos) == self.n_weak_shots
                for shot in weak_demos:
                    shot_q = self.get_question(demo)
                    shot_a =  self.weak_answer(demo)
                    shots.append([shot_q, shot_a])
            random.shuffle(shots)
            
            if self.n_noisy_shots > 0:    
                # shots = shots[:n_shots - self.n_noisy_shots]
                if self.noisy_type == "arithmetic_error":
                    noisy_shots = []
                    noisy_demos = demos.split(',')[n_shots:n_shots+n_noisy_shot]
                    for demo in noisy_demos:
                        shot_q = self.get_question(demo)
                        shot_a =  self.arithmetic_error_answer(demo)
                        noisy_shots.append([shot_q, shot_a])
                elif self.noisy_type == "distracting":
                    noisy_shots = []
                    noisy_demos = demos.split(',')[n_shots:n_shots+n_noisy_shot]
                    for demo in noisy_demos:
                        shot_q = self.get_question(demo)
                        shot_a =  self.distracting_answer(demo)
                        noisy_shots.append([shot_q, shot_a])
                else:
                    raise ValueError(f"noisy type not support:{self.noisy_type}")
                shots = shots + noisy_shots
                random.shuffle(shots)
            if self.prefix_context:
                for shot in shots:
                    prefix += "user:{}\nassistant:{}\n".format(shot[0], shot[1])
            else:    
                case["in-context"] = shots
        question = self.get_question(expr)
        real_answer = self.get_label(expr)
        case["question"] = prefix + question
        case["label"] = real_answer 
        return case
        
        
    def load_data(self):
        noise_file = "./data/base_math/noise/factsOfNumber.json".format(self.base)
        data_file = "./data/base_math/icl/base{}.txt".format(self.base)
        dataset = [line.strip() for line in open(data_file)]
        with open(noise_file, encoding="utf-8") as f:
            self.noise_data = json.load(f)["noise_info"]
        return dataset


