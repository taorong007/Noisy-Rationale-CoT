# !/usr/bin/env python3
# _*_ coding:utf-8 _*_

class ProblemMethod():
    def __init__(self, problem_method="Normal"):
        self.problem_method = problem_method
        # assert answer_method is not None
        # self.answer_method=answer_method
        pass
    def forward(self):
        pass


class ProblemSelfPolish(ProblemMethod):
    def __init__(self,problem_method="SP", model=None):
        self.problem_method=problem_method
        self.final_choose=final_choose
        self.max_times = 3
        self.answer_method=answer_method
        self.model=model
        pass



    def generate_one_new_answer(self, eng, dataset, prompt_index, original_question):
        generate_rewrite_version_prompt_path = "prompt/my_prompts/rewrite_prompt/{}/{}_rewrite_prompt_{}.txt".format(
            dataset, dataset, prompt_index)
        with open(file=generate_rewrite_version_prompt_path, mode="r", encoding="utf-8") as f:
            generate_rewrite_version_prompt = f.read().strip()
        # if i == 0:
        #     print(f"\nmodel: {eng}")
        #     print(f"generate_rewrite_version_prompt_path: {generate_rewrite_version_prompt_path}\n")
        #     print(f"\ngenerate rewrite version prompt: {generate_rewrite_version_prompt}\n")
        prompt_input_to_generate_new_question = generate_rewrite_version_prompt + "\n\nOriginal: {}\nNew:".format(
            original_question)
        messages = [{"role": "system",
                    "content": "Please rewrite new versions of the original answer to be more understandable and more relevant to the question. Don't omit any useful information, especially the numbers, and please maintain their original meaning when polysemous words appear."},
                {"role": "user", "content": prompt_input_to_generate_new_question}]
        new_generated_question = new_generated_question["choices"][0]["message"]["content"]
        return new_generated_question

    def forward(self,question,answer,direct_answer_trigger_for_fewshot,dataset,prompt_index):

        original_answer = None
        original_correctness=False
        consistent_correctness=False


        consistent_answer = None
        vote_answer = None
        convergence_flag = True
        times = 0

        last_answer = original_answer
        last_question = question
        # print("------------------------------------------------------------")
        # print("Self-Polish start!")
        all_answers = [original_answer]
        while True:
            times += 1
            if times >= self.max_times:
                convergence_flag = False
                vote_answer = max(all_answers, key=all_answers.count)
                cnt = all_answers.count(max(all_answers, key=all_answers.count))
                if cnt == 1:
                    vote_answer = original_answer
                else:
                    vote_answer = vote_answer

                if self.final_choose == "original":
                    consistent_answer = original_answer
                elif self.final_choose == "last_one":
                    consistent_answer = last_answer
                elif self.final_choose == "first_one":
                    consistent_answer = first_answer
                elif self.final_choose == "vote":
                    consistent_answer = vote_answer

                print("More than {} times!".format(self.max_times))
                break
            
            # new problem
            new_generated_question = self.generate_one_new_answer(
                eng=self.eng,
                dataset=dataset,
                prompt_index=prompt_index,
                original_question=last_question
            )
            print("The {} times generated: {}".format(times, new_generated_question))

            # new answer
            new_answer,new_correctness = self.process_problem_for_one_time(question=new_generated_question,
                                        answer=answer,
                                        direct_answer_trigger_for_fewshot=direct_answer_trigger_for_fewshot,
                                        dataset=dataset,
                                        )
            if times == 1:
                first_answer = new_answer

            
            if last_answer != None and new_answer != None and last_answer == new_answer:
                print("Answer Converged!")
                consistent_answer = new_answer
                break

        return original_correctness, original_answer, consistent_correctness, consistent_answer


if __name__ == '__main__':
    pass
