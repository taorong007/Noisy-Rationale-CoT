import numpy as np
from .code.old_code.denoiser import denoise_instance
from .code.utils.mask import mask_sentence, mask_forbidden_index

class SimpleArgs:
    def __init__(self,  denoise_method, mask_word, sparse_mask_rate):
        self.denoise_method = denoise_method
        self.mask_word = mask_word
        self.sparse_mask_rate = sparse_mask_rate


class SelfDenoise:
    def __init__(self) -> None:
        self.mask_word = "###"
        self.args = SimpleArgs(denoise_method="chatgpt_single_by_model", mask_word="###", sparse_mask_rate=0.1)
    

    def mask_instance_decorator(self, instance, numbers:int=1, return_indexes:bool=False,random_probs=None):
        args = self.args
        return mask_sentence(instance, args.sparse_mask_rate, self.tokenizer.mask_token, numbers, return_indexes,random_probs=random_probs)

    
    def certify(self, case_batch, model, **kwargs):
    
        # log_file = open(os.path.join(args.save_path,'log.txt'),'w+')
        
        index_org_sentence = -1
        for case in case_batch:
            
            
            keep_nums = data_length - round(data_length * args.sparse_mask_rate)

            tmp_instances = self.mask_instance_decorator(data, args.predict_ensemble, random_probs = None)

            for instance in tmp_instances:
                instance.text_a = instance.text_a.replace("<mask>", args.mask_word)
                if instance.text_b is not None:
                    instance.text_b = instance.text_b.replace("<mask>", args.mask_word)
            
            denoise_instance(tmp_instances, args={""})
            # save or load pred_denoised_sentence
                    
            # load pred_prediction

            # load pred_prediction_prob
            if args.recover_past_data:
                if os.path.exists(os.path.join(args.save_path,"pred_prediction_prob",f"{index_org_sentence}.npy")):
                    past_pred_predictions_prob = np.load(os.path.join(args.save_path,"pred_prediction_prob",f"{index_org_sentence}.npy"))
                else:
                    past_pred_predictions_prob = None
            else:
                past_pred_predictions_prob = None

            tmp_probs, pred_predictions = predictor.predict_batch(tmp_instances,past_pred_predictions,past_pred_predictions_prob)

            cancate_p_list.append(tmp_probs)
            cancate_label_list.extend( [target for _ in range(len(tmp_probs))] )
                

            guess = np.argmax(tmp_probs, axis=-1).reshape(-1)
            print(list(guess),np.argmax(np.bincount(np.argmax(tmp_probs, axis=-1), minlength=num_labels)),'|',target,file=log_file,flush=True)
            
            guess = np.argmax(np.bincount(np.argmax(tmp_probs, axis=-1), minlength=num_labels))

            guess_distri[guess] += 1

            all_guess = np.argmax(tmp_probs, axis=-1)

            for item in all_guess:
                guess_distri_ensemble[item]+=1
            

            # print('certify',flush=True)
            if guess != target:
                radius = np.nan
            else:

                tmp_instances = self.mask_instance_decorator(args, data, args.ceritfy_ensemble, random_probs=random_probs)

                # save or load certify_masked_sentence
                index_certify_masked_sentence=-1
                if not os.path.exists(os.path.join(args.save_path,"certify_masked_sentence",f"{index_org_sentence}")):
                    os.makedirs(os.path.join(args.save_path,"certify_masked_sentence",f"{index_org_sentence}"))
                for instance in tmp_instances:
                    index_certify_masked_sentence+=1
                    if instance.text_b is None:
                        certify_masked_sentence_path = os.path.join(args.save_path,"certify_masked_sentence",f"{index_org_sentence}",f"a-{index_certify_masked_sentence}")
                        
                        if args.recover_past_data:
                            if os.path.exists(certify_masked_sentence_path):
                                with open(certify_masked_sentence_path, 'r') as file:
                                    content = file.read()
                                    instance.text_a = content
                        with open(certify_masked_sentence_path, 'w') as file:
                            file.write(instance.text_a)
                    else:
                        raise RuntimeError

                for data in tmp_instances:
                    data.text_a = data.text_a.replace("<mask>", args.mask_word)
                    if data.text_b is not None:
                        data.text_b = data.text_b.replace("<mask>", args.mask_word)
                
                denoise_instance(tmp_instances, args)
                
                # save or load certify_denoised_sentence
                if args.denoise_method is not None:
                    if not os.path.exists(os.path.join(args.save_path,"certify_denoised_sentence",f"{index_org_sentence}")):
                        os.makedirs(os.path.join(args.save_path,"certify_denoised_sentence",f"{index_org_sentence}"))
                    index_certify_denoised_sentence=-1
                    for instance in tmp_instances:
                        index_certify_denoised_sentence+=1
                        if instance.text_b is None:
                            certify_denoised_sentence_path = os.path.join(args.save_path,"certify_denoised_sentence",f"{index_org_sentence}",f"a-{index_certify_denoised_sentence}")
                            if args.recover_past_data:
                                if os.path.exists(certify_denoised_sentence_path):
                                    with open(certify_denoised_sentence_path, 'r') as file:
                                        content = file.read()
                                        instance.text_a = content
                            with open(certify_denoised_sentence_path, 'w') as file:
                                file.write(instance.text_a)
                        else:
                            raise RuntimeError

                # load certify_prediction
                if args.recover_past_data:
                    if os.path.exists(os.path.join(args.save_path,"certify_prediction",f"{index_org_sentence}",f"0")):
                        past_certify_predictions = []
                        for i in range(len(tmp_instances)):
                            with open(os.path.join(args.save_path,"certify_prediction",f"{index_org_sentence}",f"{i}"), 'r') as file:
                                content = file.read()
                                past_certify_predictions.append(content)
                    else:
                        past_certify_predictions = None
                else:
                    past_certify_predictions = None

                if args.recover_past_data:
                    if os.path.exists(os.path.join(args.save_path,"certify_prediction_prob",f"{index_org_sentence}.npy")):
                        past_pred_predictions_prob = np.load(os.path.join(args.save_path,"certify_prediction_prob",f"{index_org_sentence}.npy"))
                    else:
                        past_pred_predictions_prob = None
                else:
                    past_pred_predictions_prob = None

                if args.predictor == 'bert':
                    tmp_probs = predictor.predict_batch(tmp_instances)
                    # certify_predictions = None
                else:
                    tmp_probs, certify_predictions = predictor.predict_batch(tmp_instances,past_certify_predictions,past_pred_predictions_prob)
                    # save pred_prediction
                    if not os.path.exists(os.path.join(args.save_path,"certify_prediction",f"{index_org_sentence}")):
                        os.makedirs(os.path.join(args.save_path,"certify_prediction",f"{index_org_sentence}"))
                    
                    if certify_predictions is not None:
                        for i in range(len(certify_predictions)):
                            certify_prediction_path = os.path.join(args.save_path,"certify_prediction",f"{index_org_sentence}",f'{i}')
                            with open(certify_prediction_path, 'w') as file:
                                    file.write(certify_predictions[i])

                    # save certify_prediction_prob
                    np.save(os.path.join(args.save_path,"certify_prediction_prob",f"{index_org_sentence}.npy"), tmp_probs)

                guess_count = np.bincount(np.argmax(tmp_probs, axis=-1), minlength=num_labels)[guess]
                lower_bound, upper_bound = lc_bound(guess_count, args.ceritfy_ensemble, args.alpha)

                guess_counts = np.bincount(np.argmax(tmp_probs, axis=-1), minlength=num_labels)
                print('guess_counts:',guess_counts)
                tmp = guess_counts/guess_count.sum()
                
                entropy_list.append(-tmp*np.log(np.clip(tmp, 1e-6, 1)))
                print("lower_bound:",lower_bound,file=log_file,flush=True)