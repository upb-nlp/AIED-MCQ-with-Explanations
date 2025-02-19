from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class QuestionGenerationUtils:
    def __init__(self, device, model_name, token=None):
        if device not in ['cpu', 'cuda', 'mps']:
            raise ValueError(f"Invalid device: {device}. Must be one of 'cpu', 'cuda', 'mps'.")
        
        if not model_name:
            raise ValueError("model_name cannot be empty.")
        
        self.device = device
        self.qall_tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", token=token)
        self.qall_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", token=token)
        self.qall_tokenizer.pad_token_id = self.qall_tokenizer.eos_token_id
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
        self.loss_fn_without_reduction = torch.nn.CrossEntropyLoss(reduction='none')

    def _find_sublist_index(self, main_list, sublist):
        n, m = len(main_list), len(sublist)
        for i in range(n - m + 1):
            if main_list[i:i + m] == sublist:
                return i
        return -1
    
    def _cut_sublist(self, main_list, num_elements, end_idx):
        sublist = main_list[:end_idx]
        res = sublist[-num_elements:]
        return res
    
    def generate_all_questions(self, context, num_samples):
        messages = [
            {"role": "system", "content": "You are an educational expert."},
            {"role": "user", "content": f"Generate a question, an answer and 3 distractors based on the context.\n\nContext:\n{context}"},
            {"role": "assistant", "content": f"Question:"},
        ]
        prompt = self.qall_tokenizer.apply_chat_template(messages, add_special_tokens=False, tokenize=False, continue_final_message=True)

        inputs = self.qall_tokenizer([prompt], return_tensors="pt", max_length=2048, truncation=True, padding='longest').to(self.device)

        with torch.no_grad():
            outputs = self.qall_model.generate(
                inputs['input_ids'], 
                attention_mask=inputs['attention_mask'], 
                do_sample=True,
                temperature=None,
                top_k=None, 
                top_p=None,
                max_new_tokens=40, 
                num_return_sequences=num_samples,
                pad_token_id=self.qall_tokenizer.eos_token_id,
                tokenizer=self.qall_tokenizer,
                stop_strings=["Answer:"],
                output_logits=True,
                return_dict_in_generate=True,
            )
            stop_string_ids = self.qall_tokenizer.encode("Answer:", add_special_tokens=False)

            generated_ids_sequences = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip([inputs.input_ids[0]] * num_samples, outputs['sequences'])
            ]

            generated_ids_logits = torch.transpose(torch.stack(outputs['logits']), 0, 1)

            losses = []

            for i in range(len(generated_ids_sequences)):
                end_idx_sequence = self._find_sublist_index(generated_ids_sequences[i].tolist(), stop_string_ids)
                generated_ids_sequences[i] = generated_ids_sequences[i][:end_idx_sequence]

                good_logit = generated_ids_logits[i][:end_idx_sequence]

                loss = self.loss_fn(
                        good_logit,
                        generated_ids_sequences[i],
                )
                losses.append(loss.item())

            responses = self.qall_tokenizer.batch_decode(generated_ids_sequences, skip_special_tokens=True)

        return responses, losses
    
    def generate_all_answers(self, context, question, num_samples):
        messages = [
            {"role": "system", "content": "You are an educational expert."},
            {"role": "user", "content": f"Generate a question, an answer and 3 distractors based on the context.\n\nContext:\n{context}"},
            {"role": "assistant", "content": f"Question: {question}\n\nAnswer:"},
        ]
        prompt = self.qall_tokenizer.apply_chat_template(messages, add_special_tokens=False, tokenize=False, continue_final_message=True)

        inputs = self.qall_tokenizer([prompt], return_tensors="pt", max_length=2048, truncation=True, padding='longest').to(self.device)

        with torch.no_grad():
            outputs = self.qall_model.generate(
                inputs['input_ids'], 
                attention_mask=inputs['attention_mask'], 
                do_sample=True,
                temperature=None,
                top_k=None, 
                top_p=None,
                max_new_tokens=40, 
                num_return_sequences=num_samples,
                pad_token_id=self.qall_tokenizer.eos_token_id,
                tokenizer=self.qall_tokenizer,
                stop_strings=["Distractors:\n"],
                output_logits=True,
                return_dict_in_generate=True,
            )
            stop_string_ids = self.qall_tokenizer.encode("Distractors:\n", add_special_tokens=False)

            generated_ids_sequences = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip([inputs.input_ids[0]] * num_samples, outputs['sequences'])
            ]

            generated_ids_logits = torch.transpose(torch.stack(outputs['logits']), 0, 1)

            losses = []

            for i in range(len(generated_ids_sequences)):
                end_idx_sequence = self._find_sublist_index(generated_ids_sequences[i].tolist(), stop_string_ids)
                generated_ids_sequences[i] = generated_ids_sequences[i][:end_idx_sequence]

                good_logit = generated_ids_logits[i][:end_idx_sequence]

                loss = self.loss_fn(
                        good_logit,
                        generated_ids_sequences[i],
                )
                losses.append(loss.item())

            responses = self.qall_tokenizer.batch_decode(generated_ids_sequences, skip_special_tokens=True)

        return responses, losses

    def generate_all_distractors(self, context, question, answer, num_samples):
        messages = [
            {"role": "system", "content": "You are an educational expert."},
            {"role": "user", "content": f"Generate a question, an answer and 3 distractors based on the context.\n\nContext:\n{context}"},
            {"role": "assistant", "content": f"Question: {question}\n\nAnswer: {answer}\n\nDistractors:"},
        ]
        prompt = self.qall_tokenizer.apply_chat_template(messages, add_special_tokens=False, tokenize=False, continue_final_message=True) + "\n"

        inputs = self.qall_tokenizer([prompt], return_tensors="pt", max_length=2048, truncation=True, padding='longest').to(self.device)

        with torch.no_grad():
            outputs = self.qall_model.generate(
                inputs['input_ids'], 
                attention_mask=inputs['attention_mask'], 
                do_sample=True,
                temperature=None,
                top_k=None, 
                top_p=None,
                max_new_tokens=40, 
                num_return_sequences=num_samples,
                pad_token_id=self.qall_tokenizer.eos_token_id,
                output_logits=True,
                return_dict_in_generate=True,
            )

            generated_ids_sequences = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip([inputs.input_ids[0]] * num_samples, outputs['sequences'])
            ]

            generated_ids_logits = torch.transpose(torch.stack(outputs['logits']), 0, 1)

            all_losses = []
            all_responses = []

            token_ids_with_newline = {token_id for token, token_id in self.qall_tokenizer.vocab.items() if 'Ċ' in token}

            for i in range(len(generated_ids_sequences)):
                losses = []
                distractors_ids = []
                end_idx_sequence = generated_ids_sequences[i].tolist().index(self.qall_tokenizer.eos_token_id) + 1 if self.qall_tokenizer.eos_token_id in generated_ids_sequences[i] else len(generated_ids_sequences[i])
                generated_ids_sequences[i] = generated_ids_sequences[i][:end_idx_sequence]

                good_logit = generated_ids_logits[i][:end_idx_sequence]

                indices = [i+1 for i, x in enumerate(generated_ids_sequences[i]) if x.item() in token_ids_with_newline]
                generated_ids_splitted = [generated_ids_sequences[i][j:k] for j, k in zip([0] + indices, indices + [len(generated_ids_sequences[i])])]
                generated_logits_splitted = [good_logit[j:k] for j, k in zip([0] + indices, indices + [len(good_logit)])]

                for gen_id, gen_logit in zip(generated_ids_splitted, generated_logits_splitted):
                    loss = self.loss_fn(
                        gen_logit,
                        gen_id,
                    )
                    losses.append(loss.item())
                    distractors_ids.append(gen_id)

                response = self.qall_tokenizer.batch_decode(distractors_ids, skip_special_tokens=True)
                if len(response) == 3:
                    all_losses.append(losses)
                    all_responses.append(response)

        return all_responses, all_losses
    
    def get_qa_loss(self, context, question, answers):
        losses = []

        messages_input = [
            [
                {"role": "system", "content": "You are an educational expert."},
                {"role": "user", "content": f"Generate a question, an answer and 3 distractors based on the context.\n\nContext:\n{context}"},
                {"role": "assistant", "content": f"Question: {question}\n\nAnswer:"},
            ] for _ in answers
        ]
        texts_input = [self.qall_tokenizer.apply_chat_template(messages, add_special_tokens=False, tokenize=False, continue_final_message=True) for messages in messages_input]

        messages_whole = [
            [
                {"role": "system", "content": "You are an educational expert."},
                {"role": "user", "content": f"Generate a question, an answer and 3 distractors based on the context.\n\nContext:\n{context}"},
                {"role": "assistant", "content": f"Question: {question}\n\nAnswer: {ans}\n\nDistractors:"},
            ] for ans in answers
        ]
        texts_whole = [self.qall_tokenizer.apply_chat_template(messages, add_special_tokens=False, tokenize=False, continue_final_message=True) + '\n' for messages in messages_whole]

        input_prompts = self.qall_tokenizer(texts_input, return_tensors="pt", truncation=True, max_length=2048, padding='longest').to(self.device)
        whole_prompts = self.qall_tokenizer(texts_whole, return_tensors="pt", truncation=True, max_length=2048, padding='longest').to(self.device)

        with torch.no_grad():
            outputs = self.qall_model(input_ids=whole_prompts['input_ids'], attention_mask=whole_prompts['attention_mask'])
            logits = outputs.logits

        for logit, input, whole in zip(logits, input_prompts['input_ids'], whole_prompts['input_ids']):
            # Remove padding
            padding = torch.count_nonzero(whole == self.qall_tokenizer.pad_token_id)
            whole = whole[padding:]
            padding = torch.count_nonzero(input == self.qall_tokenizer.pad_token_id)
            input = input[padding:]

            # Remove the last logit (unnecessary, automatically added by the model)
            logit = logit[:-1]

            # Get from the logits just the ones corresponding to the actual generation (label)
            good_logit = logit[-(len(whole) - len(input)):]

            # Get the label
            good_label = whole[len(input):]

            loss = self.loss_fn(
                good_logit,
                good_label,
            )
            losses.append(loss.item())
        return losses
    
    def _split_token_ids_get_text_and_loss(self, my_generated_ids, my_logits, start_prompt_text, end_prompt_text):
        start_idx_question = 0
        end_idx_question = len(my_generated_ids)

        if start_prompt_text:
            start_prompt_ids = self.qall_tokenizer.encode(start_prompt_text, add_special_tokens=False)
            start_idx_question = self._find_sublist_index(my_generated_ids.tolist(), start_prompt_ids) + len(start_prompt_ids)
        if end_prompt_text:
            end_prompt_ids = self.qall_tokenizer.encode(end_prompt_text, add_special_tokens=False)
            end_idx_question = self._find_sublist_index(my_generated_ids.tolist(), end_prompt_ids)

        good_my_generated_ids = my_generated_ids[start_idx_question:end_idx_question]
        good_my_logits = my_logits[start_idx_question:end_idx_question]

        my_loss = self.loss_fn(
            good_my_logits,
            good_my_generated_ids,
        )

        all_losses = self.loss_fn_without_reduction(
            good_my_logits,
            good_my_generated_ids,
        )

        info_objects_list = []

        for logits, token_id, loss in zip(good_my_logits, good_my_generated_ids, all_losses):
            info_objects_list.append({
                'token': self.qall_tokenizer.decode(token_id),
                'token_id': token_id.item(),
                'loss': loss.item(),
                'diff': torch.max(logits).item() - logits[token_id].item(),
                'entropy': (-torch.sum(torch.nn.functional.softmax(logits, dim=0) * torch.nn.functional.log_softmax(logits, dim=0))).item(),
            })

        good_text = self.qall_tokenizer.decode(good_my_generated_ids, skip_special_tokens=True)
        good_text = good_text.strip()

        return good_text, my_loss.item(), info_objects_list

    def generate_all_artifacts_with_explanations(self, context, num_samples):
        messages = [
            {"role": "system", "content": "You are an educational expert."},
            {"role": "user", "content": f"Generate a question, an answer and 3 distractors based on the context.\n\n\n\nContext:\n{context}"}
        ]
        prompt = self.qall_tokenizer.apply_chat_template(messages, add_special_tokens=False, tokenize=False, add_generation_prompt=True)
        inputs = self.qall_tokenizer([prompt], return_tensors="pt", max_length=2048, truncation=True, padding='longest').to(self.device)

        with torch.no_grad():
            outputs = self.qall_model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                do_sample=True,
                temperature=None,
                top_k=None,
                top_p=0.9,
                min_p=0.2,
                max_new_tokens=1000,
                num_return_sequences=num_samples,
                pad_token_id=self.qall_tokenizer.eos_token_id,
                output_logits=True,
                return_dict_in_generate=True,
            )
        
        transposed_logits = torch.transpose(torch.stack(outputs['logits']), 0, 1)

        token_ids_with_newline = {token_id for token, token_id in self.qall_tokenizer.vocab.items() if 'Ċ' in token}
        token_ids_with_two_newlines = {token_id for token, token_id in self.qall_tokenizer.vocab.items() if 'ĊĊ' in token}

        response_list = []
        for i in range(num_samples):
            response = {}
            generated_ids = outputs['sequences'][i][len(inputs.input_ids[0]):]
            generated_logits = transposed_logits[i]

            # Question
            question_text, question_loss, question_info_objects = self._split_token_ids_get_text_and_loss(
                generated_ids, generated_logits, 
                "Question:", "Answer explanation:"
            )
            response['question_text'] = question_text
            response['question_loss'] = question_loss
            response['question_info_objects'] = question_info_objects

            # Answer explanation
            answer_explanation_text, answer_explanation_loss, answer_explanation_info_objects = self._split_token_ids_get_text_and_loss(
                generated_ids, generated_logits, 
                "Answer explanation:", "Answer:"
            )
            response['answer_explanation_text'] = answer_explanation_text
            response['answer_explanation_loss'] = answer_explanation_loss
            response['answer_explanation_info_objects'] = answer_explanation_info_objects

            # Answer
            answer_text, answer_loss, answer_info_objects = self._split_token_ids_get_text_and_loss(
                generated_ids, generated_logits, 
                "Answer:", "Distractors:\n"
            )
            response['answer_text'] = answer_text
            response['answer_loss'] = answer_loss
            response['answer_info_objects'] = answer_info_objects

            # Distractors
            distractors_start_prompt_ids = self.qall_tokenizer.encode("Distractors:\n", add_special_tokens=False)

            ids = generated_ids.tolist()
            for i in range(len(ids) - 1, -1, -1):
                if ids[i] != self.qall_tokenizer.eos_token_id:
                    end_idx_distractors = min(i + 2, len(ids))
                    break
            #end_idx_distractors = find_sublist_index(generated_ids.tolist(), [qall_tokenizer.eos_token_id])
            start_idx_distractors = self._find_sublist_index(generated_ids.tolist(), distractors_start_prompt_ids) + len(distractors_start_prompt_ids)

            generated_ids_distractors = generated_ids[start_idx_distractors:end_idx_distractors]
            good_logit_distractors = generated_logits[start_idx_distractors:end_idx_distractors]

            indices = [i+1 for i, x in enumerate(generated_ids_distractors) if x.item() in token_ids_with_two_newlines]
            generated_distractors_ids_splitted = [generated_ids_distractors[j:k] for j, k in zip([0] + indices, indices + [len(generated_ids_distractors)])]
            generated_distractors_logits_splitted = [good_logit_distractors[j:k] for j, k in zip([0] + indices, indices + [len(good_logit_distractors)])]

            distractors = []

            for gen_distractor_ids, gen_distractor_logits in zip(generated_distractors_ids_splitted, generated_distractors_logits_splitted):
                # Distractor category
                distractor_category_text, distractor_category_loss, distractor_category_info_objects = self._split_token_ids_get_text_and_loss(gen_distractor_ids, gen_distractor_logits, "Distractor category:", "Distractor explanation:")

                # Distractor explanation
                distractor_explanation_text, distractor_explanation_loss, distractor_explanation_info_objects = self._split_token_ids_get_text_and_loss(gen_distractor_ids, gen_distractor_logits, "Distractor explanation:", "Distractor:")
                
                # Distractor
                distractor_text, distractor_loss, distractor_info_objects = self._split_token_ids_get_text_and_loss(gen_distractor_ids, gen_distractor_logits, "Distractor:", None)
                
                distractors.append({
                    'distractor_category_text': distractor_category_text,
                    'distractor_category_loss': distractor_category_loss,
                    'distractor_category_info_objects': distractor_category_info_objects,
                    'distractor_explanation_text': distractor_explanation_text,
                    'distractor_explanation_loss': distractor_explanation_loss,
                    'distractor_explanation_info_objects': distractor_explanation_info_objects,
                    'distractor_text': distractor_text,
                    'distractor_loss': distractor_loss,
                    'distractor_info_objects': distractor_info_objects,
                })

            response['distractors'] = distractors

            response_list.append(response)

        return response_list
    
    def generate_all_artifacts_without_explanations(self, context, num_samples):
        messages = [
            {"role": "system", "content": "You are an educational expert."},
            {"role": "user", "content": f"Generate a question, an answer and 3 distractors based on the context.\n\nContext:\n{context}"}
        ]
        prompt = self.qall_tokenizer.apply_chat_template(messages, add_special_tokens=False, tokenize=False, add_generation_prompt=True)
        inputs = self.qall_tokenizer([prompt], return_tensors="pt", max_length=2048, truncation=True, padding='longest').to(self.device)

        with torch.no_grad():
            outputs = self.qall_model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                do_sample=True,
                temperature=None,
                top_k=None,
                top_p=0.9,
                min_p=0.2,
                max_new_tokens=1000,
                num_return_sequences=num_samples,
                pad_token_id=self.qall_tokenizer.eos_token_id,
                output_logits=True,
                return_dict_in_generate=True,
            )
        
        transposed_logits = torch.transpose(torch.stack(outputs['logits']), 0, 1)

        token_ids_with_newline = {token_id for token, token_id in self.qall_tokenizer.vocab.items() if 'Ċ' in token}
        token_ids_with_two_newlines = {token_id for token, token_id in self.qall_tokenizer.vocab.items() if 'ĊĊ' in token}

        response_list = []
        for i in range(num_samples):
            response = {}
            generated_ids = outputs['sequences'][i][len(inputs.input_ids[0]):]
            generated_logits = transposed_logits[i]

            # Question
            question_text, question_loss, question_info_objects = self._split_token_ids_get_text_and_loss(
                generated_ids, generated_logits, 
                "Question:", "Answer:"
            )
            response['question_text'] = question_text
            response['question_loss'] = question_loss
            response['question_info_objects'] = question_info_objects

            # Answer
            answer_text, answer_loss, answer_info_objects = self._split_token_ids_get_text_and_loss(
                generated_ids, generated_logits, 
                "Answer:", "Distractors:\n"
            )
            response['answer_text'] = answer_text
            response['answer_loss'] = answer_loss
            response['answer_info_objects'] = answer_info_objects

            # Distractors
            distractors_start_prompt_ids = self.qall_tokenizer.encode("Distractors:\n", add_special_tokens=False)

            ids = generated_ids.tolist()
            for i in range(len(ids) - 1, -1, -1):
                if ids[i] != self.qall_tokenizer.eos_token_id:
                    end_idx_distractors = min(i + 2, len(ids))
                    break
            #end_idx_distractors = find_sublist_index(generated_ids.tolist(), [qall_tokenizer.eos_token_id])
            start_idx_distractors = self._find_sublist_index(generated_ids.tolist(), distractors_start_prompt_ids) + len(distractors_start_prompt_ids)

            generated_ids_distractors = generated_ids[start_idx_distractors:end_idx_distractors]
            good_logit_distractors = generated_logits[start_idx_distractors:end_idx_distractors]

            indices = [i+1 for i, x in enumerate(generated_ids_distractors) if x.item() in token_ids_with_newline]
            generated_distractors_ids_splitted = [generated_ids_distractors[j:k] for j, k in zip([0] + indices, indices + [len(generated_ids_distractors)])]
            generated_distractors_logits_splitted = [good_logit_distractors[j:k] for j, k in zip([0] + indices, indices + [len(good_logit_distractors)])]

            distractors = []

            for gen_distractor_ids, gen_distractor_logits in zip(generated_distractors_ids_splitted, generated_distractors_logits_splitted):
                distractor_text, distractor_loss, distractor_info_objects = self._split_token_ids_get_text_and_loss(gen_distractor_ids, gen_distractor_logits, None, None)
                
                distractors.append({
                    'distractor_text': distractor_text,
                    'distractor_loss': distractor_loss,
                    'distractor_info_objects': distractor_info_objects,
                })

            response['distractors'] = distractors

            response_list.append(response)

        return response_list

    
    def generate_all_artifacts(self, context, num_samples):
        messages = [
            {"role": "system", "content": "You are an educational expert."},
            {"role": "user", "content": f"Generate a question, an answer and 3 distractors based on the context.\n\nContext:\n{context}"},
        ]
        prompt = self.qall_tokenizer.apply_chat_template(messages, add_special_tokens=False, tokenize=False, add_generation_prompt=True)
        inputs = self.qall_tokenizer([prompt], return_tensors="pt", max_length=2048, truncation=True, padding='longest').to(self.device)

        with torch.no_grad():
            outputs = self.qall_model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                do_sample=True,
                temperature=None,
                top_k=None,
                top_p=0.9,
                max_new_tokens=1000,
                num_return_sequences=num_samples,
                pad_token_id=self.qall_tokenizer.eos_token_id,
                output_logits=True,
                return_dict_in_generate=True,
            )
        
        transposed_logits = torch.transpose(torch.stack(outputs['logits']), 0, 1)

        question_losses = []
        question_string = []
        answer_losses = []
        answer_string = []
        distractor_set_losses = []
        distractor_set_string = []
        token_ids_with_newline = {token_id for token, token_id in self.qall_tokenizer.vocab.items() if 'Ċ' in token}

        for i in range(num_samples):
            generated_ids = outputs['sequences'][i][len(inputs.input_ids[0]):]
            generated_logits = transposed_logits[i]


            # Question
            question_start_prompt_ids = self.qall_tokenizer.encode("Question:", add_special_tokens=False)
            question_end_prompt_ids = self.qall_tokenizer.encode("Answer:", add_special_tokens=False)
            end_idx_question = self._find_sublist_index(generated_ids.tolist(), question_end_prompt_ids)
            start_idx_question = self._find_sublist_index(generated_ids.tolist(), question_start_prompt_ids) + len(question_start_prompt_ids)

            generated_ids_question = generated_ids[start_idx_question:end_idx_question]
            good_logit_question = generated_logits[start_idx_question:end_idx_question]

            loss_question = self.loss_fn(
                good_logit_question,
                generated_ids_question,
            )

            question = self.qall_tokenizer.decode(generated_ids_question, skip_special_tokens=True)
            question = question.strip()
            question_string.append(question)
            question_losses.append(loss_question.item())

            # Answer
            answer_start_prompt_ids = self.qall_tokenizer.encode("Answer:", add_special_tokens=False)
            answer_end_prompt_ids = self.qall_tokenizer.encode("Distractors:\n", add_special_tokens=False)
            end_idx_answer = self._find_sublist_index(generated_ids.tolist(), answer_end_prompt_ids)
            start_idx_answer = self._find_sublist_index(generated_ids.tolist(), answer_start_prompt_ids) + len(answer_start_prompt_ids)

            generated_ids_answer = generated_ids[start_idx_answer:end_idx_answer]
            good_logit_answer = generated_logits[start_idx_answer:end_idx_answer]

            loss_answer = self.loss_fn(
                good_logit_answer,
                generated_ids_answer,
            )

            answer = self.qall_tokenizer.decode(generated_ids_answer, skip_special_tokens=True)
            answer = answer.strip()
            answer_string.append(answer)
            answer_losses.append(loss_answer.item())

            # Distractors
            distractors_start_prompt_ids = self.qall_tokenizer.encode("Distractors:\n", add_special_tokens=False)

            ids = generated_ids.tolist()
            for i in range(len(ids) - 1, -1, -1):
                if ids[i] != self.qall_tokenizer.eos_token_id:
                    end_idx_distractors = min(i + 2, len(ids))
                    break
            #end_idx_distractors = find_sublist_index(generated_ids.tolist(), [qall_tokenizer.eos_token_id])
            start_idx_distractors = self._find_sublist_index(generated_ids.tolist(), distractors_start_prompt_ids) + len(distractors_start_prompt_ids)

            generated_ids_distractors = generated_ids[start_idx_distractors:end_idx_distractors]
            good_logit_distractors = generated_logits[start_idx_distractors:end_idx_distractors]

            indices = [i+1 for i, x in enumerate(generated_ids_distractors) if x.item() in token_ids_with_newline]
            generated_ids_splitted = [generated_ids_distractors[j:k] for j, k in zip([0] + indices, indices + [len(generated_ids_distractors)])]
            generated_logits_splitted = [good_logit_distractors[j:k] for j, k in zip([0] + indices, indices + [len(good_logit_distractors)])]

            losses_distractors = []
            distractors = []

            for gen_id, gen_logit in zip(generated_ids_splitted, generated_logits_splitted):
                loss = self.loss_fn(
                    gen_logit,
                    gen_id,
                )
                losses_distractors.append(loss.item())
                distractors.append(self.qall_tokenizer.decode(gen_id, skip_special_tokens=True).strip())

            distractor_set_losses.append(losses_distractors)
            distractor_set_string.append(distractors)

        response_list = []
        for q, q_loss, a, a_loss, d, d_loss in zip(question_string, question_losses, answer_string, answer_losses, distractor_set_string, distractor_set_losses):
            #if max(d_loss) > 20:
            #    continue
            response_list.append({
                "question": q,
                "qgen_loss": q_loss,
                "answer": a,
                "qa_loss": a_loss,
                "distractors": d,
                "dgen_loss": d_loss,
            })

        return response_list
