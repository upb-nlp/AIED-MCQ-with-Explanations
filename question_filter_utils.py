from minicheck.minicheck import MiniCheck
from sentence_transformers import SentenceTransformer, util

class QuestionFilterUtils:
    def __init__(self, device):
        if device not in ['cpu', 'cuda', 'mps']:
            raise ValueError(f"Invalid device: {device}. Must be one of 'cpu', 'cuda', 'mps'.")
        
        self.device = device
        self.scorer = MiniCheck(model_name='Bespoke-MiniCheck-7B', enable_prefix_caching=False, max_model_len=16000, cache_dir='./ckpts')
        self.emb_model = SentenceTransformer('all-MiniLM-L6-v2').to(self.device)

    def _calculate_sentence_embeddings(self, sentences_list):
        embeddings = self.emb_model.encode(sentences_list, convert_to_tensor=True)
        embeddings_list = [emb for emb in embeddings]
        return embeddings_list

    def _get_cosine_similarity(self, emb1, emb2):
        return util.pytorch_cos_sim(emb1, emb2).item()

    def _filter_loss_higher_than_entropy(self, question, margin_threshold):
        for field in ["question_info_objects", "answer_info_objects", "answer_explanation_info_objects"]:
            if field not in question:
                continue
            if any(entry["loss"] > entry["entropy"] + margin_threshold for entry in question[field]):
                return True
        for distractor in question["distractors"]:
            for field in ["distractor_explanation_info_objects", "distractor_info_objects"]:
                if field not in distractor:
                    continue
                if any(entry["loss"] > entry["entropy"] + margin_threshold for entry in distractor[field]):
                    return True
        return False

    def _filter_high_diff_value(self, res, margin_threshold):
        for field in ["question_info_objects", "answer_info_objects"]:
            if any(entry["diff"] > margin_threshold for entry in res[field]):
                return True
        for distractor in res["distractors"]:
            if any(entry["diff"] > margin_threshold for entry in distractor["distractor_info_objects"]):
                return True
        return False

    def _filter_similar_distractors(self, distractors, embeddings, ):
        for i in range(len(distractors)):
            for j in range(i + 1, len(distractors)):
                if self._get_cosine_similarity(embeddings[i], embeddings[j]) >= 0.8:
                    return True
        return False

    def _eliminate_duplicates_questions(self, questions, embeddings):
        response_filtered = [(questions[0], embeddings[0])]
        for i in range(1, len(questions)):
            duplicate = False
            for j in range(len(response_filtered)):
                if self._get_cosine_similarity(embeddings[i], response_filtered[j][1]) >= 0.8:
                    duplicate = True
                    break
            if not duplicate:
                response_filtered.append((questions[i], embeddings[i]))

        return [res[0] for res in response_filtered]
    
    def _filter_out_loss_higher_than_entropy(self, responses):
        new_responses = []
        for response in responses:
            if not self._filter_loss_higher_than_entropy(response, 1.1):
                new_responses.append(response)
        return new_responses
    
    def _filter_out_similar_distractors(self, responses):
        new_responses = []
        for response in responses:
            distractors_text = [d['distractor_text'] for d in response['distractors']]
            distractors_embeddings = self._calculate_sentence_embeddings(distractors_text)
            if not self._filter_similar_distractors(distractors_text, distractors_embeddings):
                new_responses.append(response)
        return new_responses
    
    def _filter_out_invalid_options(self, responses, context):
        docs = []
        claims = []

        for data in responses:
            docs.append(context)
            claims.append(f"\"{data['answer_text']}\" is the correct answer to the question: \"{data['question_text']}\"")
            for distractor in data['distractors']:
                docs.append(context)
                claims.append(f"\"{distractor['distractor_text']}\" is the correct answer to the question: \"{data['question_text']}\"")

        _, raw_prob, _, _ = self.scorer.score(docs=docs, claims=claims)

        i = 0
        for data in responses:
            data['answer_prob'] = raw_prob[i]
            i += 1
            for distractor in data['distractors']:
                distractor['distractor_prob'] = raw_prob[i]
                i += 1

        new_responses = []
        for data in responses:
            fact_checking_diffs = [data['answer_prob'] - ds['distractor_prob'] for ds in data['distractors']]
            if all(diff > 0.1 for diff in fact_checking_diffs):
                new_responses.append(data)
        return new_responses

    def filter_questions(self, responses, context):
        responses = self._filter_out_loss_higher_than_entropy(responses)
        responses = self._filter_out_similar_distractors(responses)
        responses = self._filter_out_invalid_options(responses, context)
        
        if 'answer_explanation_loss' in responses[0]:
            responses = sorted(responses, key=lambda x: x['question_loss'] + x['answer_explanation_loss'] + x['answer_loss'] + sum(d['distractor_category_loss'] for d in x['distractors']) + sum(d['distractor_explanation_loss'] for d in x['distractors']) + sum(d['distractor_loss'] for d in x['distractors']))
        else:
            responses = sorted(responses, key=lambda x: x['question_loss'] + x['answer_loss'] + sum(d['distractor_loss'] for d in x['distractors']))


        question_embeddings = self._calculate_sentence_embeddings([res['question_text'] + "\n" + res["answer_text"] for res in responses])
        responses = self._eliminate_duplicates_questions(responses, question_embeddings)

        return responses
    
    def clean_response_dict(self, responses):
        new_responses = []
        for response in responses:
            new_response = {
                "question": response["question_text"],
                "answer": response["answer_text"],
                "distractors": [d["distractor_text"] for d in response["distractors"]]
            }
            if 'answer_explanation_text' in response:
                new_response["answer_explanation"] = response["answer_explanation_text"]
            if 'distractor_explanation_text' in response['distractors'][0]:
                new_response["distractor_explanation"] = [d["distractor_explanation_text"] for d in response["distractors"]]
            new_responses.append(new_response)
        return new_responses