import json
import re
import random
random.seed(42)

def remove_funny_race_questions(dataset):
    choices = [
        "a personal diary",
        "a news report",
        "a TV interview",
        "a book review",
        "a story book",
        "a travel magazine",
        "a letter",
        "an advertisement",
        "a newspaper article",
        "a newspaper",
        "children's magazine",
        "a short-story collection",
        "an introduction to tourist attractions",
        "an accident record",
        "a research paper",
        "a geography book",
        "An advertisement.",
        "A health report.",
        "Science fiction.",
        "a popular science magazine",
        "a novel",
        "a personal blog",
        "a letter between friends",
        "a speech on new technology",
        "an official paper",
    ]
    choices = set([c.lower() for c in choices])
    
    for data in dataset:
        data['to_keek'] = True
        if re.search(r'\btitle\b', data['question']):
            data['to_keek'] = random.random() < 0.1
        if re.search(r'\btitle\b.*\bpassage\b', data['question']):
            data['to_keek'] = random.random() < 0.1
        if re.search(r'\bpassage\b.*\bmainly\b', data['question']):
            data['to_keek'] = random.random() < 0.2
        
        if 'distractors' in data:
            my_choices = data['distractors'] + [data['answer']]
            my_choices = set([c.lower() for c in my_choices])
            if my_choices.intersection(choices):
                data['to_keek'] = random.random() < 0.1
            
    return [d for d in dataset if d['to_keek']]

def prepare_qgen_qans():
    new_datasets = {}
    for PARTITION in ['train', 'test', 'val']:
        new_datasets[PARTITION] = []
        for data_name in ['fairytaleqa', 'hotpotqa', 'narrativeqa', 'squad']:
            dataset = json.load(open(f'AIED-MCQ-with-Explanations/datasets_q/reshaped_{data_name}_{PARTITION}.json'))
            dataset = remove_funny_race_questions(dataset)
            for data in dataset:
                new_datasets[PARTITION].append({
                    'prompt': f"Generate a question and an answer based on the context.\n\nContext:\n{data['context']}",
                    'generated_prompt': f"Question: {data['question']}\n\nAnswer: {data['answer']}",
                    'task': 'qgen_qans',
                })
    return new_datasets

def prepare_qgen_qans_dgen():
    new_datasets = {}
    for PARTITION in ['train', 'test', 'val']:
        new_datasets[PARTITION] = []
        for data_name in ['mctest', 'race', 'eduqg']:
            dataset = json.load(open(f'AIED-MCQ-with-Explanations/datasets_q/reshaped_{data_name}_{PARTITION}.json'))
            dataset = remove_funny_race_questions(dataset)
            for data in dataset:
                joined_distractors = '\n'.join(data['distractors'])
                new_datasets[PARTITION].append({
                    'prompt': f"Generate a question, an answer and 3 distractors based on the context.\n\nContext:\n{data['context']}",
                    'generated_prompt': f"Question: {data['question']}\n\nAnswer: {data['answer']}\n\nDistractors:\n{joined_distractors}",
                    'task': 'qgen_qans_dgen',
                })
    return new_datasets

def clsq(seq):
    seq = seq.lstrip('.,?!-#')
    seq = seq.strip()
    return seq

def prepare_qgen_qans_with_explanations():
    new_datasets = {}
    for PARTITION in ['train', 'val']:
        new_datasets[PARTITION] = []
        for data_name in ['fairytaleqa', 'hotpotqa', 'narrativeqa', 'squad']:
            dataset = json.load(open(f'AIED-MCQ-with-Explanations/datasets_synthetic_annotated/reshaped_{data_name}_{PARTITION}_annotated.json'))
            dataset = remove_funny_race_questions(dataset)
            for data in dataset:
                try:
                    new_datasets[PARTITION].append({
                        'prompt': f"Generate a question and an answer based on the context.\n\n\n\nContext:\n{data['context']}",
                        'generated_prompt': f"Question: {data['question']}\n\n\n\nAnswer explanation: {data['answer_explanation']}\n\n\n\nAnswer: {data['answer']}",
                        'task': 'qgen_qans_with_explanations',
                        'data_name': data_name,
                    })
                except:
                    pass
    return new_datasets

def prepare_qgen_qans_dgen_with_explanations():
    new_datasets = {}
    for PARTITION in ['train', 'val']:
        new_datasets[PARTITION] = []
        for data_name in ['mctest', 'race', 'eduqg']:
            if PARTITION == 'val' and data_name == 'eduqg':
                continue
            dataset = json.load(open(f'AIED-MCQ-with-Explanations/datasets_synthetic_annotated/reshaped_{data_name}_{PARTITION}_annotated.json'))
            dataset = remove_funny_race_questions(dataset)
            for data in dataset:
                try:
                    joined_distractors = '\n\n'.join([f"Distractor category: {clsq(dc)}\nDistractor explanation: {clsq(de)}\nDistractor: {d}" for dc, de, d in zip(data['distractor_categories'], data['distractor_explanations'], data['distractors'])])
                    new_datasets[PARTITION].append({
                        'prompt': f"Generate a question, an answer and {len(data['distractors'])} distractors based on the context.\n\n\n\nContext:\n{data['context']}",
                        'generated_prompt': f"Question: {data['question']}\n\n\n\nAnswer explanation: {data['answer_explanation']}\n\n\n\nAnswer: {data['answer']}\n\n\n\nDistractors:\n{joined_distractors}",
                        'task': 'qgen_qans_dgen_with_explanations',
                        'data_name': data_name,
                    })
                except:
                    pass
    return new_datasets