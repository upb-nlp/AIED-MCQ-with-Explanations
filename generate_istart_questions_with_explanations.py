from question_generation_utils import QuestionGenerationUtils
from question_filter_utils import QuestionFilterUtils
import json
from tqdm import tqdm

QG = QuestionGenerationUtils('cuda', 'ORGANIZATION-NAME/llama3.1_8b_qall_with_explanations', 'TOKEN_KEY')

istart_dataset = json.load(open("AIED-MCQ-with-Explanations/istart_human_questions.json"))
new_dataset = {}

for filename, data in tqdm(istart_dataset.items()):
    new_dataset[filename] = {}
    new_dataset[filename]['context'] = data['context']
    
    context = data['context']
    res = []
    for i in range(3):
        res += QG.generate_all_artifacts_with_explanations(context=context, num_samples=20)
        
    new_dataset[filename]['responses'] = res

del QG

FQ = QuestionFilterUtils('cuda')

for filename, data in tqdm(new_dataset.items()):
    res = data['responses']
    context = data['context']
    filtered_res = FQ.filter_questions(res, context)
    clean_res = FQ.clean_response_dict(filtered_res)

    new_dataset[filename]['responses'] = clean_res
    
json.dump(new_dataset, open("AIED-MCQ-with-Explanations/istart_questions_with_explanations.json", "w"), indent=4)
