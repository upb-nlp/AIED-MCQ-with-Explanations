from openai import OpenAI
import json
from tqdm import tqdm
from pydantic import BaseModel

client = OpenAI(
    api_key="TOKEN_KEY",
)

class QuizItem(BaseModel):
    question: str
    answer_explanation: str
    answer: str
    distractors: list[str]
    distractors_explanation: list[str]
    
class Quiz(BaseModel):
    quiz: list[QuizItem]

def generate_response(model_inputs):
    completion = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an educational expert."},
            {"role": "user", "content": model_inputs}
        ],
        seed=42,
        response_format=Quiz,
    )

    response = completion.choices[0].message.content
    
    response = json.loads(response)

    return response["quiz"]

task_description_prompt = """Based on the following context, generate 5 multiple-choice quiz items. Return your answer in JSON format. Return a JSON array of 5 objects. Each object corresponds to a quiz item. Each object will be like this: 
{"question": the question, "answer_explanation": the explanation for the correct answer, "answer": the correct answer, "distractors": [list with 3 distractors], "distractors_explanation": [list with why each of the 3 distractors are an incorrect answer]}
Return a JSON array of 5 objects.

Context:
"""

istart_dataset = json.load(open("AIED-MCQ-with-Explanations/istart_human_questions.json"))
new_dataset = {}

for filename, data in tqdm(istart_dataset.items()):
    new_dataset[filename] = {}
    new_dataset[filename]['context'] = data['context']
    
    try:
        res = generate_response(task_description_prompt + data['context'])
    except:
        res = None
        
    new_dataset[filename]['responses'] = res
    
json.dump(new_dataset, open("AIED-MCQ-with-Explanations/istart_questions_gpt.json", "w"), indent=4)