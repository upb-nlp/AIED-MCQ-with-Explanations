# Prompts
## Dataset Augmentation Prompt
```
Given a context and a question with distractors, you have to name a general mistake/misconception behind each of the distractors. Use the following classification for the mistake categories:

# Conceptual Misunderstanding:
* Misinterpreting fundamental principles or core ideas of a subject.
* Examples include confusing terms like "acidophiles" and "alkaliphiles" or misunderstanding accounting principles like "FIFO" and "LIFO."

# Overgeneralization:
* Applying characteristics of one category or context broadly without understanding the nuances.
* For instance, assuming that all microorganisms can thrive in extreme conditions without specifics.

# Terminological Confusion:
* Mistaking similar-sounding terms or concepts for one another.
* Examples: Confusing "osteoblasts" and "osteoclasts," or mistaking "thyroxine" for "TSH."

# Functional Misattribution:
* Attributing the role or function of one entity to another.
* Example: Assigning hormone regulation roles to unrelated glands or assuming functions for bones outside their skeletal classification.

# Overlooking Contextual Details:
* Ignoring specific details provided in the context that clarify a concept.
* Example: Misjudging the scope of historical events or biological features due to skipping over key qualifiers.

# Simplification of Relationships:
* Reducing complex interactions into overly simplistic cause-effect relationships.
* Example: Thinking ATP functions only in one particular aspect without its broader role in energy storage.

# Chronological/Process Missteps:
* Misunderstanding sequences or temporal relationships in processes.
* Example: Confusing the stages in biological cycles or historical progressions.

# Misclassification:
* Categorizing items, organisms, or processes under incorrect headings.
* Example: Mixing up elements of axial and appendicular skeletons.

Use the following format: "Distractor i -> {{category}}: {{response}}", with one distractor on each line. Similarly, add a line with an explanation why the Answer is the correct one in the format "Answer -> {{response}}".

Context: {context}

Question: {question}

Answer: {answer}

Distractor 1: {distractor_1}
Distractor 2: {distractor_2}
Distractor 3: {distractor_3}
```
## Training Prompt with Explanations
```
You are an educational expert.

Generate a question, an answer and {no_distractors} distractors based on the context.



Context:
{context}



Question: {question}



Answer explanation: {answer_explanation}



Answer: {answer}



Distractors:
Distractor category: {distractor_category}
Distractor explanation: {distractor_explanation}
Distractor: {distractor}

Distractor category: {distractor_category}
Distractor explanation: {distractor_explanation}
Distractor: {distractor}

Distractor category: {distractor_category}
Distractor explanation: {distractor_explanation}
Distractor: {distractor}

...
```