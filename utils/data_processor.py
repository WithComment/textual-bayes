import textgrad as tg
from textgrad.variable import Variable


def prepare_multi_choice_example(example):
    choices = "\n".join(f"{chr(65 + i)}. {ans}" for i, ans in enumerate(example["answers"]))
    question_text = f"{example['question']}\n\n{choices}"
    correct_answer = chr(65 + example["label"])

    x = tg.Variable(
        question_text,
        requires_grad=False,
        role_description="multiple choice question",
    )
    y_true = tg.Variable(
        correct_answer,
        requires_grad=False,
        role_description="correct answer for the multiple choice question",
    )
    return x, y_true


def prepare_integer_math_example(example):
    question_text = f"{example['question']}"
    if "####" in str(example["answer"]):
        correct_answer = str(int(str(example["answer"]).split("####")[1].strip().replace(",", "")))
    else:
        correct_answer = str(int(str(example["answer"])))

    x = Variable(
        question_text,
        requires_grad=False,
        role_description="math question",
    )
    y_true = Variable(
        correct_answer,
        requires_grad=False,
        role_description="correct answer for the math question",
    )
    return x, y_true


def prepare_qasper_example(example):
    question_text = f"{example['question']}"
    context_text = f"{example['context']}"
    formatted_question = f"Context: {context_text}\n\n Question: {question_text}\n\n"
    question_type = example["question_type"]
    if question_type != "freeform":
        print(f"Noticed question type {question_type}")

    unanswerable_flag = example["unanswerable"]
    if unanswerable_flag:
        correct_answer = "Unanswerable"
    else:
        correct_answer = example["answer"]

    x = Variable(
        formatted_question,
        requires_grad=False,
        role_description="open-ended question with context/evidence",
    )
    y_true = Variable(
        correct_answer,
        requires_grad=False,
        role_description="correct answer for the open-ended question",
    )
    return x, y_true


def prepare_simpleqa_example(example):
    x = Variable(
        example["question"],
        requires_grad=False,
        role_description="open-ended question",
    )
    y_true = Variable(
        example["answer"],
        requires_grad=False,
        role_description="correct answer for the open-ended question",
    )
    return x, y_true

import textgrad as tg
from textgrad.variable import Variable

def prepare_agreement_example(example):
    # Updated template with "Answer: agree/disagree" format
    template = """
[POSITIVE EXAMPLE]
<|im_start|>user
[QUESTION]
What is the capital of China?
(A) Shanghai
(B) Chongqing
(C) Beijing

[RESPONSE]
The capital of China is Beijing, which is option (C). {{C}}

[FINAL ANSWER]
C<|im_end|>
<|im_start|>assistant
Beijing is the capital of China, so I agree with the final answer.
Answer: agree<|im_end|>

[NEGATIVE EXAMPLE]
<|im_start|>user
[QUESTION]
Which city is the largest by population in Canada?
(A) Toronto
(B) Vancouver
(C) Ottawa

[RESPONSE]
The capital is usually the most populated city. Ottawa is the capital of Canada, so the answer is (C) Ottawa.

[FINAL ANSWER]
C<|im_end|>
<|im_start|>assistant
Ottawa is not the largest city by population in Canada; Toronto is. Therefore, I disagree with the final answer.
Answer: disagree<|im_end|>
---

[QUESTION]
{question}

[RESPONSE]
{response}

[FINAL ANSWER]
{prediction}
"""
    formatted_prompt = template.format(
        question=example['q'],
        response=example['response'],
        prediction=example['prediction']
    )
    
    x = Variable(
        formatted_prompt,
        requires_grad=False,
        role_description="context with question, response, and prediction"
    )
    
    # Target is now just the word, lowercase
    target = "disagree" if example['should_disagree'] else "agree"
    
    y_true = Variable(
        target,
        requires_grad=False,
        role_description="correct verdict (agree or disagree)"
    )
    
    return x, y_true