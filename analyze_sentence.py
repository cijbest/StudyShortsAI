from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import ollama
import torch
import torch.nn.functional as F
import json, time

level_map = {
    "A1": "Elementary",
    "A2": "Elementary",
    "B1": "Intermediate",
    "B2": "Intermediate",
    "C1": "Advanced",
    "C2": "Advanced"
}

def assess_level(sentences) :
    model_name = "AbdulSami/bert-base-cased-cefr"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to("cuda").eval()

    # 토큰화
    inputs = tokenizer(sentences, return_tensors='pt', padding=True, truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()} # 토크나이즈한 결과(input_ids, attention_mask)를 모델이 올라가 있는 GPU로 전송

    # 예측
    with torch.no_grad(): # 추론 시 불필요한 gradient 계산을 끄는 모드 (메모리 절약 & 빠른 실행)
        outputs = model(**inputs) # 모델에 입력을 넣어 예측 결과 받기
        probs = F.softmax(outputs.logits, dim=1) # 모델이 출력한 logits(CEFR 각 레벨에 대한 점수)을 softmax로 확률로 변환
        preds = torch.argmax(probs, dim=1) # 확률이 가장 높은 인덱스를 뽑아서 최종 예측값으로 사용

    # 레벨 매핑 리스트 (A1~C2)
    label_map = ["A1", "A2", "B1", "B2", "C1", "C2"]

    assessed_levels = []
    for i, sentence in enumerate(sentences):
        assessed_levels.append(f"{level_map[label_map[preds[i]]]}({label_map[preds[i]]})")

    return assessed_levels

    ### 배치처리 X ###

    # 파이프라인으로 실행
    # difficulty_classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

    # 분류 실행 (CERT 기준 : A1, A2, B1, B2, C1, C2)
    # level = difficulty_classifier(text)[0]["label"]

    # "Elementary", "Intermediate", "Advanced" 도 함께 출력
    # print(f"{level_map.get(level, 'Unknown')} ({level})")

def extract_key_expressions(sentences) :

    # LLM에게 요청할 프롬프트 작성
    prompt = f"""
            You are an expert in English linguistics.  
            Identify all Multiword Expressions (MWEs) in each given English sentence.

            Definition: A Multiword Expression is a group of words that together form a single meaning that cannot be understood by analyzing each word individually. These include:

            - Phrasal verbs
            - Idioms
            - Fixed expressions
            - Prepositional phrases with idiomatic meaning
            - Conjunction phrases and discourse markers

            Rules:
            - Extract all MWEs present in the sentence; if multiple expressions of the same or different types exist, include them all.
            - Do NOT include literal or compositional phrases.
            - Extract the minimal core expression only, without extra words or context.
            - Only include expressions that can be found in the Cambridge Dictionary. If the expression is not listed there, skip it.
            - The "meaning" field should provide the dictionary-style definition of the expression.

            Output format:  
            Return a list of objects, each representing a sentence and its MWEs:
            [
            {{
            "sentence": "Original English sentence here",
            "MWEs": [
                {{"expression": "expression1", "meaning": "meaning1"}},
                {{"expression": "expression2", "meaning": "meaning2"}},
                ...
            ]
            }},
            ...
            ]

            Respond with JSON output ONLY. No extra text, explanation, or notes.

            Process the following sentences : {sentences}

    """

    # ollama에서 사용할 LLM 설정
    response = ollama.chat(model="llama3", messages=[
        {"role": "user", "content": prompt}
    ])

    # 결과값 반환
    result = response["message"]["content"].strip()
    
    # 결과값 타입 변환(json(str) -> list)
    return json.loads(result)



    ### 번역 모델을 이용하는 경우 ###
    # # 결과값 타입 변환(json(str) -> list)
    # data = json.loads(result)

    # meaning 값 추가
    # for group in data:
    #     for item in group["expressions"]:
    #         item["meaning"] = translate_text_facebook3B.translate_en_to_ko(item["expression"])
    # print(data)


if __name__ == '__main__':
    # 예문
    # text = "Having been let down by his closest friend, Jack struggled to hold back his emotions while trying to figure out how to move on with his life."
    text = "Despite being brought up in a small town where opportunities were scarce, Emily managed to pull through the challenges, holding on to her dreams even when things seemed to fall apart around her."
    sentences = [
        'People think this man deserves two medals instead of one, because right after completing this race, the officials told him to go back and finish it once more.', 
        'But why?', 
        "At first everyone thought it was because he broke the rules or something, but that wasn't the case.", 
        "You see, the reason why they told him to finish it again was because he completed it so fast that they couldn't even set up to finish line ribbon."
    ]

    # 영어 문장 수준 측정
    # start = time.time()
    # assessed_levels = assess_level(sentences)
    # end = time.time()
    # print(f'{end - start: .5f} sec')
    # print(assessed_levels)

    # 영어 문장 내 숙어 표현 추출
    start = time.time()

    key_expressions_list = extract_key_expressions(sentences)
    print(key_expressions_list)

    end = time.time()
    print(f'{end - start: .5f} sec')


    test = """[
        {
            "sentence": "People think this man deserves two medals instead of one, because right after completing this race, the officials told him to go back and finish it once more.",
            "MWEs": [
            {"expression": "right after", "meaning": "immediately following"},
            {"expression": "go back", "meaning": "to return or revert"}
            ]
        },
        {
            "sentence": "But why?",
            "MWEs": [
            {"expression": "why", "meaning": "for what reason"}
            ]
        },
        {
            "sentence": "At first everyone thought it was because he broke the rules or something, but that wasn't the case.",
            "MWEs": []
        },
        {
            "sentence": "You see, the reason why they told him to finish it again was because he completed it so fast that they couldn't even set up to finish line ribbon.",
            "MWEs": [
            {"expression": "set up", "meaning": "to prepare or establish"}
            ]
        }
        ]"""

    # print(type(json.loads(test)))
