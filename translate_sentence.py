from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import time

model_name = 'facebook/nllb-200-distilled-600M'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to("cuda").eval()

def translate_en_to_ko(sentences):
    # text를 토큰으로 변환
    inputs = tokenizer(sentences, return_tensors='pt', padding=True, truncation=True).to("cuda")

    # 변환한 토큰을 한국어로 번역한 토큰 생성
    translated_tokens = model.generate(
        **inputs, 
        forced_bos_token_id=tokenizer.convert_tokens_to_ids('kor_Hang'),
        num_beams=6
    )

    # 토큰을 다시 텍스트로 변환
    translated_sentences = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
    return translated_sentences


if __name__ == '__main__':
    text = "How's the weather?"
    sentences = [
        'People think this man deserves two medals instead of one, because right after completing this race, the officials told him to go back and finish it once more.', 
        'But why?', 
        "At first everyone thought it was because he broke the rules or something, but that wasn't the case.", 
        "You see, the reason why they told him to finish it again was because he completed it so fast that they couldn't even set up to finish line ribbon."
    ]    
    # start = time.time()
    translated_sentences = translate_en_to_ko(sentences)
    print(translated_sentences)
    # end = time.time()
    # print(f'{end - start: .5f} sec')