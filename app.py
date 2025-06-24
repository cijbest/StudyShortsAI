import streamlit as st
import audio_to_text, translate_sentence, analyze_sentence
import torch, time

torch.classes.__path__ = [] # add this line to manually set it to empty. 

st.title("StudyShorts AI ")

url = st.text_input("YouTube Shorts 링크") # https://youtube.com/shorts/UmDQcoDoCRg?si=OpT4F2A4gnBggFzK

analyze_button = st.button("분석 시작")

DATA_KEY = "analysis_data"
SELECTED_TEXT_KEY = "selected_text"

if analyze_button:
    if url != '' and "shorts" in url:
        with st.status("분석 중입니다...", expanded=True) as status:
            audio_filename = 'audio.m4a'
            # start = time.time()
            # end = time.time()
            # print(f'{end - start: .5f} sec')
            
            audio_to_text.extract_audio_from_video(url, audio_filename)
            
            sentences = audio_to_text.transcribe_audio(audio_filename)

            translated_sentences = translate_sentence.translate_en_to_ko(sentences)

            assesed_levels = analyze_sentence.assess_level(sentences)

            expression_lists = analyze_sentence.extract_key_expressions(sentences)

            video_id = url.split("shorts/")[1].split("?")[0]

            # session_state에 정보를 저장해서 상태 유지
            st.session_state[DATA_KEY] = {
                "url": "https://www.youtube.com/watch?v=" + video_id,
                "sentences": sentences,
                "translated_sentences": translated_sentences,
                "assesed_levels": assesed_levels,
                "expression_lists": expression_lists
            }

            status.update(label="", state="complete")
    else:
        st.toast('Please enter a YouTube Shorts video only!')

# session_state에 데이터가 있으면 UI 출력
if DATA_KEY in st.session_state:
    data = st.session_state[DATA_KEY]
    url = data["url"]
    sentences = data["sentences"]
    translated_sentences = data["translated_sentences"]
    assesed_levels = data["assesed_levels"]
    expression_lists = data["expression_lists"]

    # 유튜브 영상 출력
    st.video(url)

    for i, text in enumerate(sentences):
        # 추출한 텍스트 출력 및 핵심표현 세션에 저장
        if st.button(text):
            st.session_state[SELECTED_TEXT_KEY] = expression_lists[i]["MWEs"]

        # 문장 레벨에 따른 색상 표시
        level = assesed_levels[i]
        if "Elementary" in level:
            st.badge(level, color="green")
        elif "Intermediate" in level:
            st.badge(level, color="blue")
        else:
            st.badge(level, color="orange")

        # 번역한 문장 출력
        st.write(translated_sentences[i])

# 텍스트 클릭 시 핵심 표현 출력
selected = st.session_state.get(SELECTED_TEXT_KEY, None)
if selected:
    st.markdown("---")
    st.write("핵심 표현 :")
    for contents in selected:
        st.info(f"✔️ {contents["expression"]} : {contents["meaning"]}" )



        

