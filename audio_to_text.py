import yt_dlp
from faster_whisper import WhisperModel
import time, re

def extract_audio_from_video(url, audio_filename):
    # 옵션 설정
    ydl_opts = {
        'format': 'm4a/bestaudio/best',      # 최고 음질
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',     # ffmpeg로 오디오 추출
            'preferredcodec': 'm4a',         # m4a로 변환
        }],
        'outtmpl' : audio_filename
    }

    # 음원 추출
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download(url)
    
def transcribe_audio(audio_filename):
     # 모델 설정
    model = WhisperModel('base', device='cuda', compute_type='float16')

    # 음성 파일 선택
    segments, info = model.transcribe(audio_filename, beam_size=5)

    # 언어 감지 출력
    print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

    # 결과 출력
    full_text = ""
    for segment in segments:
        full_text += segment.text
        
    # 문장 단위로 리스트화
    pattern = r'([^.!?]*[.!?])'
    sentences = [match.strip() for match in re.findall(pattern, full_text) if match.strip()]

    return sentences
    

if __name__ == '__main__':
    # start = time.time()
    # 유튜브 동영상에서 음원 추출
    url = ['https://youtube.com/shorts/UmDQcoDoCRg?si=OpT4F2A4gnBggFzK']
    audio_filename = 'audio.m4a'
    extract_audio_from_video(url, audio_filename)
    # end = time.time()
    # print(f'{end - start: .5f} sec')

    start = time.time()
    # 추출한 음원을 텍스트로 변환
    extrated_text = transcribe_audio(audio_filename)
    
    end = time.time()
    print(f'{end - start: .5f} sec')
    print(extrated_text)