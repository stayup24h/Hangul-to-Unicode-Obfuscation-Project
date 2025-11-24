import flask
from flask import Flask, request, jsonify, render_template_string
import threading
import webbrowser
import logging
from hangul_to_unicode_obfuscator import H2UObfuscator

app = Flask(__name__)
obfuscator = None

# Flask 기본 로깅을 비활성화
log = logging.getLogger("werkzeug")
log.setLevel(logging.ERROR)


def initialize_obfuscator():
    """백그라운드에서 H2UObfuscator 모델을 로드합니다."""
    global obfuscator
    print("Initializing H2UObfuscator... (might take a moment)")
    obfuscator = H2UObfuscator()
    print("H2UObfuscator initialized. The web service is ready.")


# # 유니코드 코드 포인트 범위에 따른 Noto 글꼴 매핑
# # 모든 유니코드를 포함하진 않지만, 주요 스크립트를 포함합니다.
# FONT_MAP = [
#     ((0x0370, 0x03FF), "Noto Sans Greek"),  # 그리스어
#     ((0x0400, 0x04FF), "Noto Sans Cyrillic"),  # 키릴어
#     ((0x0590, 0x05FF), "Noto Sans Hebrew"),  # 히브리어
#     ((0x0600, 0x06FF), "Noto Sans Arabic"),  # 아랍어
#     ((0x0900, 0x097F), "Noto Sans Devanagari"),  # 데바나가리
#     ((0x3040, 0x309F), "Noto Sans JP"),  # 히라가나
#     ((0x30A0, 0x30FF), "Noto Sans JP"),  # 가타카나
#     ((0x4E00, 0x9FFF), "Noto Sans SC"),  # CJK 통합 한자 (중국어 간체)
#     ((0xAC00, 0xD7A3), "Noto Sans KR"),  # 한글 음절 (원본)
#     ((0x2200, 0x22FF), "Noto Sans Math"),  # 수학 연산자
#     ((0x2600, 0x26FF), "Noto Sans Symbols"),  # 여러 기호
#     ((0x1F300, 0x1F5FF), "Noto Color Emoji"),  # 그림 이모티콘
#     ((0x1F600, 0x1F64F), "Noto Color Emoji"),  # 이모티콘
# ]


# def get_font_for_char(char):
#     """문자의 유니코드 코드 포인트를 기반으로 적절한 Noto 글꼴을 결정합니다."""
#     code = ord(char)
#     for (start, end), font in FONT_MAP:
#         if start <= code <= end:
#             return font
#     return "Noto Sans"  # 기본 글꼴


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <title>한글 난독화기</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Noto+Sans&family=Noto+Sans+KR&display=swap');
        body {
            font-family: 'Noto Sans KR', 'Noto Sans', sans-serif;
            max-width: 800px; margin: 40px auto; padding: 20px;
            background-color: #f8f9fa; color: #343a40; text-align: center;
        }
        h1 { color: #0056b3; }
        #input-text {
            width: 80%; padding: 12px; font-size: 1.2em; border-radius: 8px;
            border: 1px solid #ced4da; margin-bottom: 15px;
        }
        #convert-btn {
            padding: 12px 24px; font-size: 1.1em; cursor: pointer;
            background-color: #007bff; color: white; border: none; border-radius: 8px;
        }
        .result-container {
            margin-top: 25px; padding: 20px; background-color: #fff;
            border: 1px solid #dee2e6; border-radius: 8px;
            min-height: 60px; font-size: 2.5em; word-wrap: break-word;
        }
        #status { margin-top: 15px; color: #6c757d; }
    </style>
</head>
<body>
    <h1>한글 유니코드 난독화기</h1>
    <p>변환할 한글 문자열을 입력하면, 각 음절과 가장 유사한 형태의 다른 유니코드 문자로 변환합니다.</p>
    
    <form id="obfuscate-form" onsubmit="return false;">
        <input type="text" id="input-text" placeholder="예: 안녕하세요">
        <button type="submit" id="convert-btn">변환하기</button>
    </form>
    
    <div id="result-container-1" class="result-container"></div>
    <div id="result-container-2" class="result-container"></div>
    <div id="status">
        {% if not ready %}
        모델 초기화 중입니다...
        {% else %}
        준비 완료.
        {% endif %}
    </div>

    <script>
        const form = document.getElementById('obfuscate-form');
        const inputText = document.getElementById('input-text');
        const resultContainer1 = document.getElementById('result-container-1');
        const resultContainer2 = document.getElementById('result-container-2');
        const statusDiv = document.getElementById('status');
        const convertBtn = document.getElementById('convert-btn');

        if (statusDiv.textContent.includes('초기화 중')) {
            const statusInterval = setInterval(async () => {
                try {
                    const response = await fetch('/status');
                    const data = await response.json();
                    if (data.ready) {
                        statusDiv.textContent = '준비 완료.';
                        clearInterval(statusInterval);
                    }
                } catch (error) {
                    console.error('상태 확인 오류:', error);
                    clearInterval(statusInterval);
                }
            }, 2000);
        }

        const dynamicFontsStyle = document.createElement('style');
        document.head.appendChild(dynamicFontsStyle);
        const loadedFonts = new Set(['Noto Sans', 'Noto Sans KR']);

        async function handleConversion() {
            const text = inputText.value.trim();
            if (!text) return;

            statusDiv.textContent = '변환 중입니다. 잠시만 기다려주세요...';
            resultContainer1.innerHTML = '';
            resultContainer2.innerHTML = '';
            convertBtn.disabled = true;

            try {
                const response = await fetch('/convert', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text: text })
                });

                if (!response.ok) {
                    const error = await response.json();
                    throw new Error(error.error || '서버 응답 오류');
                }

                const data = await response.json();
                candidate1 = data[0]
                candidate2 = data[1]
                
                candidate1.forEach(item => {
                    const charSpan = document.createElement('span');
                    charSpan.textContent = item;
                    resultContainer1.appendChild(charSpan);
                });

                candidate2.forEach(item => {
                    const charSpan = document.createElement('span');
                    charSpan.textContent = item;
                    resultContainer2.appendChild(charSpan);
                });

                statusDiv.textContent = '변환 완료.';
            } catch (error) {
                console.error('Error:', error);
                statusDiv.textContent = `오류: ${error.message}`;
            } finally {
                convertBtn.disabled = false;
            }
        }
        
        form.addEventListener('submit', handleConversion);
    </script>
</body>
</html>
"""

# Flask 라우트 정의
@app.route("/")
def index():
    """메인 페이지를 렌더링합니다."""
    return render_template_string(HTML_TEMPLATE, ready=(obfuscator is not None))

@app.route("/status")
def status():
    """모델 초기화 상태를 반환합니다."""
    return jsonify({"ready": obfuscator is not None})


@app.route("/convert", methods=["POST"])
def convert():
    """문자열 변환 요청을 처리합니다."""
    if obfuscator is None:
        return jsonify({"error": "난독화 모델이 아직 초기화되지 않았습니다."}), 503

    input_string = request.json.get("text", "")
    if not input_string:
        return jsonify([])

    results1 = []
    results2 = []
    for char in input_string:
        if char.isspace():
            results1.append(" ")
            results2.append(" ")
            continue

        # 한글 또는 변환 가능한 문자인지 확인
        if "가" <= char <= "힣":
            similar_chars = obfuscator.find_similar_unicode(char, k=2)
            if similar_chars and similar_chars[0]["Character"]:
                obfuscated_char1 = similar_chars[0]["Character"]
                obfuscated_char2 = similar_chars[1]["Character"]
                results1.append(obfuscated_char1)
                results2.append(obfuscated_char2)
        else:  # 한글이 아니면 원본 유지
            results1.append(char)
            results2.append(char)

    return jsonify([results1, results2])


def run_app():
    """Flask 앱을 실행합니다."""
    app.run(port=5000)


if __name__ == "__main__":
    # 백그라운드 스레드에서 모델 초기화 시작
    init_thread = threading.Thread(target=initialize_obfuscator, daemon=True)
    init_thread.start()

    # 웹 브라우저를 열어 앱 페이지 표시
    url = "http://127.0.0.1:5000"
    print(f"Starting web server at {url}")
    print("The server is running, but the model is still loading in the background.")
    webbrowser.open_new(url)

    run_app()
