# 유니코드 블록을 분류하는 스크립트

WHOLE_RANGE = [0, int("40000", base=16)]

HANGUL_UNICODE_BLOCKS = (
    "Hangul Jamo",
    "Hangul Compatibility Jamo",
    "Hangul Jamo Extended-A",
    "Hangul Syllables",
    "Hangul Jamo Extended-B",
)

ADDITIONAL_EXCLUDE_RANGES = [
    [0x0000, 0x001F],  # C0 Controls
    [0x007F, 0x009F],  # C1 Controls
    [0xD800, 0xDFFF],  # High and Low Surrogates
    [0xE000, 0xF8FF],  # Private Use Area
    [0xFFF0, 0xFFFF],  # Specials
]


def get_unicode_blocks(block_txt: str) -> dict[str, list]:
    """
    유니코드 블록 정보 문서로부터 {블록 이름: [시작, 끝]} 형식의 딕셔너리를 반환합니다.

    :param block_txt: 각 유니코드 블록의 시작과 끝 정보가 적혀있는 http://unicode.org/Public/UNIDATA/Blocks.txt 의 파일 경로
    :type block_txt: str
    :returns: 유니코드 블록 이름과 해당 블록의 [시작 유니코드, 끝 유니코드] 리스트를 매핑하는 딕셔너리
    :rtype: dict[str, list]
    """
    unicode_blocks = dict()
    with open(block_txt, "r") as file:
        while cur_line := file.readline():
            if cur_line[0] == "#" or cur_line.strip() == "":
                continue
            area, name = cur_line.strip().split(";")
            start, end = map(lambda x: int(x, base=16), area.split(".."))
            name = name.strip()
            unicode_blocks[name] = [start, end]
    return unicode_blocks
