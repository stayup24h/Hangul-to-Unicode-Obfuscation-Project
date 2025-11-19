import unicodedata
import numpy as np
import itertools
from collections import defaultdict


class UniversalScriptCombiner:
    def __init__(self):
        self.script_data = defaultdict(lambda: {"bases": [], "combines": []})
        self._scan_unicode()

    def _scan_unicode(self, end=0x40000):
        """유니코드 전체를 스캔하여 스크립트별로 분류합니다."""
        print("Scanning Unicode...")

        for code_point in range(end):
            char = chr(code_point)
            category = unicodedata.category(char)

            # 제어 문자 등은 건너뛰기
            if category.startswith("C") or category.startswith("Z"):
                continue

            try:
                script_name = unicodedata.name(char).split()[0]
            except ValueError:  # 이름이 없음
                continue

            # 1. Base 문자
            if category.startswith("L"):
                self.script_data[script_name]["bases"].append(char)

            # 2. Combining 문자
            elif category.startswith("M"):
                self.script_data[script_name]["combines"].append(char)

        print(f"Found {len(self.script_data)} scripts.")

    def get_binary_combinations(self, target_script):
        """특정 스크립트의 모든 Base와 Combining의 이진 조합을 리스트로 반환"""
        target_script = target_script.upper()

        if target_script not in self.script_data:
            print(f"Error: can't find '{target_script}' script.")
            return None

        bases = self.script_data[target_script]["bases"]
        combines = self.script_data[target_script]["combines"]

        if not bases or not combines:
            print(
                f"'{target_script}': 조합 가능한 Base 혹은 Combining 문자가 없습니다."
            )
            return None

        print(f"Processing {target_script}...")
        print(f"# of Bases: {len(bases)}, # of Combinings: {len(combines)}")

        # 모든 경우의 수
        combinations = list(itertools.product(bases, combines))

        result_list = []
        for b, c in combinations:
            result_list.append(unicodedata.normalize("NFC", b + c))

        return result_list

    def list_available_scripts(self):
        """조합 문자가 있는 스크립트 목록"""
        valid_scripts = []
        for script, data in self.script_data.items():
            if data["bases"] and data["combines"]:
                valid_scripts.append(script)
        return sorted(valid_scripts)


if __name__ == "__main__":
    # 실행
    combiner = UniversalScriptCombiner()

    # 1. 조합 가능한 스크립트 목록 확인
    available = combiner.list_available_scripts()
    print(f"\n조합 가능한 스크립트: {available}")

    all_combined_array = []
    for script in available:
        if combined_array := combiner.get_binary_combinations(script.upper()):
            all_combined_array += combined_array

    print(all_combined_array)
