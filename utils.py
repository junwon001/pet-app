import re

def extract_bcs_number(text: str) -> int:
    match = re.search(r"\b[1-9]\b", text)
    if not match:
        raise ValueError("BCS 숫자 추출 실패")
    return int(match.group())