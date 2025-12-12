import os
import sys
from typing import List

try:
    from rag_retriever import retrieve_knowledge
    print("✅ RAG 검색 모듈(rag_retriever) 로드 성공.")
except ImportError:
    print("❌ 오류: 'rag_retriever.py' 파일을 찾을 수 없습니다.")
    sys.exit()


# ============================================
# 🚨 Groq LLaMA 모델 불러오기
# ============================================
try:
    from groq import Groq

    if not os.getenv("GROQ_API_KEY"):
        print("\n⚠️ 경고: GROQ_API_KEY 환경 변수가 없습니다.")
        print("PowerShell에서: setx GROQ_API_KEY \"API키\"")
        LLM_CLIENT = None
    else:
        LLM_CLIENT = Groq(api_key=os.getenv("GROQ_API_KEY"))
        LLM_MODEL = "llama-3.3-70b-versatile"
        print(f"✅ LLM 로드 성공: {LLM_MODEL}")

except ImportError:
    print("❌ 'groq' 라이브러리가 설치되지 않음. pip install groq 필요.")
    LLM_CLIENT = None


# ============================================
# 📌 프롬프트 구성 함수
# ============================================
def build_prompt(query: str, contexts: List[str]) -> str:

    if not contexts:
        return (
            f"사용자 질문: {query}\n\n"
            "참고할 지식이 없습니다. 일반적인 지식으로 답변해주세요."
        )

    context_text = "\n---\n".join(contexts)

    system_prompt = (
        "당신은 수의학 전문가 AI입니다. "
        "제공된 참고 지식만 사용해 답변하세요. "
        "지식에 없는 내용은 추론하지 말고 '모르겠다'고 말하세요."
    )

    full_prompt = (
        f"{system_prompt}\n\n"
        f"--- 참고 지식 ---\n"
        f"{context_text}\n"
        f"------------------\n\n"
        f"사용자 질문: {query}"
    )
    return full_prompt


# ============================================
# 📌 답변 생성 함수
# ============================================
def generate_answer(query: str, filters: dict = None) -> str:

    if LLM_CLIENT is None:
        return "❌ LLM(API)가 로드되지 않았습니다. API 키 환경 변수를 확인하세요."

    print(f"\n[🔍] 질문: {query}")

    # 1) Retrieval
    try:
        retrieved_contexts = retrieve_knowledge(query, filters=filters, top_k=5)
    except Exception as e:
        return f"❌ 검색 오류: {e}"

    # 2) Prompt 구성
    final_prompt = build_prompt(query, retrieved_contexts)
    print(f"[💬] 검색된 컨텍스트 {len(retrieved_contexts)}개 기반으로 답변 생성 중...")

    # 3) Groq LLaMA 호출
    try:
        response = LLM_CLIENT.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "너는 수의학 전문 상담 AI야."},
                {"role": "user", "content": final_prompt},
            ],
            max_tokens=512,
            temperature=0.2,
        )

        return response.choices[0].message.content
    
    except Exception as e:
        return f"❌ LLM API 오류: {e}"


# ============================================
# 🚀 테스트 실행
# ============================================
if __name__ == "__main__":

    q1 = "우리 강아지가 밤에 기침을 많이 해. 왜 그럴까?"
    print("\n--- 답변 1 ---")
    print(generate_answer(q1))

    q2 = "노령견 치주 질환 관리법 알려줘."
    print("\n--- 답변 2 (치과 필터) ---")
    print(generate_answer(q2, filters={"department_meta": "치과"}))

    q3 = "새끼 강아지 설사할 때 집에서 뭘 해줄 수 있어?"
    print("\n--- 답변 3 (새끼 필터) ---")
    print(generate_answer(q3, filters={"lifeCycle": "새끼"}))
