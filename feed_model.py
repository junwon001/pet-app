# 사료 추천 시스템 프로토타입 스크립트
# - DB 스키마 (SQLite 예시)
# - 사료 정보 크롤러(예시: requests + BeautifulSoup)
# - 텍스트 전처리 + 임베딩(embedding placeholder)
# - RAG 인덱싱(FAISS 예시) + 검색
# - 간단한 추천 로직 (규칙 기반 + RAG 기반 혼합)
#
# 사용법 요약:
# 1) 데이터 수집: crawl_feed_site(url)로 제품 페이지 수집(사이트별 커스터마이즈 필요)
# 2) DB에 저장: save_feed_to_db(records)
# 3) 인덱싱: build_faiss_index(db_path)
# 4) 추천: recommend_feed(profile, topk=3)

# 필요한 패키지 예시:
# pip install requests beautifulsoup4 lxml sqlite3 sentence-transformers faiss-cpu transformers

import sqlite3
import requests
from bs4 import BeautifulSoup
import json
import os
from typing import List, Dict, Any

# --- 1) DB 스키마 ---
# SQLite로 간단히 구성: products 테이블과 raw_html(옵션)
DB_SCHEMA = '''
CREATE TABLE IF NOT EXISTS products (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    brand TEXT,
    product_name TEXT,
    species TEXT,            -- dog / cat
    life_stage TEXT,         -- puppy / adult / senior
    features TEXT,           -- e.g. hypoallergenic, weight-control
    ingredients TEXT,        -- 원재료 텍스트
    analysis TEXT,           -- 영양분 표시(예: 단백질 26%, 지방 12%)
    kcal_per_100g REAL,
    price REAL,
    product_url TEXT UNIQUE,
    scraped_at TEXT
);
'''

# DB helper
def init_db(db_path='feeds.db'):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.executescript(DB_SCHEMA)
    conn.commit()
    conn.close()

def save_feed_to_db(records: List[Dict[str,Any]], db_path='feeds.db'):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    for r in records:
        try:
            cur.execute('''
                INSERT OR IGNORE INTO products
                (brand, product_name, species, life_stage, features, ingredients, analysis, kcal_per_100g, price, product_url, scraped_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,datetime('now'))
            ''', (
                r.get('brand'), r.get('product_name'), r.get('species'), r.get('life_stage'), r.get('features'),
                r.get('ingredients'), r.get('analysis'), r.get('kcal_per_100g'), r.get('price'), r.get('product_url')
            ))
        except Exception as e:
            print('DB insert error', e, r.get('product_url'))
    conn.commit()
    conn.close()

# --- 2) 크롤러 예시 ---
# 사이트마다 구조가 다르므로, 페이지 구조에 맞춰 select/parse 해야 함.
# 아래는 generic 예시: list page -> detail page -> 파싱

def crawl_feed_site(list_page_url: str, domain: str) -> List[Dict[str,Any]]:
    """간단한 크롤러 예시: 리스트 페이지에서 제품 링크 수집 후 상세정보 파싱
    반드시 사이트의 robots.txt와 TOS 확인 후 사용하세요.
    """
    headers = {'User-Agent': 'Mozilla/5.0 (compatible)'}
    out = []
    resp = requests.get(list_page_url, headers=headers, timeout=10)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, 'lxml')

    # 이 부분은 사이트마다 달라짐. 예: 모든 제품 링크가 <a class="product-link">
    links = []
    for a in soup.select('a'):
        href = a.get('href')
        if href and 'product' in href:
            if href.startswith('/'):
                href = domain.rstrip('/') + href
            links.append(href)

    for link in set(links):
        try:
            d = parse_product_page(link)
            if d:
                out.append(d)
        except Exception as e:
            print('crawl error', link, e)
    return out


def parse_product_page(url: str) -> dict:
    """
    Coupang 전용 파서 (Selenium 기반 필수)
    - 제품명
    - 가격
    - 성분표(텍스트 기반 추출)
    - 특징/설명
    """
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.chrome.options import Options

    options = Options()
    options.add_argument("--headless=new")
    driver = webdriver.Chrome(options=options)
    driver.get(url)

    product = {
        "product_name": None,
        "price": None,
        "ingredients": None,
        "features": None,
        "url": url
    }

    try:
        # 제품명
        product["product_name"] = driver.find_element(By.CSS_SELECTOR, "h2.prod-buy-header__title").text
    except:
        pass

    try:
        # 가격
        product["price"] = driver.find_element(By.CSS_SELECTOR, "span.total-price > strong").text
    except:
        pass

    try:
        # 상세설명 탭 클릭
        driver.find_element(By.XPATH, "//li[contains(text(), '상품정보')]").click()
    except:
        pass

    try:
        # 성분/설명 추출
        desc = driver.find_element(By.CSS_SELECTOR, "div.prod-description-container").text
        product["ingredients"] = desc
        product["features"] = desc
    except:
        pass

    driver.quit()
    return product(url: str) -> Dict[str,Any]:
    """상세페이지 파서(사이트 구조에 맞게 수정)
    최소로 아래 필드를 반환:
    brand, product_name, species, life_stage, features, ingredients, analysis, kcal_per_100g, price, product_url
    """
    headers = {'User-Agent': 'Mozilla/5.0 (compatible)'}
    resp = requests.get(url, headers=headers, timeout=10)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, 'lxml')

    # 아래 선택자는 예시. 실제 사이트에 맞춰 수정 필요.
    brand = soup.select_one('.brand') and soup.select_one('.brand').get_text(strip=True)
    product_name = soup.select_one('h1') and soup.select_one('h1').get_text(strip=True)
    features = ' '.join([t.get_text(strip=True) for t in soup.select('.feature')]) if soup.select('.feature') else ''
    ingredients = soup.select_one('.ingredients') and soup.select_one('.ingredients').get_text(separator=', ', strip=True)
    analysis = soup.select_one('.analysis') and soup.select_one('.analysis').get_text(separator='; ', strip=True)
    price = None
    price_tag = soup.select_one('.price')
    if price_tag:
        try:
            price = float(''.join(ch for ch in price_tag.get_text() if ch.isdigit() or ch=='.'))
        except:
            price = None

    # species, life_stage, kcal_per_100g는 상세 파싱 로직 필요
    species = 'dog' if 'dog' in url or 'dog' in (features or '').lower() else 'cat'
    life_stage = 'adult'
    kcal_per_100g = None

    return {
        'brand': brand,
        'product_name': product_name,
        'species': species,
        'life_stage': life_stage,
        'features': features,
        'ingredients': ingredients,
        'analysis': analysis,
        'kcal_per_100g': kcal_per_100g,
        'price': price,
        'product_url': url
    }

# --- 3) 텍스트 전처리 + 임베딩 (placeholder) ---
# 실제로는 sentence-transformers나 OpenAI embedding API 사용

def text_for_embedding(product_record: Dict[str,Any]) -> str:
    parts = []
    parts.append(product_record.get('brand',''))
    parts.append(product_record.get('product_name',''))
    if product_record.get('features'):
        parts.append('FEATURES: ' + product_record['features'])
    if product_record.get('ingredients'):
        parts.append('INGREDIENTS: ' + product_record['ingredients'])
    if product_record.get('analysis'):
        parts.append('ANALYSIS: ' + product_record['analysis'])
    return '\n'.join([p for p in parts if p])

# Embedding 함수 예시(여기선 더미로 반환)
def embed_texts(texts: List[str]):
    """실전: sentence-transformers 또는 OpenAI embeddings 사용
    여기서는 임베딩을 더미로 만들어 반환
    """
    import numpy as np
    return np.random.rand(len(texts), 768).astype('float32')

# --- 4) FAISS 인덱스 생성 예시 ---
def build_faiss_index(db_path='feeds.db', index_path='feeds.faiss', mapping_path='id2meta.json'):
    try:
        from sentence_transformers import SentenceTransformer
        import faiss
        import numpy as np
    except Exception as e:
        print('Missing packages: pip install sentence-transformers faiss-cpu')
        raise

    model = SentenceTransformer('all-MiniLM-L6-v2')

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute('SELECT id, brand, product_name, features, ingredients, analysis FROM products')
    rows = cur.fetchall()
    texts = []
    ids = []
    id2meta = {}
    for r in rows:
        pid = r[0]
        rec = {
            'brand': r[1], 'product_name': r[2], 'features': r[3], 'ingredients': r[4], 'analysis': r[5]
        }
        txt = text_for_embedding(rec)
        texts.append(txt)
        ids.append(pid)
        id2meta[str(pid)] = rec
    conn.close()

    if not texts:
        print('No texts to index')
        return

    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, index_path)
    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump({'ids': ids, 'meta': id2meta}, f, ensure_ascii=False, indent=2)
    print('Index built', index.ntotal)

# --- 5) 검색 및 추천 ---

def search_similar_products(query: str, index_path='feeds.faiss', mapping_path='id2meta.json', topk=5):
    try:
        import faiss
        import numpy as np
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        print('Missing packages for search')
        raise

    model = SentenceTransformer('all-MiniLM-L6-v2')
    q_emb = model.encode([query], convert_to_numpy=True)

    index = faiss.read_index(index_path)
    D, I = index.search(q_emb, topk)

    with open(mapping_path, 'r', encoding='utf-8') as f:
        mapping = json.load(f)
    ids = mapping['ids']
    meta = mapping['meta']

    out = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0:
            continue
        pid = ids[idx]
        out.append({'id': pid, 'score': float(score), 'meta': meta.get(str(pid))})
    return out

# 규칙 기반 필터

def rule_filter(products: List[Dict[str,Any]], profile: Dict[str,Any]):
    """간단한 규칙 기반 필터 예:
    - 비만(고지방/고칼로리)인 경우 저칼로리/weight-control 키워드 필터
    - 알러지: 알러지 유발 성분 포함 여부 검사
    """
    out = []
    target_keywords = []
    if profile.get('goal') == 'weight_loss':
        target_keywords.append('weight')
        target_keywords.append('low fat')
        target_keywords.append('low calorie')
    if profile.get('allergy'):
        # 단순히 성분 텍스트에 알러지 성분 포함 여부 확인
        allergen = profile['allergy'].lower()
    else:
        allergen = None

    for p in products:
        meta = p.get('meta', {})
        text = ' '.join([str(meta.get('features','') or ''), str(meta.get('ingredients','') or ''), str(meta.get('analysis','') or '')]).lower()
        score = 0
        if any(k in text for k in target_keywords):
            score += 1
        if allergen and allergen in text:
            score -= 10
        out.append((score, p))
    out.sort(key=lambda x: x[0], reverse=True)
    return [p for s,p in out]

# 최종 추천 함수

def recommend_feed(profile: Dict[str,Any], topk=3):
    """
    profile 예시:
    {
      'species': 'dog',
      'age': 3,
      'weight_kg': 12.5,
      'goal': 'weight_loss' / 'maintenance',
      'allergy': 'chicken' or None,
      'notes': 'skin issues'
    }
    """
    # 1) 규칙 기반 빠른 후보 추출(간단)
    q = f"species: {profile.get('species')}, life_stage: {profile.get('life_stage','adult')}, goal: {profile.get('goal')}, notes: {profile.get('notes','')}, allergy: {profile.get('allergy','') }"
    candidates = search_similar_products(q, topk=20)
    filtered = rule_filter(candidates, profile)
    top = filtered[:topk]
    # 결과 포맷 정리
    res = []
    for r in top:
        entry = r.get('meta', {})
        res.append({
            'brand': entry.get('brand'),
            'product_name': entry.get('product_name'),
            'features': entry.get('features'),
            'ingredients': entry.get('ingredients')
        })
    return res

# --- 6) LLM용 RAG prompt 템플릿 (예시) ---
# RAG를 활용해 추천 근거를 제공하는 프롬프트 예시
RAG_PROMPT_TEMPLATE = '''
You are a helpful pet nutrition assistant.
User profile: {profile}

You have these candidate products (with metadata & analysis):
{candidates}

Task:
1) Rank the candidates for the user's goal and give short explanation (1-2 sentences each).
2) For the top recommendation, provide feeding guideline (amount per day) and why it's suitable.
3) Add a short note about possible concerns.

Answer in Korean.
'''

# 예시: LLM 호출은 OpenAI나 다른 모델로 대체
def call_llm_rag(profile: Dict[str,Any], candidates: List[Dict[str,Any]]):
    prompt = RAG_PROMPT_TEMPLATE.format(profile=json.dumps(profile, ensure_ascii=False), candidates=json.dumps(candidates, ensure_ascii=False))
    # 실제: OpenAI/chat completions 호출
    print('LLM prompt (trimmed):', prompt[:800])
    # return fake response
    return 'LLM response placeholder'

# --- 7) 예제 실행 흐름 ---
if __name__ == '__main__':
    init_db()
    print('DB initialized (feeds.db)')
    # 1) 크롤링 예시 (실제 사이트 URL로 바꿔서 사용)
    # records = crawl_feed_site('https://example.com/pet-foods', 'https://example.com')
    # save_feed_to_db(records)

    # 2) 인덱스 빌드 (한번만)
    # build_faiss_index()

    # 3) 추천 예시
    sample_profile = {
        'species': 'cat', 'age': 3, 'weight_kg': 5.0, 'goal': 'weight_loss', 'allergy': None, 'notes': 'slightly overweight'
    }
    # result = recommend_feed(sample_profile)
    # print(result)

    print('Prototype ready. Update parse_product_page() for target sites, then run full pipeline.')
