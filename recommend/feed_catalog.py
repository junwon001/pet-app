import os
import requests

NAVER_CLIENT_ID = os.getenv("NAVER_CLIENT_ID")
NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")

NAVER_SHOPPING_URL = "https://openapi.naver.com/v1/search/shop.json"


def search_feed_from_naver(
    query: str,
    display: int = 5,
    min_price: int | None = None,
    max_price: int | None = None
):
    headers = {
        "X-Naver-Client-Id": NAVER_CLIENT_ID,
        "X-Naver-Client-Secret": NAVER_CLIENT_SECRET
    }

    params = {
        "query": query,
        "display": display,
        "sort": "sim"   # 정확도순
    }

    response = requests.get(
        NAVER_SHOPPING_URL,
        headers=headers,
        params=params,
        timeout=5
    )

    response.raise_for_status()
    items = response.json().get("items", [])

    # 가격 필터링
    results = []
    for item in items:
        price = int(item["lprice"])
        if min_price and price < min_price:
            continue
        if max_price and price > max_price:
            continue

        results.append({
            "title": item["title"],
            "price": price,
            "link": item["link"],
            "mall": item["mallName"],
            "image": item["image"]
        })

    return results
