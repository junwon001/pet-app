from bcs_repository import get_latest_bcs
from recommend.recommend_logic import (
    bcs_to_feed_type,
    feed_type_to_query
)
from recommend.feed_catalog import search_feed_from_naver


def recommend_feed_by_bcs(pet_id: int):
    bcs = get_latest_bcs(pet_id)
    if bcs is None:
        return {"error": "BCS 기록 없음"}

    feed_type = bcs_to_feed_type(bcs)
    query = feed_type_to_query(feed_type)

    feeds = search_feed_from_naver(
        query=query,
        display=5
    )

    return {
        "pet_id": pet_id,
        "bcs": bcs,
        "feed_type": feed_type,
        "search_query": query,
        "recommended_feeds": feeds
    }
