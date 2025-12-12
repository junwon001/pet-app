def bcs_to_feed_type(bcs: int) -> str:
    if bcs <= 3:
        return "weight_gain"
    elif bcs <= 5:
        return "maintenance"
    elif bcs <= 7:
        return "weight_control"
    else:
        return "diet"


def feed_type_to_query(feed_type: str) -> str:
    mapping = {
        "weight_gain": "강아지 고단백 사료",
        "maintenance": "강아지 성견 사료",
        "weight_control": "강아지 체중관리 사료",
        "diet": "강아지 다이어트 사료"
    }
    return mapping[feed_type]
