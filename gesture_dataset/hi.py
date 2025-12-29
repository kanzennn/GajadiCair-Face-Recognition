EXT_THRESH = 0.02
THUMB_THRESH = 0.04

def is_extended(tip, mcp):
    return tip.y < mcp.y - EXT_THRESH

def is_bent(tip, mcp):
    return tip.y > mcp.y + EXT_THRESH

def thumb_extended(tip, mcp):
    return abs(tip.x - mcp.x) > THUMB_THRESH

def isHi(
    ujung_jempol, ujung_telunjuk, ujung_tengah, ujung_manis, ujung_kelingking,
    pangkal_jempol, pangkal_telunjuk, pangkal_tengah, pangkal_manis, pangkal_kelingking
):
    fingers = [
        is_extended(ujung_telunjuk, pangkal_telunjuk),
        is_extended(ujung_tengah, pangkal_tengah),
        is_extended(ujung_manis, pangkal_manis),
        is_extended(ujung_kelingking, pangkal_kelingking),
        thumb_extended(ujung_jempol, pangkal_jempol)
    ]

    if sum(fingers) >= 4:
        return "Hi"

    return None
