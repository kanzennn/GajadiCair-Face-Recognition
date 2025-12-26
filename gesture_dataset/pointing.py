EXT_THRESH = 0.02
THUMB_THRESH = 0.04

def is_extended(tip, mcp):
    return tip.y < mcp.y - EXT_THRESH

def is_bent(tip, mcp):
    return tip.y > mcp.y + EXT_THRESH

def thumb_extended(tip, mcp):
    return abs(tip.x - mcp.x) > THUMB_THRESH

def isPointing(
    ujung_jempol, ujung_telunjuk, ujung_tengah, ujung_manis, ujung_kelingking,
    pangkal_jempol, pangkal_telunjuk, pangkal_tengah, pangkal_manis, pangkal_kelingking
):
    index_ok = is_extended(ujung_telunjuk, pangkal_telunjuk)

    bent_count = sum([
        is_bent(ujung_tengah, pangkal_tengah),
        is_bent(ujung_manis, pangkal_manis),
        is_bent(ujung_kelingking, pangkal_kelingking)
    ]) >= 2

    thumb_ok = not thumb_extended(ujung_jempol, pangkal_jempol)

    if index_ok and bent_count and thumb_ok:
        return "Pointing"

    return None
