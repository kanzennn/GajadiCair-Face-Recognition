THRESH = 0.03
def isL(
    ujung_jempol, ujung_telunjuk, ujung_tengah, ujung_manis, ujung_kelingking,
    pangkal_jempol, pangkal_telunjuk, pangkal_tengah, pangkal_manis, pangkal_kelingking
):
    thumb_extended_sideways = abs(ujung_jempol.x - pangkal_telunjuk.x) > 0.08
    index_extended = ujung_telunjuk.y < pangkal_telunjuk.y
    other_fingers_bent = (
        ujung_tengah.y > pangkal_tengah.y and
        ujung_manis.y > pangkal_manis.y and
        ujung_kelingking.y > pangkal_kelingking.y
    )
    thumb_not_up = ujung_jempol.y > pangkal_telunjuk.y - 0.02

    if thumb_extended_sideways and index_extended and other_fingers_bent and thumb_not_up:
        return "L"
    return None
