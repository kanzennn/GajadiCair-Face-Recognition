THRESH = 0.03
def isPointing(
    ujung_jempol, ujung_telunjuk, ujung_tengah, ujung_manis, ujung_kelingking,
    pangkal_jempol, pangkal_telunjuk, pangkal_tengah, pangkal_manis, pangkal_kelingking
):
    index_extended = ujung_telunjuk.y < pangkal_telunjuk.y - THRESH
    other_fingers_bent = (
        ujung_tengah.y > pangkal_tengah.y and
        ujung_manis.y > pangkal_manis.y and
        ujung_kelingking.y > pangkal_kelingking.y
    )
    thumb_bent = ujung_jempol.y > pangkal_jempol.y

    if index_extended and other_fingers_bent and thumb_bent:
        return "Pointing"
    return None