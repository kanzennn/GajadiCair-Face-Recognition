THRESH = 0.03
def isPeace(
    ujung_jempol, ujung_telunjuk, ujung_tengah, ujung_manis, ujung_kelingking,
    pangkal_jempol, pangkal_telunjuk, pangkal_tengah, pangkal_manis, pangkal_kelingking
):
    index_middle_extended = (
        ujung_telunjuk.y < pangkal_telunjuk.y and
        ujung_tengah.y < pangkal_tengah.y
    )

    other_fingers_bent = (
        ujung_manis.y > pangkal_manis.y and
        ujung_kelingking.y > pangkal_kelingking.y
    )

    thumb_bent = ujung_jempol.y > pangkal_jempol.y

    if index_middle_extended and other_fingers_bent and thumb_bent:
        return "Peace"

    return None
