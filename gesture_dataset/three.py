THRESH = 0.03
def isThree(
    ujung_jempol, ujung_telunjuk, ujung_tengah, ujung_manis, ujung_kelingking,
    pangkal_jempol, pangkal_telunjuk, pangkal_tengah, pangkal_manis, pangkal_kelingking
):
    three_extended = (
        ujung_telunjuk.y < pangkal_telunjuk.y and
        ujung_tengah.y < pangkal_tengah.y and
        ujung_manis.y < pangkal_manis.y
    )
    pinky_bent = ujung_kelingking.y > pangkal_kelingking.y
    thumb_bent = ujung_jempol.y > pangkal_jempol.y

    if three_extended and pinky_bent and thumb_bent:
        return "Three"
    return None