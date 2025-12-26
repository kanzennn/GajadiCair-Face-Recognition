THRESH = 0.04   
def isPeace(
    ujung_jempol, ujung_telunjuk, ujung_tengah, ujung_manis, ujung_kelingking,
    pangkal_jempol, pangkal_telunjuk, pangkal_tengah, pangkal_manis, pangkal_kelingking
):
    index_extended = (pangkal_telunjuk.y - ujung_telunjuk.y) > THRESH
    middle_extended = (pangkal_tengah.y - ujung_tengah.y) > THRESH

    ring_bent = (ujung_manis.y - pangkal_manis.y) > -THRESH
    pinky_bent = (ujung_kelingking.y - pangkal_kelingking.y) > -THRESH

    index_middle_gap = abs(ujung_telunjuk.x - ujung_tengah.x) > THRESH

    thumb_not_extended = abs(ujung_jempol.x - pangkal_jempol.x) < THRESH * 1.5

    if (
        index_extended and
        middle_extended and
        ring_bent and
        pinky_bent and
        index_middle_gap and
        thumb_not_extended
    ):
        return "Peace"

    return None
