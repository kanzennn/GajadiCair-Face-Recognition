THRESH = 0.03
def isFist(
    ujung_jempol, ujung_telunjuk, ujung_tengah, ujung_manis, ujung_kelingking,
    pangkal_jempol, pangkal_telunjuk, pangkal_tengah, pangkal_manis, pangkal_kelingking
):
    all_bent = (
        ujung_jempol.y > pangkal_jempol.y and
        ujung_telunjuk.y > pangkal_telunjuk.y and
        ujung_tengah.y > pangkal_tengah.y and
        ujung_manis.y > pangkal_manis.y and
        ujung_kelingking.y > pangkal_kelingking.y
    )
    if all_bent:
        return "Fist"
    return None