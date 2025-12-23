THRESH = 0.03
def isOk(
    ujung_jempol, ujung_telunjuk, ujung_tengah, ujung_manis, ujung_kelingking,
    pangkal_jempol, pangkal_telunjuk, pangkal_tengah, pangkal_manis, pangkal_kelingking
):
    ok_sign = (
        abs(ujung_jempol.x - ujung_telunjuk.x) < 0.05 and
        abs(ujung_jempol.y - ujung_telunjuk.y) < 0.05 and
        ujung_tengah.y < pangkal_tengah.y and
        ujung_manis.y < pangkal_manis.y and
        ujung_kelingking.y < pangkal_kelingking.y
    )

    if ok_sign:
        return "OK"
    return None