THRESH = 0.08  

def isRock(
    ujung_jempol, ujung_telunjuk, ujung_tengah, ujung_manis, ujung_kelingking,
    pangkal_jempol, pangkal_telunjuk, pangkal_tengah, pangkal_manis, pangkal_kelingking
):
    if (ujung_telunjuk.y < pangkal_telunjuk.y and
        ujung_kelingking.y < pangkal_kelingking.y and
        ujung_tengah.y > pangkal_tengah.y and
        ujung_manis.y > pangkal_manis.y):
        return "Rock"
    
    else:
        return None