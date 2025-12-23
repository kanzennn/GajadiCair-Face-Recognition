THRESH = 0.08 
def isHi(
    ujung_jempol, ujung_telunjuk, ujung_tengah, ujung_manis, ujung_kelingking,
    pangkal_jempol, pangkal_telunjuk, pangkal_tengah, pangkal_manis, pangkal_kelingking
):
    all_extended = (
        ujung_jempol.y < pangkal_jempol.y - THRESH and
        ujung_telunjuk.y < pangkal_telunjuk.y - THRESH and
        ujung_tengah.y < pangkal_tengah.y - THRESH and
        ujung_manis.y < pangkal_manis.y - THRESH and
        ujung_kelingking.y < pangkal_kelingking.y - THRESH
    )
    
    if all_extended:
        return "Hi"
    
    return None