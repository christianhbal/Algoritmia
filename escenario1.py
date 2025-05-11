import numpy as np
from pyedflib import highlevel

def cargar_y_segmentar(nombre_edf, inicio_seg, fin_seg, seg_antes=120, seg_despues=120):
    senales, headers, _ = highlevel.read_edf(nombre_edf)
    frecuencia = headers[0]["sample_frequency"]
    etiquetas = [h["label"] for h in headers]
    i_muestra_0 = max(0, int((inicio_seg - seg_antes) * frecuencia))
    i_muestra_1 = int(inicio_seg * frecuencia)
    i_muestra_2 = int(fin_seg * frecuencia)
    i_muestra_3 = min(senales.shape[1], int((fin_seg + seg_despues) * frecuencia))

    before = centrar_bloque(senales, i_muestra_0, i_muestra_1)
    crisis = centrar_bloque(senales, i_muestra_1, i_muestra_2)
    after  = centrar_bloque(senales, i_muestra_2, i_muestra_3)

    return before, crisis, after, etiquetas

def centrar_bloque(arr, i_muestra_inicial, i_muestra_final):
    bloque = arr[:, i_muestra_inicial:i_muestra_final]
    return bloque - bloque.mean(axis=1, keepdims=True)

def calcular_descriptores(segmento):
    return {
        'varianza': np.var(segmento, axis=1),
        'std': np.std(segmento, axis=1),
        'media': np.mean(np.abs(segmento), axis=1)
    }

def main():
    # Usamos la primera crisis como ejemplo
    nombre_edf = "chb05_13.edf"
    inicio = 1732
    fin = 1772

    before, crisis, after, etiquetas = cargar_y_segmentar(nombre_edf, inicio, fin)

    desc_before = calcular_descriptores(before)
    desc_durante = calcular_descriptores(crisis)
    desc_after = calcular_descriptores(after)

    print("Descriptores por canal:")
    canales = before.shape[0]
    for ch in range(canales):
        etiqueta = etiquetas[ch]
        print(f"\nCanal {ch} - [{etiqueta}]:")
        print(f"Before:   var = {desc_before['varianza'][ch]:.2f}, std = {desc_before['std'][ch]:.2f}, media = {desc_before['media'][ch]:.2f}")
        print(f"Crisis:   var = {desc_durante['varianza'][ch]:.2f}, std = {desc_durante['std'][ch]:.2f}, media = {desc_durante['media'][ch]:.2f}")
        print(f"After:    var = {desc_after['varianza'][ch]:.2f}, std = {desc_after['std'][ch]:.2f}, media = {desc_after['media'][ch]:.2f}")


if __name__ == "__main__":
    main()
