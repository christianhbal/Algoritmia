import numpy as np
import matplotlib.pyplot as plt
from escenario1 import cargar_y_segmentar
from pyedflib import highlevel



def graficar_descriptor(resultados, canal, descriptor, etiquetas):
    tiempos = [r['inicio'] for r in resultados]
    valores = [r[descriptor][canal] for r in resultados]

    plt.figure(figsize=(10, 4))
    plt.plot(tiempos, valores, label=f'{descriptor} - {etiquetas[canal]}')
    plt.xlabel("Tiempo (s)")
    plt.ylabel(descriptor)
    plt.grid(True)
    plt.title(f'Evolución del descriptor {descriptor} - Canal {etiquetas[canal]}')
    plt.legend()
    plt.tight_layout()
    plt.show()



def calcular_descriptores_completos(segmento):              #Calculo estadísticos del segmento antes, crisis, despues
    return {
        'varianza': np.var(segmento, axis=1),
        'std': np.std(segmento, axis=1),
        'media_abs': np.mean(np.abs(segmento), axis=1),
        'autocorrelacion': np.array([np.correlate(c, c, mode='full')[len(c)-1] for c in segmento]),
        'correlacion_de_Pearson': np.corrcoef(segmento)
    }

def analizar_por_frecuencia(senales, etiquetas, frecuencia, ventana_seg=2, paso_seg=0.5):          #Analizo la señal por pasos de frecuencia                                                                                             
                                                                                                    # analizo ventanas de 2 seg
    ventana_muestras = int(ventana_seg * frecuencia)                                                # la ventana avanza 0.5s
    paso_muestras = int(paso_seg * frecuencia)
    resultados = []

    for i in range(0, senales.shape[1] - ventana_muestras + 1, paso_muestras):
        segmento = senales[:, i:i+ventana_muestras]
        descriptores = calcular_descriptores_completos(segmento)
        resultados.append({
            'inicio': i / frecuencia,
            'fin': (i + ventana_muestras) / frecuencia,
            **descriptores
        })

    return resultados


def main():

    nombre_edf = "chb05_13.edf"
    inicio = 1732
    fin = 1772
    senales, headers, _ = highlevel.read_edf(nombre_edf)
    canal = 0
    descriptor = 'std'

    before, crisis, after, etiquetas = cargar_y_segmentar(nombre_edf, inicio, fin)
    frecuencia = headers[0]["sample_frequency"]

    senal_total = np.concatenate([before,crisis,after],axis=1);                        #concateno y armo la matriz de la muestra de la señal
    senial_procesada = analizar_por_frecuencia(senal_total,etiquetas, frecuencia)
    graficar_descriptor(senial_procesada, canal, descriptor, etiquetas)


main()
