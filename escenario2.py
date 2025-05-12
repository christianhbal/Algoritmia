import numpy as np
import matplotlib.pyplot as plt
from escenario1 import cargar_y_segmentar
from pyedflib import highlevel



def graficar_todos_los_descriptores(resultados, etiquetas, descriptores):
    n_canales = len(etiquetas)
    n_descriptores = len(descriptores)

    for canal in range(n_canales):
        fig, axes = plt.subplots(n_descriptores, 1, figsize=(10, 2.5 * n_descriptores), sharex=True)
        fig.suptitle(f'Canal: {etiquetas[canal]}', fontsize=14)

        for idx, descriptor in enumerate(descriptores):
            tiempos = [r['inicio'] for r in resultados]
            valores = [r[descriptor][canal] for r in resultados]
            ax = axes[idx] if n_descriptores > 1 else axes
            ax.plot(tiempos, valores, label=descriptor)
            ax.set_ylabel(descriptor)
            ax.grid(True)
            ax.legend()

        axes[-1].set_xlabel("Tiempo (s)")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()




def calcular_descriptores_completos(segmento):              #Calculo estadísticos del segmento antes, crisis, despues
    return {
        'rms': np.sqrt(np.mean(segmento ** 2, axis=1)),
        'varianza': np.var(segmento, axis=1),
        'std': np.std(segmento, axis=1),
        'media_abs': np.mean(np.abs(segmento), axis=1),
        'autocorrelacion': np.array([
            np.correlate(c, c, mode='full')[len(c)//2] for c in segmento
        ]),
        'autocovarianza': np.array([
            np.mean((c - np.mean(c)) * (c - np.mean(c))) for c in segmento
        ]),
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
    inicio = 1086
    fin = 1196
    senales, headers, _ = highlevel.read_edf(nombre_edf)

    before, crisis, after, etiquetas = cargar_y_segmentar(nombre_edf, inicio, fin)
    frecuencia = headers[0]["sample_frequency"]

    senal_total = np.concatenate([before,crisis,after],axis=1);                        #concateno y armo la matriz de la muestra de la señal
    senial_procesada = analizar_por_frecuencia(senal_total,etiquetas, frecuencia)
    
    descriptores_a_graficar = ['rms', 'varianza', 'std', 'media_abs', 'autocorrelacion', 'autocovarianza']
    graficar_todos_los_descriptores(senial_procesada, etiquetas, descriptores_a_graficar)


main()