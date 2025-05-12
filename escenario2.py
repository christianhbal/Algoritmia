import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, expon, lognorm
from pyedflib import highlevel
from escenario1 import cargar_y_segmentar

def ajustar_y_graficar_pdf_normal_exponencial(data, nombre_desc):
    x = np.linspace(min(data), max(data), 100)

    # --- Ajuste Normal ---
    mu, sigma = norm.fit(data)
    pdf_norm = norm.pdf(x, mu, sigma)

    # --- Ajuste Exponencial ---
    loc_exp, scale_exp = expon.fit(data)
    pdf_expon = expon.pdf(x, loc_exp, scale_exp)

    # --- Ajuste Lognormal (forzando loc=0) ---
    shape_log, loc_log, scale_log = lognorm.fit(data, floc=0)
    pdf_log = lognorm.pdf(x, shape_log, loc_log, scale_log)

    # --- Gráfico Normal ---
    plt.figure(figsize=(10, 4))
    plt.hist(data, bins=50, density=True, alpha=0.5, label="Datos")
    plt.plot(x, pdf_norm, 'r-', label=f'Normal PDF\nμ={mu:.3f}, σ={sigma:.3f}')
    plt.title(f"Ajuste Normal - {nombre_desc}")
    plt.xlabel(nombre_desc)
    plt.ylabel("Densidad")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Boxplot Normal
    plt.figure(figsize=(6, 2))
    plt.boxplot(data, vert=False)
    plt.title(f"Boxplot (Normal) - {nombre_desc}")
    plt.xlabel(nombre_desc)
    plt.grid(True)
    plt.show()

    # --- Gráfico Exponencial ---
    plt.figure(figsize=(10, 4))
    plt.hist(data, bins=50, density=True, alpha=0.5, label="Datos")
    plt.plot(x, pdf_expon, 'g-', label=f'Exponencial PDF\nloc={loc_exp:.3f}, scale={scale_exp:.3f}')
    plt.title(f"Ajuste Exponencial - {nombre_desc}")
    plt.xlabel(nombre_desc)
    plt.ylabel("Densidad")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Boxplot Exponencial
    plt.figure(figsize=(6, 2))
    plt.boxplot(data, vert=False)
    plt.title(f"Boxplot (Exponencial) - {nombre_desc}")
    plt.xlabel(nombre_desc)
    plt.grid(True)
    plt.show()

    # --- Gráfico Lognormal ---
    plt.figure(figsize=(10, 4))
    plt.hist(data, bins=50, density=True, alpha=0.5, label="Datos")
    plt.plot(x, pdf_log, 'b-', label=f'Lognormal PDF\nshape={shape_log:.3f}, scale={scale_log:.3f}')
    plt.title(f"Ajuste Lognormal - {nombre_desc}")
    plt.xlabel(nombre_desc)
    plt.ylabel("Densidad")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Boxplot Lognormal
    plt.figure(figsize=(6, 2))
    plt.boxplot(data, vert=False)
    plt.title(f"Boxplot (Lognormal) - {nombre_desc}")
    plt.xlabel(nombre_desc)
    plt.grid(True)
    plt.show()

    return (mu, sigma), (loc_exp, scale_exp), (shape_log, loc_log, scale_log)




def detectar_crisis_por_umbral_todos_canales(resultados, descriptor, umbral):
    n_canales = len(resultados[0][descriptor])
    tiempos_detectados = [None] * n_canales

    for r in resultados:
        valores = r[descriptor]
        for ch in range(n_canales):
            if tiempos_detectados[ch] is None and valores[ch] > umbral:
                tiempos_detectados[ch] = r['inicio']

    return tiempos_detectados



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

    desc_crisis = calcular_descriptores_completos(crisis)
    umbral_varianza = np.min(desc_crisis['varianza'])  # establezco un minimo del descriptor

    tiempos_detectados = detectar_crisis_por_umbral_todos_canales(senial_procesada, 'varianza', umbral_varianza)
    inicio_real_crisis = 120  # porque before dura 2 minutos

    print(f"\n--- Detección por canal usando umbral de varianza ---")
    for i, t in enumerate(tiempos_detectados):
        if t is not None:
            retardo = t - inicio_real_crisis
            if retardo < 0:
                retardo = -retardo
            print(f"Canal {i} ({etiquetas[i]}): Detectado en {t:.2f}s \t-> Retardo: {retardo:.2f} segundos")
        else:
            print(f"Canal {i} ({etiquetas[i]}): No detectado")

    desc_total = calcular_descriptores_completos(senal_total)
    varianzas = desc_total['varianza'].flatten()
    (mu, sigma), (loc_exp, scale_exp), (shape, loc_log, scale_log) = ajustar_y_graficar_pdf_normal_exponencial(varianzas, "varianza")


main()

