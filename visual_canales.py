import matplotlib.pyplot as plt
from pyedflib import highlevel
import numpy as np


# Cargar datos
senales, headers, _ = highlevel.read_edf("chb05_13.edf")
print(headers)
etiquetas = [h["label"] for h in headers]

# Elegimos 3 canales para mostrar
canales_mostrar = [0, 1, 2]
tiempo = np.arange(senales.shape[1]) / headers[0]["sample_frequency"]  # tiempo en segundos

plt.figure(figsize=(12, 6))
for idx in canales_mostrar:
    plt.plot(tiempo[:1000], senales[idx, :1000], label=etiquetas[idx])  # 1000 muestras

plt.title("Primeros 3 canales del EEG - 4 segundos de señal")
plt.xlabel("Tiempo (segundos)")
plt.ylabel("Voltaje (uV)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
