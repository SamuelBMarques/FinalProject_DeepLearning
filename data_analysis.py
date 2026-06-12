import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# ============================================================================
# 1. Carregar e reamostrar os dados para 250 Hz
# ============================================================================

# Modifiquei o caminho para apontar para o arquivo que você subiu de exemplo
file_path = "physionet.org/files/respiratory-oximetry-apnoea/1.0.0/Inline_PQ_Data/Subject2/Subject2_8cmH2O_apnea2.csv"

# Lendo o CSV mapeando corretamente as colunas pelo nome presente no arquivo
data = pd.read_csv(file_path)

time_raw = data["Time [s]"].values
gauge_raw = data["Gauge Pressure [cmH2O]"].values
insp_raw = data["Inspiratory differential pressure [cmH2O]"].values

# Reamostragem para 250 Hz
fs_target = 250.0
dt_target = 1.0 / fs_target
t_new = np.arange(time_raw.min(), time_raw.max(), dt_target)

interp_gauge = interp1d(
    time_raw, gauge_raw, kind="linear", fill_value="extrapolate"
)
interp_insp = interp1d(
    time_raw, insp_raw, kind="linear", fill_value="extrapolate"
)

gauge_resampled = interp_gauge(t_new)
insp_resampled = interp_insp(t_new)

# ============================================================================
# 2. Detectar candidatos a apneia usando desvio padrão móvel
# ============================================================================

window_sec = 1.0  # Janela móvel de 1 segundo
window_samples = int(window_sec * fs_target)
df = pd.DataFrame({"insp": insp_resampled})

# Preenchemos os NaNs das bordas com o primeiro/último valor válido calculado
df["std"] = (
    df["insp"]
    .rolling(window_samples, center=True, min_periods=1)
    .std()
    .bfill()
    .ffill()
)

# Ajuste do limiar: 0.12 é ideal para ignorar ruído físico/cardíaco do sensor
std_threshold = 0.03
low_std_mask = (df["std"] < std_threshold).to_numpy()

# Converter máscara booleana em intervalos contíguos
regions = []
in_apnea = False
start_idx = 0

for i, is_low in enumerate(low_std_mask):
    if is_low and not in_apnea:
        in_apnea = True
        start_idx = i
    elif not is_low and in_apnea:
        in_apnea = False
        end_idx = i - 1
        duration = t_new[end_idx] - t_new[start_idx]
        # Flexibilidade na duração: aceita pausas a partir de 4 segundos
        if duration > 4.0:
            regions.append((start_idx, end_idx, duration))

# Tratar caso o sinal termine em apneia
if in_apnea:
    duration = t_new[-1] - t_new[start_idx]
    if duration > 4.0:
        regions.append((start_idx, len(low_std_mask) - 1, duration))

# ============================================================================
# 3. Classificar o tipo de apneia baseado no tempo do protocolo
# ============================================================================

apnea_events = []
for s_idx, e_idx, dur in regions:
    dur_rounded = round(dur)

    # Damos uma margem de tolerância para o início/fim físico do evento
    if 7 <= dur_rounded <= 13:
        apnea_type = "10s apnea"
    elif 14 <= dur_rounded <= 25:
        apnea_type = "20s apnea"
    else:
        apnea_type = f"Apnea alternativa ({dur_rounded}s)"

    start_time = t_new[s_idx]
    end_time = t_new[e_idx]
    apnea_events.append((start_time, end_time, apnea_type))

# Exibir eventos detectados no terminal
print("\n=== Eventos de Apneia Detectados ===")
if not apnea_events:
    print("Nenhum evento detectado. Tente subir levemente o 'std_threshold'.")
else:
    for start, end, atype in apnea_events:
        print(
            f"{atype}: {start:.2f}s – {end:.2f}s  (Duração Real = {end-start:.2f}s)"
        )

# ============================================================================
# 4. Plotagem Gráfica
# ============================================================================

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(
    t_new,
    insp_resampled,
    "b-",
    linewidth=0.8,
    label="Pressão Inspiratória Diferencial",
)

# Cores para cada tipo de evento
colors = {
    "10s apnea": "orange",
    "20s apnea": "red",
    "Apnea alternativa": "purple",
}

for start, end, atype in apnea_events:
    # Evita quebra se o nome não bater exato com o dicionário de cores
    cor = colors.get(atype, "purple")
    ax.axvspan(start, end, alpha=0.3, color=cor, label=atype)

# Remover duplicatas na legenda
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), loc="upper right")

ax.set_xlabel("Tempo [s]")
ax.set_ylabel("Pressão [cmH₂O]")
ax.set_title(f"Detecção Automática de Apneia – {file_path}")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()