import os
import cv2
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, Sequential

# Configuración inicial de Streamlit (DEBE SER LO PRIMERO)
st.set_page_config(page_title="Detector de Incendios", layout="wide")

# --- FUNCIONES DE MODELO ---
def create_fire_model():
    base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights="imagenet")
    base_model.trainable = False
    model = Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss="binary_crossentropy", metrics=["accuracy"])
    return model

@st.cache_resource
def load_trained_model():
    weights_path = "final_corrected_model.weights.h5"
    if not os.path.exists(weights_path):
        st.error(f"No se encontró el archivo de pesos: {weights_path}")
        return None
    try:
        model = create_fire_model()
        model.load_weights(weights_path)
        return model
    except Exception as e:
        st.error(f"Error al cargar pesos: {e}")
        return None

# --- ALGORITMOS MATEMÁTICOS (BUZA-AKAGIC) ---
def calc_geometric_mean(channel):
    epsilon = 1e-5
    log_c = np.log(channel.astype(np.float32) + epsilon)
    return np.exp(np.mean(log_c))

def algorithm_1_roi(rgb_img):
    I = rgb_img.astype(np.float32)
    mu_R = calc_geometric_mean(I[:, :, 0])
    mu_G = calc_geometric_mean(I[:, :, 1])
    mu_B = calc_geometric_mean(I[:, :, 2])
    I_prime_R = I[:, :, 0] - mu_R
    I_prime_G = I[:, :, 1] - mu_G
    I_prime_B = I[:, :, 2] - mu_B
    mask_beta = np.ones_like(I_prime_R, dtype=np.uint8)
    mask_beta[I_prime_R < I_prime_B] = 0
    mask_beta[(I_prime_R + I_prime_B) < I_prime_G] = 0
    alpha_R = I[:, :, 0] * mask_beta
    alpha_B = I[:, :, 2] * mask_beta
    alpha_prime = np.clip((2 * alpha_R) - (2 * alpha_B), 0, 255).astype(np.uint8)
    _, gamma = cv2.threshold(alpha_prime, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(gamma, connectivity=8)
    final_roi = np.zeros_like(gamma)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= 8:
            final_roi[labels == i] = 255
    return cv2.morphologyEx(final_roi, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

def algorithm_3_get_thresholds(rgb_stripe):
    hist_r = cv2.calcHist([rgb_stripe], [0], None, [256], [0, 256]).flatten()
    hist_g = cv2.calcHist([rgb_stripe], [1], None, [256], [0, 256]).flatten()
    hist_b = cv2.calcHist([rgb_stripe], [2], None, [256], [0, 256]).flatten()
    eta = np.where((hist_r > hist_g) & (hist_g > hist_b))[0]
    tau1 = np.median(eta) if len(eta) > 0 else 0.0
    eta2 = np.where((hist_b > hist_g) & (hist_b > hist_r))[0]
    tau2_raw = float(eta2[-1]) if len(eta2) > 0 else 0.0
    q_tau2 = 5.0 # Simplificación para estabilidad
    return tau1, q_tau2

def algorithm_2_stripes(rgb_img, orientation='horizontal'):
    tau1_star, tau2_star = algorithm_3_get_thresholds(rgb_img)
    axis = 0 if orientation == 'horizontal' else 1
    stripes_data = np.array_split(rgb_img, 3, axis=axis)
    processed_stripes = []
    for S_i in stripes_data:
        if S_i.size == 0: continue
        tau1_i, _ = algorithm_3_get_thresholds(S_i)
        gray_S = cv2.cvtColor(S_i, cv2.COLOR_RGB2GRAY).astype(np.float32)
        epsilon_i = (255.0 + np.std(gray_S)) / 255.0
        delta_i = max(epsilon_i, tau1_star / (tau2_star * 2.0 + 1e-5))
        f_tau1 = max(tau1_star, tau1_i)
        R, G, B = S_i[:,:,0].astype(np.float32), S_i[:,:,1].astype(np.float32), S_i[:,:,2].astype(np.float32)
        g_i = np.where((f_tau1 > R) & (B >= R) | (B > G) & (B >= R) | ((G + B) > R) & ((B * delta_i) > G), 0, 255).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(g_i, connectivity=8)
        h_i = np.zeros_like(g_i)
        for k in range(1, num_labels):
            if stats[k, cv2.CC_STAT_AREA] >= 8: h_i[labels == k] = 255
        processed_stripes.append(cv2.bitwise_and(S_i[:,:,0], S_i[:,:,0], mask=h_i))
    return np.vstack(processed_stripes) if orientation == 'horizontal' else np.hstack(processed_stripes)

def buza_akagic_segmentation(rgb_img):
    roi_mask = algorithm_1_roi(rgb_img)
    R_h = algorithm_2_stripes(rgb_img, 'vertical')
    R_v = algorithm_2_stripes(rgb_img, 'horizontal')
    seed_mask = ((R_h > 0) & (R_v > 0)).astype(np.uint8) * 255
    num_labels, labels, _, _ = cv2.connectedComponentsWithStats(roi_mask, connectivity=8)
    final_mask = np.zeros_like(roi_mask)
    if np.sum(seed_mask) > 0:
        for i in range(1, num_labels):
            comp = (labels == i).astype(np.uint8) * 255
            if np.sum(cv2.bitwise_and(comp, seed_mask)) > 0:
                final_mask = cv2.bitwise_or(final_mask, comp)
    
    vis = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, contours, -1, (0, 255, 255), 2)
    return vis

# --- LÓGICA DE PROCESAMIENTO ---
def main():
    st.title("🔥 Sistema de Detección de Fuego")
    st.write("Análisis híbrido: Red Neuronal (MobileNetV2) + Algoritmo Buza-Akagic")

    model = load_trained_model()
    file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])

    if file and model:
        # Leer imagen
        img_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(img_bytes, 1)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # 1. IA Prediction
        img_ai = cv2.resize(img_rgb, (224, 224)) / 255.0
        prediction = model.predict(np.expand_dims(img_ai, axis=0), verbose=0)
        confidence = float(prediction[0][0])

        # 2. Mostrar Resultados
        st.divider()
        if confidence > 0.6:
            st.error(f"¡PELIGRO: FUEGO DETECTADO! (Confianza: {confidence:.2%})")
            with st.spinner("Ejecutando segmentación matemática..."):
                res_img = buza_akagic_segmentation(img_rgb)
            
            col1, col2 = st.columns(2)
            col1.image(img_rgb, caption="Imagen Original")
            col2.image(cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB), caption="Segmentación Buza-Akagic")
        else:
            st.success(f"Todo despejado: No se detecta fuego ({1-confidence:.2%})")
            st.image(img_rgb, width=600)

if __name__ == "__main__":
    main()
            st.image(processed_path, caption="Detección Buza-Akagic", use_container_width=True)
    else:
        st.success(result_text)
        st.image(filepath, caption="Imagen procesada", use_container_width=True)
