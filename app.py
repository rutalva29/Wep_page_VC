import os
import cv2
from flask import (
    Flask,
    request,
    render_template,
    redirect,
    url_for,
    send_from_directory,
)
from werkzeug.utils import secure_filename
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, Sequential
from tensorflow.keras.preprocessing import image

UPLOAD_FOLDER = "static/uploads/"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024


def create_fire_model():
    """Creates the MobileNetV2 model structure."""
    base_model = MobileNetV2(
        input_shape=(224, 224, 3), include_top=False, weights="imagenet"
    )
    base_model.trainable = False

    model = Sequential(
        [
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def load_trained_model():
    weights_path = "final_corrected_model.weights.h5"

    if not os.path.exists(weights_path):
        print(f" No se encuentra: {weights_path}")

        print(" Archivos disponibles:")

        for file in os.listdir("."):
            print(f"   - {file}")

        return None

    try:
        model = create_fire_model()

        print(" Cargando pesos entrenados...")

        model.load_weights(weights_path)

        print(" Modelo cargado exitosamente!")

        return model

    except Exception as e:
        print(f" Error cargando el modelo: {e}")

        return None


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS



def calc_geometric_mean(channel):
    """Calculates geometric mean of a channel."""

    epsilon = 1e-5
    log_c = np.log(channel.astype(np.float32) + epsilon)
    return np.exp(np.mean(log_c))

def algorithm_1_roi(rgb_img):
    """
    Algorithm 1: Region of interest
    Extracts global candidates for fire.
    """
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

    alpha_prime = (2 * alpha_R) - (2 * alpha_B)
    alpha_prime = np.clip(alpha_prime, 0, 255).astype(np.uint8)

    thresh_val, gamma = cv2.threshold(alpha_prime, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(gamma, connectivity=8)
    final_roi = np.zeros_like(gamma)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= 8:
            final_roi[labels == i] = 255
            
    kernel = np.ones((3, 3), np.uint8)
    final_roi = cv2.morphologyEx(final_roi, cv2.MORPH_CLOSE, kernel)
    
    return final_roi

def algorithm_3_get_thresholds(rgb_stripe):
    """
    Algorithm 3: Calculation of thresholds
    Analyzes histograms to find dynamic thresholds tau1, tau2.
    """
    hist_r = cv2.calcHist([rgb_stripe], [0], None, [256], [0, 256]).flatten()
    hist_g = cv2.calcHist([rgb_stripe], [1], None, [256], [0, 256]).flatten()
    hist_b = cv2.calcHist([rgb_stripe], [2], None, [256], [0, 256]).flatten()


    eta = np.where((hist_r > hist_g) & (hist_g > hist_b))[0]

    tau1 = np.median(eta) 
    tmp = eta[0]

    if len(eta) > 0:
        found_tau1 = False


        for i in range(1, len(eta)):
            if eta[i] < (tmp + 10):
                tmp = eta[i]
            else:
                tau1 = float(eta[i])
                found_tau1 = True
                break
        
        if not found_tau1:
            if eta[-1] > 100:
                tau1 = np.mean(eta)
            else:
                tau1 = float(eta[-1])
    else:
        tau1 = 0.0 
    

    eta2 = np.where((hist_b > hist_g) & (hist_b > hist_r))[0]

    tau2_raw = 0.0
    tmp = eta2[0]

    if len(eta2) > 0:
        found_tau2 = False
        
        for i in range(1, len(eta2)):
            if eta2[i] < (tmp + 5):
                tmp = eta2[i-1] 
            else:
                tau2_raw = float(eta2[i])
                found_tau2 = True
                break
        
        if not found_tau2:
            tau2_raw = float(eta2[-1])


    if tau2_raw <= 3:
        q_tau2 = 40.0
    elif tau2_raw <= 5:
        q_tau2 = tau2_raw*10.0
    elif tau2_raw <= 10:
        q_tau2 = tau2_raw*5.0
    else:
        q_tau2 = 5.0

    return tau1, q_tau2

def algorithm_2_stripes(rgb_img, orientation='horizontal'):
    """
    Algorithm 2: Horizontal and vertical image stripes
    """
    rows, cols, _ = rgb_img.shape
    

    tau1_star, tau2_star = algorithm_3_get_thresholds(rgb_img)

    processed_stripes = []
    
    num_stripes = 3
    
    if orientation == 'horizontal':
        stripes_data = np.array_split(rgb_img, num_stripes, axis=0)
    else:
        stripes_data = np.array_split(rgb_img, num_stripes, axis=1)

    for i, S_i in enumerate(stripes_data):
        if S_i.size == 0:
            processed_stripes.append(S_i)
            continue

        tau1_i, tau2_i = algorithm_3_get_thresholds(S_i)

        gray_S = cv2.cvtColor(S_i, cv2.COLOR_RGB2GRAY).astype(np.float32)
        if gray_S.size > 1:
            sigma_i = np.std(gray_S, ddof=1)
        else:
            sigma_i = 0.0

        epsilon_i = (255.0 + sigma_i) / 255.0

 
        threshold_val = tau1_star / (tau2_star * 2.0)
        
        if epsilon_i < threshold_val:
            delta_i = threshold_val
        else:
            delta_i = epsilon_i

        if tau1_star > tau1_i:
            final_tau1 = tau1_star
        else:
            final_tau1 = tau1_i

        R = S_i[:,:,0].astype(np.float32)
        G = S_i[:,:,1].astype(np.float32)
        B = S_i[:,:,2].astype(np.float32)

        g_i = np.ones_like(R, dtype=np.uint8) * 255
        
        cond1 = (final_tau1 > R) & (B >= R)
        g_i[cond1] = 0
        
        cond2 = (B > G) & (B >= R)
        g_i[cond2] = 0
        
        cond3 = ((G + B) > R) & ((B * delta_i) > G)
        g_i[cond3] = 0
        

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(g_i, connectivity=8)
        g_hat_i = np.zeros_like(g_i)
        for k in range(1, num_labels):
            if stats[k, cv2.CC_STAT_AREA] >= 8:
                g_hat_i[labels == k] = 255


        kernel = np.ones((3, 3), np.uint8)
        h_i = cv2.morphologyEx(g_hat_i, cv2.MORPH_CLOSE, kernel)

        r_ij = cv2.bitwise_and(S_i[:,:,0], S_i[:,:,0], mask=h_i)
        
        processed_stripes.append(r_ij)

    if orientation == 'horizontal':
        return np.vstack(processed_stripes)
    else:
        return np.hstack(processed_stripes)    

def algorithm_4_seed_selection(R_h, R_v):
    """
    Algorithm 4: Seed selection
    Combines horizontal and vertical stripe results using strict binary multiplication.
    """

    p_h = np.where(R_h > 0, 1, 0).astype(np.uint8)
    
    p_v = np.where(R_v > 0, 1, 0).astype(np.uint8)
    
    f = p_h * p_v
    

    return f * 255

def buza_akagic_full_method(filepath):
    try:
        bgr_img = cv2.imread(filepath)
        if bgr_img is None: return None
        

        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

        roi_mask = algorithm_1_roi(rgb_img)

        R_h_result = algorithm_2_stripes(rgb_img, orientation='vertical') 
        
        R_v_result = algorithm_2_stripes(rgb_img, orientation='horizontal')

        seed_mask = algorithm_4_seed_selection(R_h_result, R_v_result)
        
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(roi_mask, connectivity=8)
        final_mask = np.zeros_like(roi_mask)
        
        has_seeds = np.sum(seed_mask) > 0
        
        if not has_seeds:
            final_mask = np.zeros_like(roi_mask)
        else:
            for i in range(1, num_labels):
                component_mask = (labels == i).astype(np.uint8) * 255
                
                intersection = cv2.bitwise_and(component_mask, seed_mask)
                
                if np.sum(intersection) > 0:
                    final_mask = cv2.bitwise_or(final_mask, component_mask)


        vis_img = bgr_img.copy()
        
        contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            cv2.drawContours(vis_img, contours, -1, (0, 255, 255), 2)
            
            overlay = np.zeros_like(vis_img)
            overlay[:] = [0, 0, 255] 
            masked_overlay = cv2.bitwise_and(overlay, overlay, mask=final_mask)
            vis_img = cv2.addWeighted(vis_img, 1.0, masked_overlay, 0.5, 0)
            
        return vis_img

    except Exception as e:
        print(f"Algorithm Error: {e}")
        return None

def process_image(filepath, model, threshold=0.6):
    try:
        if not os.path.exists(filepath):
            return filepath, "Error: Image file not found.", False

        img_load = image.load_img(filepath, target_size=(224, 224))
        img_array = image.img_to_array(img_load)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        prediction = model.predict(img_array, verbose=0)
        confidence = float(prediction[0][0])
        
        processed_image_path = filepath
        fire_detected = False
        text_result = f"All Clear: No Fire Detected. (Confidence: {1-confidence:.2%})"

        if confidence > threshold:
            fire_detected = True
            text_result = f"Warning: Fire Detected! (Confidence: {confidence:.2%})"

            segmented_img = buza_akagic_full_method(filepath)
            
            if segmented_img is not None:
                base_dir = os.path.dirname(filepath)
                filename = os.path.basename(filepath)
                name, ext = os.path.splitext(filename)
                new_filename = f"{name}_paper_algo{ext}"
                processed_image_path = os.path.join(base_dir, new_filename)
                
                cv2.imwrite(processed_image_path, segmented_img)

        return processed_image_path, text_result, fire_detected

    except Exception as e:
        print(f"Error processing image {filepath}: {e}")
        return filepath, f"Error: {e}", False
       
@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            if not os.path.exists(app.config["UPLOAD_FOLDER"]):
                os.makedirs(app.config["UPLOAD_FOLDER"])
            file.save(filepath)
            processed_image_path, result_text, fire_detected = process_image(
                filepath, fire_model
            )
            processed_filename = os.path.basename(processed_image_path)
            return render_template(
                "result.html",
                image_name=processed_filename,
                result_text=result_text,
                fire_detected=fire_detected,
            )
    return render_template("index.html")


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


fire_model = load_trained_model()

if __name__ == "__main__":
    if fire_model is None:
        print("Could not load model. Exiting.")
        exit()

    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=False, port=5000, host="0.0.0.0")