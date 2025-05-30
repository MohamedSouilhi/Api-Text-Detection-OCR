import cv2
import easyocr
import numpy as np
import base64
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from io import BytesIO
from PIL import Image
import os
import mlflow
import time
import tempfile

# Initialiser l'application FastAPI
app = FastAPI(title="Text Detection API", description="API pour détecter du texte dans plusieurs images")

# Set MLflow tracking URI (local for this example)
mlflow.set_tracking_uri("file:///app/mlruns")
mlflow.set_experiment("TextDetectionAPI")

# Instance text detector with English and French support
reader = easyocr.Reader(['en', 'fr'], gpu=False)  # Added French for better detection, disabled GPU

# Fonction pour convertir une image en base64
def image_to_base64(img):
    _, buffer = cv2.imencode('.png', img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return img_base64

# Route pour détecter le texte dans plusieurs images
@app.post("/detect-text-multiple/")
async def detect_text_multiple(files: list[UploadFile] = File(...), threshold: float = 0.5, return_images: bool = False):
    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("threshold", threshold)
        mlflow.log_param("return_images", return_images)
        mlflow.log_param("languages", ['en', 'fr'])
        mlflow.log_param("contrast_ths", 0.1)
        mlflow.log_param("adjust_contrast", 0.5)
        mlflow.log_param("text_threshold", 0.7)
        mlflow.log_param("low_text", 0.4)
        mlflow.log_param("link_threshold", 0.4)

        try:
            start_time = time.time()
            results = []
            total_detections = 0
            total_confidence = 0.0

            for file in files:
                # Vérifier que le fichier est une image
                if not file.content_type.startswith("image/"):
                    continue  # Ignorer les fichiers non-images

                # Lire l'image uploadée
                contents = await file.read()
                nparr = np.frombuffer(contents, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                # Vérifier si l'image a été correctement chargée
                if img is None:
                    print(f"Error: Could not load image {file.filename}")
                    continue

                # Pas de prétraitement dans votre code actuel (img_processed = img)
                img_processed = img

                # Détecter le texte avec easyocr
                text_ = reader.readtext(img_processed, detail=1, paragraph=False, 
                                        contrast_ths=0.1, adjust_contrast=0.5, 
                                        text_threshold=0.7, low_text=0.4, link_threshold=0.4)

                # Traiter les résultats
                detections = []
                for t in text_:
                    bbox, text, score = t
                    if score > threshold:
                        # Convertir les coordonnées de la boîte englobante en entiers
                        pt1 = (int(bbox[0][0]), int(bbox[0][1]))  # Coin supérieur gauche
                        pt2 = (int(bbox[2][0]), int(bbox[2][1]))  # Coin inférieur droit

                        # Dessiner la boîte englobante et le texte sur l'image
                        cv2.rectangle(img_processed, pt1, pt2, (0, 255, 0), 5)
                        cv2.putText(img_processed, text, pt1, cv2.FONT_HERSHEY_COMPLEX, 0.65, (255, 0, 0), 2)

                        # Ajouter les résultats à la liste
                        detections.append({
                            "text": text,
                            "confidence": float(score),
                            "bounding_box": {
                                "top_left": pt1,
                                "bottom_right": pt2
                            }
                        })

                        total_detections += 1
                        total_confidence += score

                # Ajouter les résultats pour cette image
                image_result = {
                    "filename": file.filename,
                    "detections": detections
                }
                if return_images:
                    img_base64 = image_to_base64(img_processed)
                    image_result["annotated_image"] = f"data:image/png;base64,{img_base64}"

                    # Save the annotated image as an artifact
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
                        cv2.imwrite(temp_file.name, img_processed)
                        mlflow.log_artifact(temp_file.name, artifact_path="annotated_images")
                        os.remove(temp_file.name)

                results.append(image_result)

            # Log metrics
            processing_time = time.time() - start_time
            avg_confidence = total_confidence / total_detections if total_detections > 0 else 0.0
            mlflow.log_metric("processing_time", processing_time)
            mlflow.log_metric("total_detections", total_detections)
            mlflow.log_metric("average_confidence", avg_confidence)
            mlflow.log_metric("num_images_processed", len(results))

            return JSONResponse(content={"status": "success", "results": results})

        except Exception as e:
            mlflow.log_param("error", str(e))
            raise HTTPException(status_code=500, detail=f"Erreur lors du traitement des images : {str(e)}")

# Route de test pour vérifier que l'API fonctionne
@app.get("/")
async def root():
    return {"message": "Bienvenue sur l'API de détection de texte ! Utilisez /detect-text-multiple/ pour uploader des images."}