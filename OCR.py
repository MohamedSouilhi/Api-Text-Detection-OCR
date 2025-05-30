import os
import io
import json
import time
import base64
import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import List
import mlflow
import psycopg2
from minio import Minio
from minio.error import S3Error
import easyocr
import PyPDF2
from pdf2image import convert_from_bytes
from fastapi.responses import JSONResponse
from fastapi import Request
# Configure MLflow tracking URI
mlflow.set_tracking_uri("http://mlflow:5000")
# Initialize FastAPI app
app = FastAPI()

# Environment variables
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", 5432))
POSTGRES_USER = os.getenv("POSTGRES_USER", "admin")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "password")
POSTGRES_DB = os.getenv("POSTGRES_DB", "ocr_db")
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "miniopassword")
MINIO_BUCKET = "ocr-bucket"

# Initialize MinIO client
try:
    minio_client = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False
    )
    # Ensure MinIO bucket exists
    if not minio_client.bucket_exists(MINIO_BUCKET):
        minio_client.make_bucket(MINIO_BUCKET)
except S3Error as e:
    raise Exception(f"Failed to initialize MinIO: {str(e)}")

# Initialize PostgreSQL connection
try:
    conn = psycopg2.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        database=POSTGRES_DB
    )
    cursor = conn.cursor()
except psycopg2.Error as e:
    raise Exception(f"Failed to connect to PostgreSQL: {str(e)}")

# Create table for detections if it doesn't exist
try:
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id SERIAL PRIMARY KEY,
            filename VARCHAR(255),
            original_url VARCHAR(255),
            annotated_url VARCHAR(255),
            text TEXT,
            confidence FLOAT,
            bounding_box JSONB,
            page_number INTEGER,
            is_embedded_image BOOLEAN,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
except psycopg2.Error as e:
    raise Exception(f"Failed to create table in PostgreSQL: {str(e)}")

# Initialize EasyOCR reader
try:
    reader = easyocr.Reader(['en'], gpu=False)
except Exception as e:
    print(f"Error initializing EasyOCR: {str(e)}")
    reader = None

def detect_text_from_file(file_data, filename, threshold=0.5, return_images=False):
    """
    Detect text from a file (image or PDF) and return detections with optional annotated images.
    
    Args:
        file_data (bytes): File content in bytes.
        filename (str): Name of the file.
        threshold (float): Confidence threshold for OCR detections.
        return_images (bool): Whether to return base64-encoded annotated images.
    
    Returns:
        tuple: (detections, total_detections, total_confidence, images_base64)
    """
    if not file_data:
        return [{"text": "Error: No file data provided", "confidence": 0.0, "bounding_box": None, "page_number": None, "is_embedded_image": False, "source": "error"}], 0, 0, []

    content_type = "application/pdf" if filename.lower().endswith('.pdf') else "image"
    print(f"\nProcessing file: {filename} (Type: {content_type})")

    detections = []
    total_detections = 0
    total_confidence = 0
    images_base64 = []

    # Process PDF or image
    if content_type == "application/pdf":
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_data))
            pdf_text = ""
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                if text:
                    pdf_text += text

            print(f"Extracted text from PDF: {pdf_text[:100]}...")

            if pdf_text.strip():
                print("Using directly extracted text from PDF.")
                detections.append({
                    "text": pdf_text,
                    "confidence": 1.0,
                    "bounding_box": None,
                    "page_number": None,
                    "is_embedded_image": False,
                    "source": "direct_extraction"
                })
                total_detections += 1
                total_confidence += 1.0
            else:
                print("No selectable text found. Converting PDF to images for OCR...")
                try:
                    images = convert_from_bytes(file_data)
                    print(f"Converted PDF to {len(images)} image(s).")
                except Exception as e:
                    print(f"Error converting PDF to images: {str(e)}")
                    detections.append({
                        "text": f"Error converting PDF to images: {str(e)}",
                        "confidence": 0.0,
                        "bounding_box": None,
                        "page_number": None,
                        "is_embedded_image": False,
                        "source": "error"
                    })
                    return detections, total_detections, total_confidence, images_base64

                # Process converted images
                for page_num, image in enumerate(images, 1):
                    try:
                        image_np = np.array(image)
                        image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                        # Optional preprocessing: adjust contrast
                        image_cv = cv2.convertScaleAbs(image_cv, alpha=1.5, beta=0)

                        if reader is None:
                            raise ValueError("EasyOCR reader not initialized")

                        ocr_results = reader.readtext(image_cv, contrast_ths=0.1, adjust_contrast=0.5)

                        for (bbox, text, confidence) in ocr_results:
                            if confidence >= threshold:
                                top_left = [int(bbox[0][0]), int(bbox[0][1])]
                                bottom_right = [int(bbox[2][0]), int(bbox[2][1])]
                                detections.append({
                                    "text": text,
                                    "confidence": confidence,
                                    "bounding_box": {
                                        "top_left": top_left,
                                        "bottom_right": bottom_right
                                    },
                                    "page_number": page_num,
                                    "is_embedded_image": False,
                                    "source": "ocr"
                                })
                                total_detections += 1
                                total_confidence += confidence

                        if return_images:
                            for detection in detections:
                                if detection["page_number"] == page_num:
                                    top_left = detection["bounding_box"]["top_left"]
                                    bottom_right = detection["bounding_box"]["bottom_right"]
                                    cv2.rectangle(image_cv, top_left, bottom_right, (0, 255, 0), 2)

                            image_pil = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
                            buffered = io.BytesIO()
                            image_pil.save(buffered, format="PNG")
                            image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                            images_base64.append({
                                "page_number": page_num,
                                "is_embedded_image": False,
                                "image_base64": image_base64
                            })
                    except Exception as e:
                        print(f"Error processing page {page_num}: {str(e)}")
                        detections.append({
                            "text": f"Error processing page {page_num}: {str(e)}",
                            "confidence": 0.0,
                            "bounding_box": None,
                            "page_number": page_num,
                            "is_embedded_image": False,
                            "source": "error"
                        })

        except Exception as e:
            print(f"Error processing PDF {filename}: {str(e)}")
            detections.append({
                "text": f"Error processing PDF: {str(e)}",
                "confidence": 0.0,
                "bounding_box": None,
                "page_number": None,
                "is_embedded_image": False,
                "source": "error"
            })
    else:
        # Process image files
        try:
            image = cv2.imdecode(np.frombuffer(file_data, np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Error decoding image {filename}")

            if reader is None:
                raise ValueError("EasyOCR reader not initialized")

            ocr_results = reader.readtext(image, contrast_ths=0.1, adjust_contrast=0.5)

            for (bbox, text, confidence) in ocr_results:
                if confidence >= threshold:
                    top_left = [int(bbox[0][0]), int(bbox[0][1])]
                    bottom_right = [int(bbox[2][0]), int(bbox[2][1])]
                    detections.append({
                        "text": text,
                        "confidence": confidence,
                        "bounding_box": {
                            "top_left": top_left,
                            "bottom_right": bottom_right
                        },
                        "page_number": None,
                        "is_embedded_image": False,
                        "source": "ocr"
                    })
                    total_detections += 1
                    total_confidence += confidence

            if return_images:
                for detection in detections:
                    top_left = detection["bounding_box"]["top_left"]
                    bottom_right = detection["bounding_box"]["bottom_right"]
                    cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)

                image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                buffered = io.BytesIO()
                image_pil.save(buffered, format="PNG")
                image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                images_base64.append({
                    "page_number": None,
                    "is_embedded_image": False,
                    "image_base64": image_base64
                })
        except Exception as e:
            print(f"Error processing image {filename}: {str(e)}")
            detections.append({
                "text": f"Error processing image: {str(e)}",
                "confidence": 0.0,
                "bounding_box": None,
                "page_number": None,
                "is_embedded_image": False,
                "source": "error"
            })

    return detections, total_detections, total_confidence, images_base64
# Middleware pour capturer et afficher toutes les exceptions
@app.middleware("http")
async def catch_exceptions_middleware(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as e:
        print(f"Exception caught: {str(e)}")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"detail": f"Internal Server Error: {str(e)}"}
        )
# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Bienvenue sur l'API de détection de texte ! Utilisez /detect-text-multiple/ pour uploader des images ou des PDFs."
    }

# Endpoint to detect text from multiple files
@app.post("/detect-text-multiple/")
async def detect_text_multiple(files: List[UploadFile] = File(...), threshold: float = 0.5, return_images: bool = False):
    start_time = time.time()
    results = []
    total_detections = 0
    total_confidence = 0

    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    # Temporarily disable MLflow
    # try:
    #     with mlflow.start_run():
    #         mlflow.log_param("threshold", threshold)
    #         mlflow.log_param("num_files", len(files))
    #         mlflow.log_param("return_images", return_images)

    for file in files:
        filename = file.filename

        # Read file data
        try:
            file_data = await file.read()
            if not file_data:
                results.append({
                    "filename": filename,
                    "detections": [{"text": "Error: Empty file", "confidence": 0.0, "bounding_box": None, "page_number": None, "is_embedded_image": False, "source": "error"}],
                    "images_base64": []
                })
                continue
        except Exception as e:
            results.append({
                "filename": filename,
                "detections": [{"text": f"Error reading file: {str(e)}", "confidence": 0.0, "bounding_box": None, "page_number": None, "is_embedded_image": False, "source": "error"}],
                "images_base64": []
            })
            continue

        # Save original file to MinIO
        original_path = f"originals/{filename}"
        try:
            minio_client.put_object(MINIO_BUCKET, original_path, io.BytesIO(file_data), len(file_data))
        except S3Error as e:
            results.append({
                "filename": filename,
                "detections": [{"text": f"Failed to save file to MinIO: {str(e)}", "confidence": 0.0, "bounding_box": None, "page_number": None, "is_embedded_image": False, "source": "error"}],
                "images_base64": []
            })
            continue

        # Detect text using the model
        try:
            detections, file_detections, file_confidence, images_base64 = detect_text_from_file(file_data, filename, threshold, return_images)
        except Exception as e:
            results.append({
                "filename": filename,
                "detections": [{"text": f"Error in OCR processing: {str(e)}", "confidence": 0.0, "bounding_box": None, "page_number": None, "is_embedded_image": False, "source": "error"}],
                "images_base64": []
            })
            continue

        # Update totals
        total_detections += file_detections
        total_confidence += file_confidence

        # Save detections to PostgreSQL and MinIO
        annotated_paths = {}
        if return_images:
            for img_info in images_base64:
                page_num = img_info["page_number"]
                is_embedded = img_info["is_embedded_image"]
                image_base64 = img_info["image_base64"]

                # Decode base64 image to save to MinIO
                try:
                    image_bytes = base64.b64decode(image_base64)
                    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)

                    # Generate annotated filename
                    if page_num:
                        annotated_filename = f"{filename}_page_{page_num}.png"
                    else:
                        annotated_filename = filename
                    annotated_path = f"annotated/{annotated_filename}"

                    # Save annotated image to MinIO
                    _, buffer = cv2.imencode(".png", image)
                    minio_client.put_object(MINIO_BUCKET, annotated_path, io.BytesIO(buffer.tobytes()), len(buffer))
                    annotated_paths[(page_num, is_embedded)] = annotated_path
                except Exception as e:
                    print(f"Failed to save annotated image to MinIO: {str(e)}")

        for detection in detections:
            # Prepare data for PostgreSQL
            text = detection["text"]
            confidence = detection["confidence"]
            bounding_box = detection.get("bounding_box")
            page_number = detection.get("page_number")
            is_embedded_image = detection["is_embedded_image"]

            # Determine annotated URL
            annotated_url = None
            if return_images and (page_number, is_embedded_image) in annotated_paths:
                annotated_url = annotated_paths[(page_number, is_embedded_image)]

            # Insert detection into PostgreSQL
            try:
                cursor.execute(
                    """
                    INSERT INTO detections (filename, original_url, annotated_url, text, confidence, bounding_box, page_number, is_embedded_image)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (filename, original_path, annotated_url, text, confidence, json.dumps(bounding_box) if bounding_box else None, page_number, is_embedded_image)
                )
                conn.commit()
            except psycopg2.Error as e:
                conn.rollback()
                print(f"Failed to insert detection into PostgreSQL: {str(e)}")

        # Prepare result for this file
        result = {
            "filename": filename,
            "detections": detections,
            "images_base64": images_base64 if return_images else []
        }
        results.append(result)

    # Log metrics to MLflow (temporarily disabled)
    # processing_time = time.time() - start_time
    # average_confidence = total_confidence / total_detections if total_detections > 0 else 0
    # mlflow.log_metric("processing_time", processing_time)
    # mlflow.log_metric("total_detections", total_detections)
    # mlflow.log_metric("average_confidence", average_confidence)

    # except Exception as e:
    #     raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

    # Return response
    return {
        "status": "success",
        "results": results
    }

# Cleanup on shutdown
@app.on_event("shutdown")
def shutdown_event():
    cursor.close()
    conn.close()