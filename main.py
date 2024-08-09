from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
import cv2
from image_processing import preprocess_image
from barcode_generation import generate_barcode
import uvicorn

app = FastAPI()

@app.post("/")
async def generate_persistence_barcode(file: UploadFile = File(...)):
    try:
 
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED) 
        
        if img is None:
            return JSONResponse(content={"error": "Failed to decode image"}, status_code=400)
        
        print(f"Image type: {img.dtype}, shape: {img.shape}")

        preprocessed_img = preprocess_image(img)
        barcode_image = generate_barcode(preprocessed_img)
        
        return JSONResponse(content={
            "message": "Barcode generated successfully",
            "barcode_image": barcode_image
        }, status_code=200)
    
    except Exception as e:
        # Log the exception message for debugging
        print(f"Exception occurred: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)