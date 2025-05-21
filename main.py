
from fastapi import FastAPI, Depends, HTTPException, status, File, UploadFile
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from auth import authenticate_user, create_access_token, get_current_user, create_user
from PIL import Image
import numpy as np
import tensorflow as tf
import io
import os
import uvicorn
from datetime import datetime

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
def root():
    return {"message": "Cannacore API funcionando correctamente"}

class UserRegister(BaseModel):
    username: str
    password: str

@app.on_event("startup")
def ensure_default_user():
    create_user("admin", "adminpass")

@app.post("/register")
def register(user: UserRegister):
    result = create_user(user.username, user.password)
    if not result:
        raise HTTPException(status_code=400, detail="El usuario ya existe")
    return {"message": "Usuario creado correctamente"}

@app.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Credenciales incorrectas")
    access_token = create_access_token(data={"sub": user["username"]})
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/predict")
async def predict_image(file: UploadFile = File(...), current_user: dict = Depends(get_current_user)):
    # Crear nombre único con timestamp
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    file_ext = os.path.splitext(file.filename)[1]
    saved_filename = f"{current_user['username']}_{timestamp}{file_ext}"
    file_path = os.path.join(UPLOAD_DIR, saved_filename)

    contents = await file.read()
    with open(file_path, "wb") as f:
        f.write(contents)

    # Procesar imagen para IA
    image = Image.open(io.BytesIO(contents)).resize((224, 224))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    model = tf.keras.models.load_model("model.h5")
    prediction = model.predict(image_array)

    return {
        "filename": saved_filename,
        "prediction": prediction.tolist()
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
