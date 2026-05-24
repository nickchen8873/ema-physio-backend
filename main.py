import os
from dotenv import load_dotenv
import jwt
import joblib
import pandas as pd
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel

# 載入 .env 檔案中的變數到系統環境變數中
load_dotenv()

# 使用 os.getenv() 讀取變數
SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET")
ALGORITHM = os.getenv("ALGORITHM")

# 告訴 FastAPI 我們的 Token 會從 Header 的 Bearer 傳進來
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/login")

# 2. 初始化 FastAPI 應用程式
app = FastAPI(title="EMA Physio API", version="1.0.0")

# 1. 設定 CORS (開發階段先全開，上線時記得改成 Vue 前端的真實網址)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # ⚠️ 正式環境請改為前端 Domain，例如 ["[https://your-vue-app.com](https://your-vue-app.com)"]
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. 載入模型 (在伺服器啟動時載入記憶體，提升 API 回應速度)
try:
    model = joblib.load('models/stress_prediction_rf_model.pkl')
    print("✅ 預測模型載入成功！")
except Exception as e:
    print(f"❌ 模型載入失敗，請確認檔案路徑: {e}")

# 4. 定義前端傳入的資料結構 (Pydantic 自動驗證型別)
class PhysioFeatures(BaseModel):
    sleep_yesterday: float
    sleep_7d_std: float
    is_weekend: int
    mvpa_yesterday: float
    line_user_id: str = None  # 💡 新增這一行，讓前端可以直接傳 LINE ID 進來

# --- 1. 驗證 Token 的依賴函數 ---
def verify_token(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(
            token, 
            SUPABASE_JWT_SECRET, 
            algorithms=["HS256"], 
            audience="authenticated"
        )
        # 👇 關鍵提取：Supabase 的 JWT 裡面，'sub' 就是使用者的 UUID
        user_id = payload.get("sub")
        return {"user_id": user_id, "email": payload.get("email")} # 回傳更豐富的資訊
        
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"無效的 Token: {str(e)}")

# --- 2. 登入並核發 Token 的端點 ---
class LoginRequest(BaseModel):
    password: str

# 5. 實作預測 API 端點
@app.post("/api/predict-stress")
def predict_daily_stress(features: PhysioFeatures):
    # 改從 features 裡面拿 line_user_id
    user_id = features.line_user_id or "匿名使用者"
    print(f"🕵️ 正在為使用者 {user_id} 計算壓力分數...")
    
    try:
        # 將 Pydantic 模型轉為 DataFrame，確保特徵名稱與訓練時完全一致
        input_data = pd.DataFrame([{
            'sleep_yesterday': features.sleep_yesterday,
            'sleep_7d_std': features.sleep_7d_std,
            'is_weekend': features.is_weekend,
            'mvpa_yesterday': features.mvpa_yesterday
        }])
        
        # 執行預測
        predicted_score = model.predict(input_data)[0]
        
        return {
            "status": "success",
            "data": {
                "predicted_stress_level": round(predicted_score, 2),
                # 假設 3.5 分為高風險閥值，可依需求調整
                "warning_flag": True if predicted_score > 3.0 else False 
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"預測過程中發生錯誤: {str(e)}")