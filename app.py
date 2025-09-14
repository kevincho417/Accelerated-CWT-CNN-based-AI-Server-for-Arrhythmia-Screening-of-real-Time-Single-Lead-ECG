# app.py
import asyncio
import json
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import numpy as np
import uvicorn
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import HTMLResponse
from motor.motor_asyncio import AsyncIOMotorClient

# 假設這些模組存在於您的專案目錄中
from segment import segment
from tfa_morlet_112m import filter_fir1
from trt_CNN import CNN_processing as CNN_processing2

# --- App 和資料庫設定 ---

# MongoDB 連線設定
MONGO_DETAILS = "mongodb://192.168.50.28:27017"
client = None
db = None

# --- Lifespan Manager 來管理應用程式生命週期 ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    在應用程式啟動時建立資料庫連線，並在關閉時中斷連線。
    """
    global client, db
    print("應用程式啟動...")
    try:
        app.mongodb_client = AsyncIOMotorClient(MONGO_DETAILS, serverSelectionTimeoutMS=5000)
        await app.mongodb_client.admin.command('ismaster')
        app.db = app.mongodb_client["ECG_db"] # 資料庫名稱
        client = app.mongodb_client
        db = app.db
        print("成功連線到 MongoDB...")
    except Exception as e:
        print(f"無法連線到 MongoDB: {e}")
        client = None
        db = None
    
    yield # 應用程式在此處執行
    
    if client:
        print("應用程式關閉...")
        app.mongodb_client.close()
        print("已中斷 MongoDB 連線。")

app = FastAPI(lifespan=lifespan)

# --- 全域變數 ---
clients = []  # 儲存 WebSocket 客戶端連接
buffer = []   # 現在只儲存原始 (raw) 數據

# --- API 端點 ---

@app.post("/submit")
async def receive_data(request: Request):
    global buffer
    
    # 在函式開頭明確初始化本次請求會用到的區域變數
    seg_list = []
    rpeak_list = []
    prediction_list = []
    
    data_request = await request.json()
    # 將新的原始數據加到 buffer 的末尾
    data = buffer + data_request

    # 1. 訊號預處理 (對完整的拼接數據進行濾波)
    data_filtered = filter_fir1(data)
    data_list = data_filtered.tolist()

    # 2. R 波偵測與訊號切分
    rpeaks = segment(data_list)
    print(f"偵測到的 R-peaks 索引: {rpeaks}")

    last_processed_rpeak_location = 0

    if rpeaks.size > 0:
        for rpeak in rpeaks:
            # 檢查是否能從 R-peak 的位置前後取出完整的 368 個點
            if rpeak - 184 >= 0 and rpeak + 184 <= len(data_list):
                seg_start = rpeak - 184
                seg_end = rpeak + 184
                seg_one = data_list[seg_start:seg_end]
                
                # 再次確認長度以防萬一
                if len(seg_one) == 368:
                    seg_list.append(seg_one)
                    rpeak_list.append(int(rpeak))
                    # 更新最後一個被成功處理的 R-peak 在原始數據中的位置
                    last_processed_rpeak_location = rpeak
            else:
                # 如果R波的任一側超出範圍，代表這是區塊邊緣的 R-peak，
                # 我們將停止處理後續的 R-peak，並將它們留在 buffer 中等待下一次數據。
                break
    
    # ============================ 核心修正：Buffer 管理 ============================
    # 新的 buffer 將會是最後一個被成功處理的 R-peak 位置之後的所有 "原始" 數據。
    # 這樣可以確保 R-peak 不會被重複偵測。
    if last_processed_rpeak_location > 0:
        buffer = data[last_processed_rpeak_location:]
    else:
        # 如果沒有任何 R-peak 被處理，為了防止 buffer 無限增長，
        # 只保留最後一部分可能包含心跳的數據。
        buffer = data[-368:]
    # ===========================================================================


    # 3. 模型推論 (Triton)
    if seg_list:
        prediction_list = CNN_processing2(seg_list)
        if prediction_list:
            print(f"--- 模型推論結果 ---> {prediction_list}")

    # 4. 透過 WebSocket 廣播即時結果 (廣播的是本次請求接收到的完整數據塊)
    pred_list_j = json.dumps(prediction_list)
    rpeak_list_j = json.dumps(rpeak_list)
    combined_data = {
        "ecg_data": data_list, # 廣播濾波後的數據以供繪圖
        "rpeak_list": rpeak_list_j,
        "predictions": pred_list_j,
    }
    await asyncio.gather(*[client_ws.send_json(combined_data) for client_ws in clients])

    # 5. 將結果存入 MongoDB
    if prediction_list and db is not None:
        documents_to_insert = []
        for i in range(len(seg_list)):
            doc = {
                "seg_data": seg_list[i],
                "prediction": prediction_list[i],
                "timestamp": datetime.now(timezone.utc)
            }
            documents_to_insert.append(doc)
        
        result = await db["ecg_results"].insert_many(documents_to_insert)
        print(f"{len(result.inserted_ids)} 筆資料已成功寫入 MongoDB。")

    return {"status": "success", "processed_beats": len(prediction_list)}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    clients.append(websocket)
    try:
        while True:
            await websocket.receive_text()
    except Exception:
        clients.remove(websocket)


@app.websocket("/ws1")
async def websocket_history_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        if db is None:
            await websocket.send_json({"error": "Database not connected"})
            await asyncio.sleep(1)
            continue
        try:
            cursor = db["ecg_results"].find({}).sort("timestamp", -1)
            data = await cursor.to_list(length=1000)

            for item in data:
                item["_id"] = str(item["_id"])
                if "timestamp" in item and isinstance(item["timestamp"], datetime):
                    item["timestamp"] = item["timestamp"].isoformat()

            await websocket.send_json(data)
        except Exception as e:
            print("資料庫錯誤或 WebSocket 錯誤:", e)
            break 
        await asyncio.sleep(1)


@app.get("/", response_class=HTMLResponse)
async def get_html():
    try:
        with open("./data_display.html", encoding='utf-8') as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        return HTMLResponse(content="<h1>找不到 data_display.html 檔案</h1>", status_code=404)


@app.post("/clear-table")
async def clear_table():
    if db is None:
        return {"status": "error", "message": "Database not connected"}
    try:
        delete_result = await db["ecg_results"].delete_many({})
        return {"status": "success", "deleted_count": delete_result.deleted_count}
    except Exception as e:
        print("資料庫錯誤:", e)
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=7000, reload=True)

