import argparse
import sys
import time
from functools import partial

import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException

# --- 程式碼設定 ---
label=['F','Q' , 'N','V', 'S']

# --- 命令列參數解析 ---
parser = argparse.ArgumentParser()
parser.add_argument(
    "-v",
    "--verbose",
    action="store_true",
    required=False,
    default=False,
    help="Enable verbose output",
)
parser.add_argument(
    "-u",
    "--url",
    type=str,
    required=False,
    default="192.168.50.28:8001",
    help="Inference server URL. Default is 192.168.50.28:8001.",
)
parser.add_argument(
    "-t",
    "--client-timeout",
    type=float,
    required=False,
    default=None,
    help="Client timeout in seconds. Default is None.",
)

FLAGS = parser.parse_args()

# --- Triton Client 初始化 ---
try:
    triton_client = grpcclient.InferenceServerClient(
        url=FLAGS.url, verbose=FLAGS.verbose
    )
except Exception as e:
    print("Context creation failed: " + str(e))
    sys.exit(1)

# --- 模型與輸出入設定 ---
model_name = "CWT_CNN"


input_name = "input__0"
output_name = "output__0"  

inputs = [grpcclient.InferInput(input_name, [1, 1, 368], "FP32")]
outputs = [grpcclient.InferRequestedOutput(output_name)]



# --- 非同步回呼函式 ---
def callback(user_data, result, error):
    """
    非同步推論的回呼函式，用來接收伺服器的回傳結果。
    """
    if error:
        user_data.append(error)
    else:
        user_data.append(result)


# --- 主要處理函式 ---
def CNN_processing(seg_list):
    """
    接收一個包含多個信號段的列表，並回傳每個信號段的預測標籤。
    """
    pred_list = []
    user_data = []

    # 確保 seg_list 是一個列表
    if not isinstance(seg_list, list):
        seg_list = [seg_list]

    for seg in seg_list:
        # 確保資料型態為 FP32
        seg = np.float32(seg)
        # 擴展維度以符合模型輸入 [batch_size, channels, length] -> [1, 1, 368]
        if seg.ndim == 1:
            seg = seg[None, None, :]
        elif seg.ndim == 2:
            seg = seg[None, :]

        # 將 NumPy 陣列設定為輸入資料
        inputs[0].set_data_from_numpy(seg)

        # 發送非同步推論請求
        triton_client.async_infer(
            model_name=model_name,
            inputs=inputs,
            callback=partial(callback, user_data),
            outputs=outputs,
            client_timeout=FLAGS.client_timeout,
        )

    # --- 等待並處理回傳結果 ---
    time_out = 10
    processed_requests = 0
    start_time = time.time()
    
    # 等待直到所有請求都收到回覆或超時
    while processed_requests < len(seg_list) and (time.time() - start_time) < time_out:
        if len(user_data) > processed_requests:
            result = user_data[processed_requests]
            
            # 檢查是否有錯誤
            if isinstance(result, InferenceServerException):
                print(result)
                pred_list.append("Error")
            else:

                output = result.as_numpy(output_name) # <--- 在這裡替換
                pred_index = np.argmax(output)
                pred_label = label[pred_index]
                pred_list.append(pred_label)
            
            processed_requests += 1
        else:
            time.sleep(0.001) # 短暫等待以避免 CPU 空轉

    if processed_requests < len(seg_list):
        print(f"Warning: Timed out. Only processed {processed_requests}/{len(seg_list)} requests.")

    return pred_list


# --- 主程式進入點 ---
if __name__ == "__main__":
    print("--- Running test case ---")
    
    # 建立 (1, 1, 368) 輸入的測試資料
    test_segment = np.random.randn(368).astype(np.float32)
    
    # 將測試資料放入列表中
    test_data_list = [test_segment]
    
    print(f"Input data shape: {test_segment.shape}")
    print(f"Sending request to Triton server at {FLAGS.url}...")
    
    # 呼叫函式進行推論
    predictions = CNN_processing(test_data_list)
    
    # 列印結果
    if predictions:
        print(f"\nPrediction result: {predictions[0]}")
    else:
        print("\nFailed to get a prediction.")
        
    print("--- Test case finished ---")