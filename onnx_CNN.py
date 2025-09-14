import onnx
import torch
import onnxruntime
import numpy as np
onnx_model = onnx.load("model.onnx")
onnx.checker.check_model(onnx_model)


ort_session = onnxruntime.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])


# compute ONNX Runtime output prediction


def CNN_processing(seg_list):
    pred_list=[]
    for i, seg in enumerate(seg_list):

        # 創建 PILLOW 
        # image = Image.fromarray((image * 255).astype(np.uint8))
        # input_data = transform(image).unsqueeze(0).to(device)
        seg = seg[None,None,:]
        seg=np.float32(seg)
        num_examples  = seg.shape[0]
        signal_length = seg.shape[-1]
        # 進行推理
        ort_inputs = {ort_session.get_inputs()[0].name: seg}
        output = ort_session.run(None, ort_inputs)
        
        pred = np.argmax(output)

        # 列印結果
        a=np.array(['F', 'N', 'Q','V'])
        i=pred.item()
        pred_list.append(a[i])
    return pred_list
