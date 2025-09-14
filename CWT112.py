from tfa_morlet_112m import tfa_morlet
# from wavelets_pytorch.transform import WaveletTransformTorch 
import numpy as np
# dt = 0.004         # sampling frequency
# dj = 0.125       # scale distribution parameter
# batch_size = 1
img_select = np.linspace(0, 367, 112, dtype=int)
# wa_torch = WaveletTransformTorch(dt, dj, cuda=True)

def CWT_112(seg_list):
    CWT_list = []
    for i in range(len(seg_list) ):
        seg = seg_list[i]
        B = np.array(seg)
        s_rate = 512

        aa = B - B.mean()
        anorm = aa / aa.max()
        DATA = -anorm
        sig = DATA

        fmin = 4
        fmax = 40
        img = np.array(tfa_morlet(sig, s_rate, fmin, fmax, 36/112))

        # 確保心電圖至少包含112個資料點
        
        # img = wa_torch.cwt(sig)
        img = img[::-1, img_select]
        img = img / np.max(img)
        CWT_picture = img
        CWT_list.append(CWT_picture)
    return CWT_list
