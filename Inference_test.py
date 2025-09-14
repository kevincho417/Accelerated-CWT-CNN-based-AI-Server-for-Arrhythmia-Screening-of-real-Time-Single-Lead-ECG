import numpy as np
from CWT112 import CWT_112
from CNN import CNN_processing as CNN_processing1
# from CWT_CNN import CNN_processing as CNN_processing2
from onnx_CNN import CNN_processing as CNN_processing3
from static_CNN import CNN_processing as CNN_processing2
# from RT_CNN import CNN_processing as CNN_processing3
from trt_CNN import CNN_processing as CNN_processing4
import time
# from segment import segment
from segment_from_txt import segment 






if __name__=="__main__":

    # file_path="./1001.txt"
    # start_time=time.time()
    # seg_list=segment(file_path)
    # end_time=time.time()
    # print("Segment time:",(end_time-start_time))
    # print(len(seg_list))
    number=20000
    print("Random input:",number," array as input:")
    seg_list=[]
    for i in range(number):
        seg_list.append(np.random.rand(368,).astype(np.float32))
    start_time=time.time()
    CWT_list = CWT_112(seg_list)
    pred_list1=CNN_processing1(CWT_list)
    end_time=time.time()
    print("Original time: ",(end_time-start_time))
    # print("Original output first one:"pred_list1)
    # start_time=time.time()
    # seg_list = segment(file_path)
    # end_time=time.time()
    # print("Segmention time: ",(end_time-start_time))
    # print("Loading...")
    # start_time=time.time()
    # seg_list=[np.random.rand(368,).astype(np.float32)]
    # CWT_list = CWT(seg_list)
    # end_time=time.time()
    # print("CWT time: ",(end_time-start_time))
    print("Loading...")
    start_time=time.time()
    predict_list = CNN_processing2(seg_list)
    end_time=time.time()
    print("Concat CNN_CWT model (pytorch backend) time: ",(end_time-start_time))
    print("Loading...")
    start_time=time.time()
    predict_list = CNN_processing3(seg_list)
    end_time=time.time()
    print("onnx_model (pytorch backend) time: ",(end_time-start_time))
    print("Loading...")
    start_time=time.time()
    pred_list=CNN_processing4(seg_list)
    end_time=time.time()
    print("TensorRT with Triton server time: ",(end_time-start_time))
    # print("CNN time: ",(end_time-start_time))



    # # Create the main GUI window
    # root = tk.Tk()
    # root.title("ECG Segment Viewer")

    # # Create GUI elements
    # segment_label = tk.Label(root, text="ECG Segment 1")
    # segment_label.pack()
    # cwt_label = tk.Label(root, text="label")
    # cwt_label.pack()

    # # Create Matplotlib figures and canvases for displaying ECG and CWT
    # ecg_fig, ecg_plot = plt.subplots(figsize=(6, 4))
    # cwt_fig, cwt_plot = plt.subplots(figsize=(6, 4))
    # ecg_canvas = FigureCanvasTkAgg(ecg_fig, master=root)
    # cwt_canvas = FigureCanvasTkAgg(cwt_fig, master=root)
    # ecg_canvas.get_tk_widget().pack()
    # cwt_canvas.get_tk_widget().pack()

    # # Create previous and next buttons
    # prev_button = tk.Button(root, text="Previous", command=previous_segment)
    # prev_button.pack()
    # next_button = tk.Button(root, text="Next", command=next_segment)
    # next_button.pack()

    # # Create a button to open the file dialog
    # open_button = tk.Button(root, text="Open File", command=open_file_dialog)
    # open_button.pack()

    # root.mainloop()
