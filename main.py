# Smartcow - https://www.smartcow.ai/en/

from NellieJay.NellieJay import NellieJay
import torch
if __name__ == "__main__":
    # Initialise a NellieJay Object
    my_smart_cow = NellieJay(height=720, width=1080, max_cows=6, delay=1)
    path = '/home/aparecido/Desktop/Take_Home_Computer_Vision/best.pt'
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=path, force_reload=True) 
#     model_ = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    model.conf = 0.5
    while True:
        # Generate a frame and return the number of cows placed on the frame
        frame, cow_count = my_smart_cow.generate_frame()

        ############################################ 
        ## Write your code here to count the cows ##
        ############################################
        # TODO: Implement your cow counting solution here return as answer
        # Your goal is to minimise the Mean Squared Error (MSE) and Root Mean Squared Error (RMSE) values
#         model = torch.hub.load('ultralytics/yolov5', 'custom', path=path) 
        results = model(frame)  # inference

        results.imgs # array of original images (as np array) passed to model for inference
        results.render()  # updates results.imgs with boxes and labels
#         for im in results.imgs:
#         #     buffered = BytesIO()
#             im_base64 = Image.fromarray(im)
        res = (results.pandas().xyxy[0].confidence)
        answer = len(res) # Your cow count should go here
        ############################################

        # Calculate and print the error to the screen and console 
        frame = my_smart_cow.print_scores(results.imgs[0], cow_count, answer)
        # Show the frame on screen     
        my_smart_cow.show_frame(frame) 