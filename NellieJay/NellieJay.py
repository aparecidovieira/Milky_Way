# Smartcow - https://www.smartcow.ai/en/

import cv2
import time, sys
import imutils
import numpy as np
sys.path.append('../')


from NellieJay.CowGenerator import CowGenerator

# The NellieJay Class
# Generates a frame with a random amount of randomly placed cows
class NellieJay:

    def __init__(self, width=1920, height=1080, max_cows=5, delay=0.5):
        # Set the frame dimensions
        self.width = width
        self.height = height

        # Delay between frames
        self.delay = delay

        # Max amount of cows that can be placed on the frame
        self.max_cows = max_cows
        # Max size that the cows can be resized to
        self.max_cow_size = 150
    
        # Load and resize the day background image
        self.day_background = cv2.imread("NellieJay/resources/day_background.jpg")
        # print(self.day_background.shape)
        self.day_background = cv2.resize(self.day_background, (self.width, self.height), interpolation=cv2.INTER_AREA)
        self.day_background = cv2.cvtColor(self.day_background, cv2.COLOR_BGR2BGRA)
        # Add the SmartCow Logo to the background
        self.day_background = self.add_logo(self.day_background)

        # Load and resize the night background image
        self.night_background = cv2.imread("NellieJay/resources/night_background.png")
        self.night_background = cv2.resize(self.night_background, (self.width, self.height), interpolation=cv2.INTER_AREA)
        self.night_background = cv2.cvtColor(self.night_background, cv2.COLOR_BGR2BGRA)
        # Add the SmartCow Logo to the background
        self.night_background = self.add_logo(self.night_background)

        # Load the UFO image and resize it
        self.ufo = cv2.imread("NellieJay/resources/ufo.png", -1)
        self.ufo = cv2.resize(self.ufo, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
        
        # Stores the current frame number
        self.frame_count = 0
        # Stores the total prediction error
        self.sum_error = 0
        
        # Cow Augmentation Variables
        # Max angle that a cow can be rotated [-self.max_angle:self.max_angle]
        self.max_angle = 10
        # Scale range that a cow can be scaled with
        self.scales = np.arange(0.8,1.3,0.1)
        
        # Initialise the cow generator
        self.cow_generator = CowGenerator()

    # Overlays a given image on a given frame and blends it according to the specified alpha values
    def overlay_image(self, frame, image, pos=(0,0), bg_alpha=1.0, im_alpha=1.0):
        # Get shape of image to place
        im_height, im_width = image.shape[:2]

        # Get the ROI from the background with the same shape as the image to place
        bg_crop = frame[pos[1]:pos[1]+im_height, pos[0]:pos[0]+im_width]
        # Blend the the ROI with the image and place it back on the background
        frame[pos[1]:pos[1]+im_height, pos[0]:pos[0]+im_width] = cv2.addWeighted(bg_crop, bg_alpha, image, im_alpha, 0)

        return frame

    # Draw the SmartCow logo on the background
    def add_logo(self, frame, size=125):      
        # Load the logo  
        logo = cv2.imread("NellieJay/resources/logo.png", cv2.IMREAD_UNCHANGED)
    
        # Resize it
        ratio = size/1920
        logo = cv2.resize(logo, (int(self.width*ratio), int(self.width*ratio)), interpolation=cv2.INTER_AREA)

        # Get the placement coordinates
        logo_height, logo_width = logo.shape[:2]

        logo_start_x = frame.shape[1] - int(logo_width * 1.2)
        logo_start_y = frame.shape[0] - int(logo_height * 1.2)
        
        # Overlay it on the background
        return self.overlay_image(frame, logo, pos=(logo_start_x, logo_start_y), im_alpha=0.8)

    # Display a frame on screen
    def show_frame(self, frame):
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) == ord("q"):
            raise KeyboardInterrupt
        # Delay before generating a new frame
        time.sleep(self.delay)

    # Generates a frame by randomly placing cows and a ufo on it
    def generate_frame(self):
        # Increment the frame count
        self.frame_count += 1

        # Generate a set of coordinates where the cows can be placed
        cx = np.arange(self.max_cow_size, self.width-self.max_cow_size, self.max_cow_size + (self.max_cow_size//2))
        cy = np.arange((self.height//2), self.height-self.max_cow_size, self.max_cow_size//2) + 30
        np.random.shuffle(cx)
        np.random.shuffle(cy)

        # Generate all possible permutations of the coordinates 
        positions = []
        for i in cx:
            for j in cy:
                positions.append((i, j))
        # Shuffle the coordinates
        np.random.shuffle(positions)

        # If the requested amount of cows exceeds the number of available coordinates
        # set it to the number of available coordinates
        if self.max_cows > len(positions):
            self.max_cows = len(positions)

        # Generate a random number of cows to draw between 1 and max_cows
        cow_count = np.random.randint(1, self.max_cows+1)

        # Choose a background
        if np.random.uniform() >= 0.5:
            # Make a copy of the background template
            tmp_frame = self.day_background.copy()
        else:
            tmp_frame = self.night_background.copy()

            # 20% chance for a UFO to appear at night
            if np.random.uniform() >= 0.8:
                # Generate random coordinates on the top half of the frame
                ufo_x = np.random.randint(0, self.width-self.ufo.shape[1])
                ufo_y = np.random.randint(0, (self.height//2)-self.ufo.shape[0])
                # Overlay the UFO on the frame
                tmp_frame = self.overlay_image(tmp_frame, self.ufo, (ufo_x, ufo_y), bg_alpha=1, im_alpha=1)

        # Generate and place the cows
        for i in range(cow_count):
            # Get a randomly colored cow
            tmp_cow = self.cow_generator.get_cow()
            
            # 50% chance to flip the cow horizontally
            if np.random.uniform() > 0.5:
                tmp_cow = cv2.flip(tmp_cow, 1)
            
            # Roatate the cow 
            tmp_cow = imutils.rotate(tmp_cow, np.random.randint(-self.max_angle,self.max_angle+1))

            # Rescale the cow
            tmp_scale = self.scales[np.random.randint(len(self.scales))] 
            tmp_cow = cv2.resize(tmp_cow, (0, 0), fx=tmp_scale, fy=tmp_scale, interpolation=cv2.INTER_AREA)

            # Overlay the cow on the frame
            tmp_frame = self.overlay_image(tmp_frame, tmp_cow, positions[i], bg_alpha=1, im_alpha=1)

            # Make the cow opaque
            pos = positions[i]
            cow_height, cow_width = tmp_cow.shape[:2]
            bg_crop = tmp_frame[pos[1]:pos[1]+cow_height, pos[0]:pos[0]+cow_width]
            bg_crop[(tmp_cow[:,:,3] == 255), :3] = tmp_cow[(tmp_cow[:,:,3] == 255), :3]
            tmp_frame[pos[1]:pos[1]+cow_height, pos[0]:pos[0]+cow_width] = bg_crop
        
        # Return the frame and cow count
        return tmp_frame, cow_count

    # Calculate the error for the current frame
    def get_error(self, y_true, y_pred):
        return (y_true - y_pred)**2

    # Calculates and prints the scores to screen and console
    def print_scores(self, frame, y_true, y_pred):
        # Sum the current error with the global error
        self.sum_error += self.get_error(y_true, y_pred)

        # Calculate the MSE and RMSE
        mse = ((1/self.frame_count) * self.sum_error)
        rmse = mse**0.5
        
        print("Frame: {}\t\t Actual: {} \t\tPrediction: {} \t\tMSE: {}\t\tRMSE: {}".format(self.frame_count, y_true, y_pred, round(mse, 4), round(rmse, 4)))

        # Draw the text on the frame and return the frame
        text = "Actual: {}, Prediction: {}, MSE: {}, RMSE: {}".format(y_true, y_pred, round(mse, 4), round(rmse, 4))
        return cv2.putText(frame, str(text), (10, self.height-25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 232, 255), 2, cv2.LINE_AA)