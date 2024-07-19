import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from statistics import mean
from scipy.stats import gaussian_kde
from scipy.optimize import curve_fit

def remove_edges(image, edge_threshold):
    edges = cv2.Canny(image, 50, 200)
    edges = cv2.threshold(edges, edge_threshold, 255, cv2.THRESH_BINARY)[1]
    result = cv2.bitwise_and(image, cv2.bitwise_not(edges))
    return result
 
cap = cv2.VideoCapture('Re-Experiment[2]\Castor\part1.mp4')
cap.read()
fps = cap.get(cv2.CAP_PROP_FPS)

print("FPS:", fps)
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)
crop_size = 200
edge_removal_threshold = 200

fgmask_frames = []
prev_centroid = None
# prev_frame_time = None
frame_count=0
t=[]
v=[]
d=[]
time_difference = 0
# a=True

while True:
    ret, frame = cap.read()

    if not ret:
        break
    # while a:
    #   output_path = f"oneframe.jpg"
    #   cv2.imwrite(output_path, frame)
    #   a=False
    # frame = frame[0:600, 650:1920]
    fgmask = fgbg.apply(frame)

    fgmask_no_edges = remove_edges(fgmask, edge_removal_threshold)

    fgmask_frames.append(fgmask_no_edges)
    
    
    cv2.imshow('fg', fgmask_no_edges)

    # hsv_frame = cv2.cvtColor(fgmask_no_edges, cv2.COLOR_BGR2HSV)

    # lower_color = np.array([0, 100, 100])
    # upper_color = np.array([20, 255, 255])

    _, binary_mask = cv2.threshold(fgmask_no_edges, 50, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_contour = max(contours, key=cv2.contourArea, default=None)
    

    if largest_contour is not None:
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            cv2.circle(frame, (cx, cy), 5, (255, 255, 255), -1)

            
            if frame_count%6==0:
                if prev_centroid is not None: #and prev_frame_time is not None:

                    displacement_vector = np.array([cx, cy]) - np.array(prev_centroid)
                    time_difference += 6 / 50
                    displacement_magnitude = np.sqrt((displacement_vector[0] ** 2) + (displacement_vector[1] ** 2))
                    velocity = displacement_magnitude / (6 / 50)
                    t.append(time_difference)
                    d.append(displacement_magnitude)
                    v.append(velocity)

                    print("Velocity:", velocity)

                prev_centroid = np.array([cx, cy])
                # prev_frame_time = time.time()
        
        cv2.imshow('Velocity Estimation', frame)

        k = cv2.waitKey(30) & 0xff
        if k == 27: 
            break
    frame_count+=1
print(len(v))
v_new=[]
t_new=[]
d_new=[]
for i in range(2,len(v)):
    if v[i]>55 and v[i]<77:
        v_new.append(v[i]*0.00027022727272727273) #converting into m/s from pixels/s
        t_new.append(t[i])
        d_new.append(d[i]*0.00027022727272727273) #converting into m from pixels
print(v_new)
print(t_new)
print(d_new)
# plt.plot(t_new,v_new)
# plt.xlabel("time(s)")
# plt.ylabel("velocity(m/s)")
# plt.title("Velocity-Time graph")l
# plt.show()
# print(mean(v_new))

# Define a function to compute moving averages
def moving_average(data, window_size):
    cumsum = np.cumsum(data)
    cumsum[window_size:] = cumsum[window_size:] - cumsum[:-window_size]
    return cumsum[window_size - 1:] / window_size

# Smooth the velocity data using a moving average
window_size = 9  # Adjust this parameter to change the smoothness
smoothed_velocity = moving_average(v_new, window_size)

# Plotting velocity versus time
# plt.plot(t_new[window_size - 1:],  linestyle='-', label='Smoothed Velocity')
plt.plot(t_new, v_new) #[window_size-1:]
# plt.plot(t_new,v_new)

# Adding labels and title
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.title('Velocity v/s Time [Castor Oil, 1.98mm ball]')
plt.show()

## Curve Fitting
# Define the 2nd-degree polynomial function
def second_degree_poly(x, a, b, c):
    return a * x**2 + b * x + c
# Sample data points
x_data = np.array(t_new)
y_data = np.array(v_new)
# Perform curve fitting
popt, pcov = curve_fit(second_degree_poly, x_data, y_data)
# Retrieve coefficients
a_fit, b_fit, c_fit = popt
# Plot original data points
plt.scatter(x_data, y_data, label='Data')
# Plot fitted curve
x_fit = np.linspace(min(x_data), max(x_data), 100)
y_fit = second_degree_poly(x_fit, a_fit, b_fit, c_fit)
plt.plot(x_fit, y_fit, 'r-', label='Fitted Curve')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('2nd Degree Polynomial Curve Fitting')
plt.legend()
plt.grid(True)
plt.show()

cap.release()
cv2.destroyAllWindows()