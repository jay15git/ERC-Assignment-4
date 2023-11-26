import cv2
import mediapipe as mp
import random
import time
import numpy as np

# Initialize MediaPipe Hand module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Initialize screen dimensions
screen_width, screen_height = 800, 600

# Initialize puck parameters
puck_radius = 20
puck_pos = [screen_width // 2, screen_height // 2]
puck_velocity = [5, 5]

# Initialize paddle parameters
paddle_width, paddle_height = 100, 20
paddle_pos = [screen_width // 2 - paddle_width // 2, screen_height - 2 * paddle_height]

# Initialize game variables
score = 0
time_limit = 60  # in seconds
start_time = time.time()

# Initialize targets
num_targets = 5
targets = [(random.randint(50, screen_width - 50), random.randint(50, screen_height - 50)) for _ in range(num_targets)]

# Initialize OpenCV window
cv2.namedWindow("Virtual Air Hockey", cv2.WINDOW_NORMAL)

# Function to check collision between two circles
def is_collision_circle(circle1, circle2):
    distance = np.linalg.norm(np.array(circle1) - np.array(circle2))
    return distance < circle1[2] + circle2[2]

# Function to initialize targets
def initialize_targets():
    return [(random.randint(50, screen_width - 50), random.randint(50, screen_height - 50)) for _ in range(num_targets)]

while True:
    # Capture video from the webcam
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()

    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform hand tracking using Mediapipe
    results = hands.process(rgb_frame)

    # Update paddle position based on hand tracking
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        paddle_pos[0] = int(index_finger_tip.x * screen_width)
        paddle_pos[1] = int(index_finger_tip.y * screen_height)

    # Update puck position
    puck_pos[0] += puck_velocity[0]
    puck_pos[1] += puck_velocity[1]

    # Check for collisions with walls
    if puck_pos[0] - puck_radius < 0 or puck_pos[0] + puck_radius > screen_width:
        puck_velocity[0] = -puck_velocity[0]
    if puck_pos[1] - puck_radius < 0 or puck_pos[1] + puck_radius > screen_height:
        puck_velocity[1] = -puck_velocity[1]

    # Check for collision with paddle
    if is_collision_circle((puck_pos[0], puck_pos[1], puck_radius),
                            (paddle_pos[0] + paddle_width // 2, paddle_pos[1] + paddle_height // 2, puck_radius)):
        puck_velocity[1] = -puck_velocity[1]

    # Check for collision with targets
    for i, target in enumerate(targets):
        if is_collision_circle((puck_pos[0], puck_pos[1], puck_radius),
                                (target[0], target[1], 20)):  # assuming target radius is 20
            score += 1
            targets.pop(i)
            targets += initialize_targets()
            puck_velocity[0] *= 1.2
            puck_velocity[1] *= 1.2

    # Display the game elements on the screen
    cv2.circle(frame, (puck_pos[0], puck_pos[1]), puck_radius, (255, 255, 255), -1)
    cv2.rectangle(frame, (paddle_pos[0], paddle_pos[1]),
                  (paddle_pos[0] + paddle_width, paddle_pos[1] + paddle_height), (0, 255, 0), -1)
    for target in targets:
        cv2.circle(frame, (target[0], target[1]), 20, (0, 0, 255), -1)

    # Display the score and timer on the screen
    elapsed_time = time.time() - start_time
    remaining_time = max(0, int(time_limit - elapsed_time))
    cv2.putText(frame, f"Score: {score}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Time: {remaining_time}s", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Check for game over conditions
    if not targets or elapsed_time > time_limit:
        # Display game over message with the final score
        cv2.putText(frame, f"Game Over! Final Score: {score}", (screen_width // 4, screen_height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
        cv2.imshow("Virtual Air Hockey", frame)
        cv2.waitKey(3000)  # Display the game over message for 3 seconds
        break

    # Show the frame
    cv2.imshow("Virtual Air Hockey", frame)

    # Check for user input to quit the game
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
