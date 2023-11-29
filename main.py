import cv2
import mediapipe as mp
import numpy as np
import random
import time

# Initialize Mediapipe hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Set up video capture
cap = cv2.VideoCapture(0)

# Set up game parameters
radius = 20
paddle_radius = radius
target_radius = 30
puck_color = (255, 255, 255)
paddle_color = (0, 0, 255)
score_color = (255, 255, 255)
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_thickness = 2

# Game score and time limit
score = 0
time_limit = 90  # in seconds
start_time = time.time()

# Load target image
target_image = cv2.imread('/home/jaypopos/Documents/code/cv_assignment/ERC-Assignment-4/target.png', cv2.IMREAD_UNCHANGED)
target_image = cv2.resize(target_image, (2 * target_radius, 2 * target_radius))

# Initialize puck position and velocity
puck_position = np.array([320, 240])
puck_velocity = np.array([5, 5])

# Initialize paddle position
paddle_position = np.array([400, 240])

# Initialize target positions
num_targets = 5
targets = [np.array([random.randint(50, 590), random.randint(50, 430)]) for _ in range(num_targets)]
targets_hit = [False] * num_targets


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]

        # Extract x, y coordinates of the index finger
        index_finger_x = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * frame.shape[1])
        index_finger_y = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * frame.shape[0])

        # Move the circular paddle based on index finger movement
        paddle_position[0] += (index_finger_x - paddle_position[0]) / 5
        paddle_position[1] += (index_finger_y - paddle_position[1]) / 5

    # Update puck position based on velocity
    puck_position = (puck_position + puck_velocity)

    # Check for collisions with walls
    if puck_position[0] - radius < 0 or puck_position[0] + radius > frame.shape[1]:
        puck_velocity[0] *= -1
    if puck_position[1] - radius < 0 or puck_position[1] + radius > frame.shape[0]:
        puck_velocity[1] *= -1

    # Check for collision with the circular paddle
    distance = np.linalg.norm(paddle_position - puck_position)
    if distance < 2 * radius:
        # Collision occurred, reverse puck's vertical velocity
        puck_velocity[1] *= -1

    # Check for collision with the targets
    for i in range(num_targets):
        if not targets_hit[i]:
            target_distance = np.linalg.norm(targets[i] - puck_position)
            if target_distance < target_radius + radius:
                # Player scores a point, increase puck's velocity, and mark the target as hit
                score += 1
                puck_velocity = (puck_velocity*1.2)
                targets_hit[i] = True

    # Draw the puck, circular paddle, and targets on the frame
    for i in range(num_targets):
        if not targets_hit[i]:
            x, y = int(targets[i][0]) - target_radius, int(targets[i][1]) - target_radius
            alpha_s = target_image[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s
            for c in range(0, 3):
                frame[y:y + 2 * target_radius, x:x + 2 * target_radius, c] = (
                        alpha_s * target_image[:, :, c] + alpha_l * frame[y:y + 2 * target_radius, x:x + 2 * target_radius, c])

    cv2.circle(frame, tuple(puck_position.astype(int)), radius, puck_color, -1)
    cv2.circle(frame, tuple(paddle_position.astype(int)), paddle_radius, paddle_color, -1)

    # Display the score
    cv2.putText(frame, f"Score: {score}", (10, 30), font, font_scale, score_color, font_thickness, cv2.LINE_AA)

    # Display the time remaining
    elapsed_time = int(time.time() - start_time)
    remaining_time = max(0, time_limit - elapsed_time)
    cv2.putText(frame, f"Time: {remaining_time}s", (10, 70), font, font_scale, score_color, font_thickness, cv2.LINE_AA)

    # Display game over message if time limit is reached or all targets are hit
    if remaining_time == 0 or all(targets_hit):
        game_over_message = "Game Over! Your Final Score: " + str(score)
        if all(targets_hit):
            game_over_message = "Congratulations! " + game_over_message
        cv2.putText(frame, game_over_message, (50, frame.shape[0] // 2), font, font_scale, score_color, font_thickness,
                    cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Hand-Controlled Game', frame)

    # Exit the game if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
