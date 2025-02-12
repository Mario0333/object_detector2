import cv2
import numpy as np

# Load YOLO model
net = cv2.dnn.readNetFromDarknet("yolov3.cfg", "yolov3.weights")

# Load class labels
with open("coco.names", "r") as f:
    labels = f.read().strip().split("\n")

# Define colors for each class
colors = [tuple(int(x) for x in np.random.randint(0, 255, 3)) for _ in labels]

# Start capturing video
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Prepare the frame
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    
    # Get output layer names and forward pass
    layer_names = net.getUnconnectedOutLayersNames()
    detections = net.forward(layer_names)

    # lists for bounding boxes, confidences, and class IDs
    boxes = []
    confidences = []
    class_ids = []

    h, w = frame.shape[:2]

    # Process detections
    for output in detections:
        for detection in output:
            scores = detection[5:]
            class_id = int(np.argmax(scores))
            confidence = scores[class_id]

            if confidence > 0.3:  # Adjust confidence threshold if needed
                box = detection[0:4] * np.array([w, h, w, h])
                center_x, center_y, width, height = box.astype("int")

                # Calculate top-left corner of the bounding box
                x = int(center_x - (width / 2))
                y = int(center_y - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.4)

    if len(indices) > 0:
        for i in indices:
            
            x, y, w, h = boxes[i]
            color = colors[class_ids[i]]
            label = f"{labels[class_ids[i]]}: {confidences[i]:.2f}"

            # Draw the bounding box and label
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the frame 
    cv2.imshow("Object Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
