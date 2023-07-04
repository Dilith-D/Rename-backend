import io 
import base64
from PIL import Image
import cv2
import numpy as np

from rest_framework.decorators import api_view
from rest_framework.response import Response
from .serializers import ImageSerializer

from rest_framework.parsers import MultiPartParser
from rest_framework.decorators import api_view, parser_classes



# Create your views here.
from django.http import JsonResponse

@api_view(['POST'])
# Use MultiPartParser to handle file uploads
@parser_classes([MultiPartParser])

def object_detection_view(request):
    content_type = request.content_type
    print("Content-Type:", content_type)

    if request.method == 'POST':
        serializer = ImageSerializer(data=request.data)
        
        if serializer.is_valid():
            # Load the YOLO model
            net = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')
            layer_names = net.getLayerNames()
            output_layers = []
            for i in net.getUnconnectedOutLayers():
                output_layers.append(layer_names[i.item() - 1])

            # Load the classes
            classes = []
            with open('coco.names.txt','r') as f:
                classes = [line.strip() for line in f.readlines()]

            # Load the image
            uploaded_image = serializer.validated_data['image']
            image = Image.open(uploaded_image).convert('RGB')
            image_array = np.array(image)
            height, width, channels = image_array.shape
            blob = cv2.dnn.blobFromImage(image_array, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

            # Set the input to the network
            net.setInput(blob)
            outs = net.forward(output_layers)

            # Initialize lists to store object information
            class_ids = []
            confidences = []
            boxes = []
            tags =[]

        
            # Iterate over the outputs
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]

                    if confidence > 0.5:
                        # Retrieve bounding box coordinates
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        # Store object information
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
                        tags.append(classes[class_id])

            # Apply non-maximum suppression
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

            # Draw bounding boxes and labels on the image
            output_image = image_array
             

             # Define a list of specific colors
            colors = [(250, 56, 218), (0, 255, 255), (57, 255, 20), (255, 255, 0), (255, 83, 0)]

            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    label = classes[class_ids[i]]
                    confidence = confidences[i]
                    color = colors[i % len(colors)]

                    cv2.rectangle(output_image, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(output_image, f'{label}: {confidence:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Filter out the duplicate tags
            filtered_tags = []
            for i, tag in enumerate(tags):
                if i in indexes:
                    filtered_tags.append(tag)

            # Create a list of unique tags
            unique_tags = list(set(filtered_tags))

            # Convert numpy array to PIL Image
            output_image_pil = Image.fromarray(np.uint8(output_image))

            # Create a BytesIO object to hold the image data
            output_image_data = io.BytesIO()

            # #compress the image 
            # compressed_image_data= io.BytesIO()
            # output_image_pil.save(compressed_image_data,format='JPEG',quality=80)
            # compressed_image_data.seek(0)

            # Save the PIL Image
            output_image_pil.save(output_image_data, format='JPEG')

            # Convert the output image data into base64
            output_image_base64 = base64.b64encode(output_image_data.getvalue()).decode('utf-8')


            #prepare the response data
            response_data ={
                'tags' : unique_tags,
                'output_image':output_image_base64,
                
                }

            #log
            print(response_data)
            
            
            
            return Response(response_data)
        else:
            return Response(serializer.errors, status=400)

        