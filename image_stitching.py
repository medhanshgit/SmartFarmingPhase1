import cv2
import os

# Specify the folder path where your images are located
folder_path = '15m'

# Get a list of all image files in the folder
image_paths = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path) if filename.endswith(('.JPG', '.jpeg', '.png'))]

# Check if there are any images in the folder
if not image_paths:
    print("No image files found in the folder.")
else:
    imgs = []

    # Set smaller desired width and height for resizing
    new_width = 400  # Adjust this as needed
    new_height = 300  # Adjust this as needed

    for path in image_paths:
        img = cv2.imread(path)
        
        # Resize the image to the desired dimensions
        img = cv2.resize(img, (new_width, new_height))
        
        imgs.append(img)

    # Showing the original pictures
    for i, img in enumerate(imgs):
        cv2.imshow(f'Image {i + 1}', img)

    stitcher = cv2.createStitcher() if hasattr(cv2, 'createStitcher') else cv2.Stitcher_create()

    (status, output) = stitcher.stitch(imgs)

    if status == cv2.Stitcher_OK:
        print('Your Panorama is ready!!!')
        # Final output
        cv2.imshow('Final Result', output)
    else:
        print("Stitching ain't successful")

    cv2.waitKey(0)
    cv2.destroyAllWindows()
