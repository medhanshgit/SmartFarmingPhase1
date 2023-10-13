import cv2

image_paths = ['1.jpg', '2.jpg', '3.jpg']
imgs = []

for path in image_paths:
    img = cv2.imread(path)
    # Optional: Resize the image if needed
    # You can specify the new dimensions like this:
    # img = cv2.resize(img, (new_width, new_height))
    imgs.append(img)

# Showing the original pictures
cv2.imshow('1', imgs[0])
cv2.imshow('2', imgs[1])
cv2.imshow('3', imgs[2])

stitcher = cv2.createStitcher() if hasattr(cv2, 'createStitcher') else cv2.Stitcher_create()

(status, output) = stitcher.stitch(imgs)

if status == cv2.Stitcher_OK:
    print('Your Panorama is ready!!!')
    # Final output
    cv2.imshow('final result', output)
else:
    print("Stitching ain't successful")

cv2.waitKey(0)
