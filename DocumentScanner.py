import cv2
import numpy as np
import math


# we define this function to find contours and warp the image and get a 4 point transform.
# this function only takes one argument which is the image you need to scan.
def scan(img):
    global final_img
    img_copy = img.copy()
    img_canny = cv2.Canny(img_copy, 70, 140)
    # cv2.RETR_EXTERNAL returns the outer most contours.
    contours, hierarchy = cv2.findContours(img_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        # We find the area of the contours.
        area = cv2.contourArea(contour)
        # if area > 30000:
        # print(area)
        # if the area of the contours is less than what we want then we don't draw the contours on out desired image.
        # this is useful if the cv2.findContours function detects noise in the image.
        if area > 30000:
            # print(area)
            # -1 argument means we'll draw ON img2.
            cv2.drawContours(img_copy, contour, -1, (255, 0, 0), 2)
            # we find the perimeters of the shapes.
            # True means the shapes or contours are closed
            perimeter = cv2.arcLength(contour, True)
            # cv2.approxPolyDP returns a matrix.
            # The number of rows in the matrix means the number of vertices or corner points of the shape.
            # The second parameter is related to accuracy. 0.02 * perimeter means 0.2% of the arc length (perimeter).
            # True means the contour is closed
            v1 = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            # print(v1)
            # printing vertices would give us the matrix.
            # len(vertices) gives us the number of vertices or corner points.
            # print(len(v1))
            object_corners = len(v1)
            # cv2.boundingRect gives us the x, y, width and height of the box bounding the objects.
            x, y, w, h = cv2.boundingRect(v1)
            if object_corners == 3:
                object_type = "Triangle"
            elif object_corners == 4:
                aspect_ratio = w / float(h)
                if 0.95 < aspect_ratio < 1.05:
                    object_type = "Square"
                else:
                    object_type = "Rectangle"
            elif object_corners > 4:
                object_type = "Circle or something"
            else:
                object_type = "Dunno"
            # We draw rectangles using cv2.rectangle and coordinates x, y, w, h
            cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # we put text near the middle.
            cv2.putText(img_copy, object_type, (x + (w // 2) - 10, y + (h // 2) - 10), cv2.FONT_ITALIC, 0.4,
                        (0, 0, 0), 1)
            # v is a 3D matrix with all the vertices of our object
            # we need to convert it to a 2D matrix.
            # so we create v2
            v2 = list()
            for temp in v1:
                for elem in temp:
                    v2.append(elem)
            print(v2)
            # The order of elements (points) in v2 is not always what's expected.
            # So we have to reorder them to our liking
            # a, b store the width and the height of the original image
            shape = img.shape
            a, b = shape[1], shape[0]
            # [0, 0] should correspond with the top left point of our object.
            # Because the top left point will be the new origin.
            # Similarly [0, b] ([0, height]) should correspond with bottom left point of our object etc.
            # The points stored in v2 don't necessarily line up with og_vertices. So we have to reorder it.
            # The point in v2 closest to [0, 0] will be first in the new matrix 'vertices'.
            # similarly point closest to [0, b] will be second in the new matrix 'vertices' so that they line up etc.
            og_vertices = [[0, 0], [0, b], [a, b], [a, 0]]
            vertices = list()
            for p in og_vertices:
                indexes = list()
                for q in v2:
                    r = math.pow(q[0] - p[0], 2)  # Finding distance b/w two points
                    s = math.pow(q[1] - p[1], 2)  # Finding distance b/w two points
                    indexes.append(math.sqrt(r+s))  # Finding distance b/w two points
                minimum = min(indexes)
                index = indexes.index(minimum)
                vertices.append(v2[index])
                print(vertices)

            points1 = np.float32(vertices)
            points2 = np.float32([[0, 0], [0, height], [width, height], [width, 0]])
            transformation_matrix = cv2.getPerspectiveTransform(points1, points2)
            final_img = cv2.warpPerspective(img, transformation_matrix, (width, height))
    return [img_copy, final_img]


if __name__ == "__main__":
    img = cv2.imread('Images/sample1.jpg')
    img = cv2.resize(img, (960, 1280))
    # width of the cropped image
    width = int(input("What do you want the width of the image to be? - "))
    # height of the cropped image
    height = int(input("What do you want the height of the image to be? - "))
    # print(img.shape)
    result = scan(img)
    warped_img_grey = cv2.cvtColor(result[1].copy(), cv2.COLOR_BGR2GRAY)
    thresh1 = cv2.adaptiveThreshold(warped_img_grey, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 15)
    thresh2 = cv2.adaptiveThreshold(warped_img_grey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 15)
    cv2.imshow("Page detection", result[0])
    cv2.imshow("Warped image", result[1])
    cv2.imshow("Warped image grey", warped_img_grey)
    cv2.imshow("Adaptive threshold", thresh1)
    cv2.waitKey(0)
