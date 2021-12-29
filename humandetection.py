
# face detection with mtcnn on a photograph
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
from mtcnn.mtcnn import MTCNN
import os
 
# draw an image with detected objects
def draw_image_with_boxes(filename, result_list):
	# load the image
	data = pyplot.imread(filename)
	# plot the image
	pyplot.imshow(data)
	# get the context for drawing boxes
	ax = pyplot.gca()
	# plot each box
	for result in result_list:
		# get coordinates
		x, y, width, height = result['box']
		# create the shape
		rect = Rectangle((x, y), width, height, fill=False, color='red')
		# draw the box
		ax.add_patch(rect)
		# draw the dots
		for key, value in result['keypoints'].items():
			# create and draw dot
			dot = Circle(value, radius=2, color='red')
			ax.add_patch(dot)
	# show the plot
	pyplot.show()

def draw_faces(filename, result_list):
	# load the image
	data = pyplot.imread(filename)
	# plot each face as a subplot
	for i in range(len(result_list)):
		# get coordinates
		x1, y1, width, height = result_list[i]['box']
		x2, y2 = x1 + width, y1 + height
		# define subplot
		pyplot.subplot(1, len(result_list), i+1)
		pyplot.axis('off')
		# plot face
		pyplot.imshow(data[y1:y2, x1:x2])
	# show the plot
	pyplot.show()

def main():
    try:
        filename = input("Enter image name: ")
        # load image from file
        pixels = pyplot.imread(filename)
        # create the detector, using default weights
        detector = MTCNN()
        # detect faces in the image
        faces = detector.detect_faces(pixels)
        # display faces on the original image
        draw_image_with_boxes(filename, faces)
        draw_faces(filename, faces)
    except FileNotFoundError:
        print("Enter valid image name.")
        files = [_ for _ in os.listdir(os.getcwd()) if _.endswith(".jpg")]
        print("Valid examples are: \n")
        for file in files:
            print("-" + file + "\n")
        main()

main()