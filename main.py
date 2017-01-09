from colors import colorsDescriptors
from tafuta import tafutA
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--index", required = True,
                help="path to place where features are to be stored")
ap.add_argument("-q", "--query", required =True,
                help="path to the images that are to be queried")
ap.add_argument("-r", "--result-path", required = True,
                help="path to the image result")
args = vars(ap.parse_args())

description = colorsDescriptors((8, 12, 3))
query = cv2.imread(args["query"])
features = description.explain(query)
search = tafutA(args["index"])
results = search.search(features)

cv2.imshow("Inline", query)

for (score, resultsId) in results:
    result = cv2.imread(args["result_path"] + "/" + resultsId)
    cv2.imshow("matokeo", result)
    cv2.waitKey(0)