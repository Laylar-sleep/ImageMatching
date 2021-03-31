import cv2
import numpy as np
from matplotlib import pyplot



def SIFT(queryPath, imagePath):
    query = cv2.imread(queryPath + ".jpg", cv2.IMREAD_GRAYSCALE)
    search = cv2.imread(imagePath + ".jpg", cv2.IMREAD_GRAYSCALE)

    box1 = open(queryPath + ".txt").read().split()
    x1 = int(box1[0])
    y1 = int(box1[1])
    w1 = int(box1[2])
    h1 = int(box1[3])

    query = query[y1:y1+h1, x1:x1+w1]

    if int(str(imagePath).split('Images/')[1]) <=2000:
        box2 = open(imagePath + ".txt").read().split()
        x2 = int(box2[0])
        y2 = int(box2[1])
        w2 = int(box2[2])
        h2 = int(box2[3])

        search = search[y2:y2 + h2, x2:x2 + w2]

        sift = cv2.xfeatures2d.SIFT_create()
        kp_q, des_q = sift.detectAndCompute(query, None)
        kp_s, des_s = sift.detectAndCompute(search, None)

        # Feature Matching
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des_q, des_s, k=2)

        # Filter out good matches
        matchesMask = [[0,0] for i in range(len(matches))]

        #ratio test
        for i, (m, n) in enumerate(matches):
            if m.distance < 0.7*n.distance:
                matchesMask[i] = [1,0]

        draw_params = dict(matchColor = (0, 255, 0),
                           singlePointColor = (255, 0, 0),
                           matchesMask = matchesMask,
                           flags = 0)

        # result = cv2.drawMatchesKnn(query, kp_q, search, kp_s, matches, None, **draw_params)
        # pyplot.imshow(result, 'gray'), pyplot.show()

        return len(draw_params)

def printRankList(n, qStart = 1, qEnd = 11, sStart = 1, sEnd = 5001):
    rankList = open("rankList.txt", "w")
    queryPath = "../examples/example_query"
    searchPath = "../Images"
    list = []

    for q in range(qStart, qEnd):
        queryIndex = str(q)
        for s in range(sStart, sEnd):
            searchIndex = str(s)
            confidence = None
            confidence = SIFT(queryPath+queryIndex,searchPath+searchIndex)
            print(queryIndex + searchIndex + ":" +str(confidence))
            list.append([s, confidence])

        # order the result
        list = sorted(list, key=lambda x: x[1], reverse=True)

        rankList.write("Q" + str(q) + ":")
        for s in range(0, len(list)):
            rankList.write(str(list[s][0]) + "  ")
        rankList.write("\n")
        list = []

    rankList.close()
    resultFile = open("rankList.txt").read().splitlines()
    return resultFile

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    rankList = printRankList(11)


