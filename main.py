import cv2
import numpy as np
from matplotlib import pyplot
from feature_extractor import FeatureExtractor
from PIL import Image

def getQueryImgs(qStart=1, qEnd=11):
    queryPath = "../examples/example_query/"
    qList = []
    for q in range(qStart, qEnd):
        queryIndex = str(q).zfill(2)
        qList.append(queryPath + queryIndex)
    return qList

def getSearchImgs(sStart=1, sEnd=5001):
    searchPath = "../Images/"
    sList = []
    for s in range(sStart, sEnd):
        searchIndex = str(s).zfill(4)
        sList.append(searchPath + searchIndex)
    return sList

def getQueryORB(queryPath):
    query = cv2.imread(queryPath + ".jpg")
    box1 = open(queryPath + ".txt").read().split()
    x1 = int(box1[0])
    y1 = int(box1[1])
    w1 = int(box1[2])
    h1 = int(box1[3])
    query = query[y1:y1 + h1, x1:x1 + w1]
    query = cv2.cvtColor(query, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create()
    kp_q, des_q = orb.detectAndCompute(query, None)
    return kp_q, des_q

def ORB(kp_q, des_q, imagePath):
    search = cv2.imread(imagePath + ".jpg", cv2.COLOR_RGB2GRAY)

    if int(str(imagePath).split('Images/')[1]) <= 2000:
        box2 = open(imagePath + ".txt").read().split()
        x2 = int(box2[0])
        y2 = int(box2[1])
        w2 = int(box2[2])
        h2 = int(box2[3])
        search = search[y2:y2 + h2, x2:x2 + w2]

    orb = cv2.ORB_create()
    kp_s, des_s = orb.detectAndCompute(search, None)

    # Feature Matching - BF
    MIN_MATCH_COUNT = 10
    bf = cv2.BFMatcher(cv2.NORM_HAMMING2, crossCheck=False)
    if (des_s is not None and des_q is not None):
        matches = bf.knnMatch(des_q, des_s, k=2)
        # ratio test
        good = []
        if matches is not None and len(matches[0]) > 1:
            for m, n in matches:
                if m.distance < 0.8 * n.distance:
                    good.append(m)

        # RANSAC
        if len(good)>MIN_MATCH_COUNT:
            src_pts = np.float32([kp_q[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_s[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            matchesMask = mask.ravel().tolist()
            # h, w, mode = query.shape
            # pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            # dst = cv2.perspectiveTransform(pts, M)
            # search = cv2.polylines(search, [np.int32(dst)], True, (127,255,0), 3, cv2.LINE_AA)
            count = 0
            for i in matchesMask:
                if i == 1:
                    count = count + 1
        else:
            matchesMask = None
            count = len(good)
        return (count / len(matches) * 100)
    return 0

def SIFT(queryPath, imagePath):
    query = cv2.imread(queryPath + ".jpg", cv2.COLOR_RGB2GRAY)
    search = cv2.imread(imagePath + ".jpg", cv2.COLOR_RGB2GRAY)

    box1 = open(queryPath + ".txt").read().split()
    x1 = int(box1[0])
    y1 = int(box1[1])
    w1 = int(box1[2])
    h1 = int(box1[3])

    query = query[y1:y1 + h1, x1:x1 + w1]

    if int(str(imagePath).split('Images/')[1]) <= 2000:
        box2 = open(imagePath + ".txt").read().split()
        x2 = int(box2[0])
        y2 = int(box2[1])
        w2 = int(box2[2])
        h2 = int(box2[3])

        search = search[y2:y2 + h2, x2:x2 + w2]

    sift = cv2.SIFT_create()
    kp_q, des_q = sift.detectAndCompute(query, None)
    kp_s, des_s = sift.detectAndCompute(search, None)

    # # Feature Matching - BF
    # bf = cv2.BFMatcher()
    # matches = bf.knnMatch(des_q, des_s, k=2)
    #
    # # Filter out good matches
    # # matches = sorted(matches, key=lambda x: x.distance)
    #
    # # Apply ratio test
    # good = []
    # for m, n in matches:
    #     if m.distance < 0.75 * n.distance:
    #         good.append([m])

    # # Feature Matching - FLANN
    MIN_MATCH_COUNT = 10
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des_q, des_s, k=2)

    # ratio test
    good = []
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.8 * n.distance:
            good.append(m)

    # print("match number: "+str(len(matches)))
    # print("good number: " + str(len(good)))

    # RANSAC
    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([kp_q[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_s[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
        # h, w, mode = query.shape
        # pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        # dst = cv2.perspectiveTransform(pts, M)
        # search = cv2.polylines(search, [np.int32(dst)], True, (127,255,0), 3, cv2.LINE_AA)
        count = 0
        for i in matchesMask:
            if i == 1:
                count = count + 1
    else:
        matchesMask = None
        count = len(good)

    # draw_params = dict(matchColor=(0, 255, 0),
    #                    singlePointColor=(255, 0, 0),
    #                    matchesMask=matchesMask,
    #                    flags=2)
    # result = None
    # result = cv2.drawMatches(query, kp_q, search, kp_s, good, None, **draw_params)
    # pyplot.imshow(result, 'Accent'), pyplot.show()

    return (count / len(matches) * 100)

def getFE():
    fe = FeatureExtractor()
    return fe

def getQueryFeature(fe):
    # qList = getQueryImgs()
    queryPath = "../examples/example_query/"
    for q in range(1, 11):
        queryIndex = str(q).zfill(2)
        query = queryPath+queryIndex
        queryimg = Image.open(query + ".jpg")
        box1 = open(query + ".txt").read().split()
        x1 = int(box1[0])
        y1 = int(box1[1])
        w1 = int(box1[2])
        h1 = int(box1[3])
        queryimg = queryimg.crop((x1, y1, x1 + w1, y1 + h1))
        query = fe.extract(queryimg)
        feature_path = "features/query/"+queryIndex + ".npy"
        np.save(feature_path, query)
    return None

def getSearchFeature(fe):
    features = []
    searchPath = "../Images/"
    for s in range(1, 5001):
        searchIndex = str(s).zfill(4)
        search = searchPath + searchIndex
        searchimg = Image.open(search + ".jpg")
        if s <= 2000:
            box2 = open(search + ".txt").read().split()
            x2 = int(box2[0])
            y2 = int(box2[1])
            w2 = int(box2[2])
            h2 = int(box2[3])
            searchimg = searchimg.crop((x2, y2, x2 + w2, y2 + h2))
        feature = fe.extract(img=searchimg)
        features.append(feature)
    features = np.array(features)
    feature_path = "features/search/features.npy"
    np.save(feature_path, features)
    return features

def CNN(query, features):
    dists = np.linalg.norm(features - query, axis=1)
    ids = np.argsort(dists)  # Top 30 results
    scores = [[id+1, 1 / dists[id]] for id in ids]
    # scores = dists[0]
    # confidence = 1 / scores
    return scores

# average precision
def ave_pre(array):
    ap = 0
    for idx, m in enumerate(array):
        ap_m = (idx + 1) / (m + 1)
        ap = ap + ap_m
    ap = ap / len(array)
    return ap

def map():
    rank_line = open('rank_groundtruth.txt').read().splitlines()
    rank_result = open('rankList.txt').read().splitlines()
    ap_sum = 0
    for idx, line in enumerate(rank_result):
        if idx > 11:
            break
        line_str = line.split()
        query_num = int(line_str[0][1]) - 1
        result_num = [int(x) for x in line_str[1:]]
        rank_str = rank_line[idx].split()
        rank_gt = [int(x) for x in rank_str[1:]]
        find_idx = []
        for num in rank_gt:
            ind = np.where(np.array(result_num) == num)
            find_idx.extend(ind)
        find_idx = np.array(find_idx).reshape(len(find_idx), )
        find_idx = np.sort(find_idx)
        ap = ave_pre(find_idx)
        print("Average Precision of Q%d: %.4f" % (idx + 1, ap))
        ap_sum = ap_sum + ap
    print("Mean Average Precision: %f" % (ap_sum / len(rank_result)))


def printRankList(qStart=1, qEnd=11, sStart=1, sEnd=5001):
    rankList = open("rankList.txt", "w")
    # queryPath = "../examples/example_query/"
    queryPath = "../Queries/"
    searchPath = "../Images/"
    list = []
    # fe = getFE()
    # getQueryFeature(fe)
    # getSearchFeature(fe)

    for q in range(1, 21):
        queryIndex = str(q).zfill(2)
        kp, des = getQueryORB(queryPath + queryIndex)

        for s in range(1, 5001):
            searchIndex = str(s).zfill(4)
            confidence = 0
            # confidence = SIFT(queryPath + queryIndex, searchPath + searchIndex)
            confidence = ORB(kp, des, searchPath + searchIndex)
            # print(queryIndex + searchIndex + ":" + str(confidence))
            list.append([s, confidence])

        # order the result
        list = sorted(list, key=lambda x: x[1], reverse=True)

        rankList.write("Q" + str(q) + ": ")
        for s in range(0, len(list)):
            rankList.write(str(list[s][0]) + ' ')
        rankList.write("\n")
        list = []

    rankList.close()
    resultFile = open("rankList.txt").read().splitlines()
    return resultFile

def test(qStart=1, qEnd=2, sStart=250, sEnd=260):
    rankList = open("rankList.txt", "w")
    # queries = getQueryImgs(qStart, qEnd)
    # searches = getSearchImgs(sStart, sEnd)
    fe = getFE()
    # getQueryFeature(fe)
    features = np.load("features/search/features.npy")
    for q in range(qStart, 11):
        queryIndex = str(q).zfill(2)
        query = np.load("features/query/"+ str(queryIndex) + ".npy")
        list = CNN(query, features)
        list = sorted(list, key=lambda x: x[1], reverse=True)

        rankList.write("Q" + str(q) + ": ")
        for s in range(0, len(list)):
            rankList.write(str(list[s][0]) + ' ')
        rankList.write("\n")
        list = []

    rankList.close()
    resultFile = open("rankList.txt").read().splitlines()
    return resultFile

def combine():
    ORB = open("rankList_ORB.txt").read().splitlines()
    CNN = open("rankList_CNN.txt").read().splitlines()
    rankList = open("rankList.txt", "w")

    # m2_list = []
    # for x in range(0, 20):
    #     max = 0
    #     min = 0
    #     mean = 0
    #     for y in range(0, 5000):
    #         prefix = x * 5000
    #         line = prefix + y
    #         m2_sim = float(CNN[line].split(' ')[2])
    #         max = m2_sim if m2_sim > max else max
    #         min = m2_sim if m2_sim < min else min
    #         mean = mean + m2_sim
    #
    #     mean = mean / 5000
    #     temp = [min, mean, max, max - min]
    #     m2_list.append(temp)
    #
    # for x in range(0, 20):
    #     list = []
    #     for y in range(0, 5000):
    #         prefix = x * 5000
    #         line = prefix + y
    #         m1_sim = 1000000 * float(ORB[line].split(' ')[2])
    #         m2_sim = float(CNN[line].split(' ')[2])
    #         m2_sim = 1 - (m2_sim - m2_list[x][0]) / (m2_list[x][3])
    #         sim = m1_sim * m2_sim
    #         rankList.write(str(ORB[line].split(' ')[0]) + ' ' + ORB[line].split(' ')[1] + ' ' + str(sim) + '\n')
    #         temp = [y + 1, sim]
    #         list.append(temp)
    #
    #     # reverse true = descending
    #     list = sorted(list, key=lambda x: x[1], reverse=True)
    #
    #     rankList.write('Q' + str(x + 1) + ': ')
    #     for i in range(0, len(list)):
    #         rankList.write(str(list[i][0]))
    #         rankList.write(' ')
    #     rankList.write('\n')

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    rankList = printRankList()
    # test()
    # map()
    # combine()
    # fe = getFE()
    # getSearchFeature(fe)
