from numpy import *
import operator
import os
from collections import Counter

def file2matrix(filename):
     """
     Desc:
         学習データを入力
     parameters:
         filename: データファイルパス
     return:
         データ行列 returnMat と対応する分類タイプ classLabelVector
     """
    fr = open(filename)
    # ファイルにあるデータの行数を取得
    numberOfLines = len(fr.readlines())
    # 対応する空き行列を生成
    # 例：zero(2,3)は各位置が0になる2*3の行列を生成する。
    returnMat = zeros((numberOfLines, 3)) # prepare matrix to return
    classLabelVector = [] # prepare labels return
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        # str.strip([chars]) --文字列の先頭と末尾から指定した文字を除去する
        #                          --引数を指定しないと空白文字を除去する
        line = line.strip()
        # '\t'で文字列を分割する
        listFromLine = line.split('\t')
        # 列ごとのプロパティデータ
        returnMat[index, :] = listFromLine[0:3]
        # 列ごとのラベルデータ
        # [-1] --末尾の要素を指定する
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    # データ行列returnMatと対応するラベルclassLabelVectorを返す
    return returnMat, classLabelVector

def autoNorm(dataSet):
    """
    Desc:
            正規化の特徴値、過剰に大きい/小さいが発生する影響を消去
    parameter:
            dataSet: 数据集
    return:
           正規化後のデータセット normDataSet. rangesとminValsは範囲と最小値、最終的に使わない

    正規化用数学公式:
           Y = (X-Xmin)/(Xmax-Xmin)
           min と max はデータセット中の最小特徴値と最大特徴値。この関数は自動的に数値特徴を0-1の区間に転換することが可能。
    """
    # 各属性(各列)の最大値、最小値、範囲を計算する
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    # 範囲
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    # m はデータの行数を表す、行数の第一次元でもある
    # shape[1] --配列の列数を表す
    m = dataSet.shape[0]
    # 最小値との差の値が構成する行列を作成する
    # tile(value, (m,1)) 一行の値がvalueのmサイズの行列を1回反復して作成
    normDataSet = dataSet - tile(minVals, (m,1))
    # 最小値との差を範囲で割り行列を作成
    normDataSet = normDataSet / tile(ranges,(m,1)) # element wise divide
    return normDataSet, ranges, minVals

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    # 距離測定 数式はユークリッド距離
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    # 横軸で配列を結合する
    # --axis=1 横軸
    # --axis=0 縦軸
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5

    # 距離で順番を並べ替え: min → max
    # argsort() --値をソートした際に元の配列のインデックス番号が格納されたndarrayを返す
    sortedDistIndicies = distances.argsort()
    # 前k個最短距離を選ぶ, k個中一番多い分類ラベルを選択
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        # .get(key, keyが存在しない場合の返し値)
        # classCount dictのkeyが対応する値+1
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    # items() --keyとvalueをtupleに格納したlistを生成して返す
    # iteritems() --イテレータを返す
    # sorted() --ソートした新たなリストを生成
    # operator.itemgetter(value) --valueをkeyにして並び替える
    # operator.itemgetter(1) --タプルの2つ目の要素で並び替える(Dictでは1が値、0がキーになる)
    # reverse=True --降順配列
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True))
    return sortedClassCount[0][0]

def datingClassTest():
    """
    Desc:
       出会いサイトを対象にしたテストメソッド
    parameter:
       none
    return:
       failedの数
    """
    # テストデータの比率を設定する（学習データセットの比率=1-hoRation）
    hoRation = 0.1  #テスト範囲、一部がテスト一部がサンプル
    # ファイルからデータを読み取る
    datingDateMat, datingLabels = file2matrix('./KNN/datingTestSet2.txt')
    # 正規化データ
    normMat, ranges, minVals = autoNorm(datingDateMat)
    # m はデータの行数を表す、行数の第一次元でもある
    m = normMat.shape[0]
    # テスティングサンプルの数を設定、numTestVecs:m が学習サンプルの数を表す
    numTestVecs = int(m * hoRation)
    print ('numTestVecs=', numTestVecs)
    errorCount = 0.0
    for i in range(numTestVecs):
        # テストデータに対して
        # normMat, datingLabels
        #               --numTestVecs:m → 100:1000 のデータとラベルをデータセットとして渡す
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]): errorCount += 0
     print ("the total error rate is: %f" % (errorCount / float(numTestVecs)))



def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        # lineStr に一時的に i 行のデータを読み込ませる
        # readline() --ファイルを1行読み込み、文字列を返します
        lineStr = fr.readline()
        for j in range(32):
            # 読み込んだ i 行目の数字を一つずつ抽出して returnVectに格納
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def handwritingClassTest():
    # 1. 学習データを入力する
    hwLabels = []
    # ディレクトリ下ファイルのリストを取得する
    trainingFileList = listdir(".\trainingDigits\")  # load the training set
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    # hwLabelsに0-9が対応するindex位置をセーブする, trainingMatに各位置が対応する画像のベクトルを格納する
    for i in range(m):
        fileNameStr = trainingFileList[i]
        # . 以降の文字列をカットする
        fileStr = fileNameStr.split('.')[0] # take off .txt
        # _ 以降の文字列をカットする
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        # 32*32 の行列を 1*1024 の行列に転換する
        trainingMat[i, :] = img2vector('.\trainingDigits/%s' % fileNameStr')

        # 2. テストデータを入力する
        testFileList = listdir('.\testDigits') # iterate through the test set
        errorCount = 0.0
        mTest = len(testFileList)
        for i in range(mTest):
            filerNameStr = testFileList[i]
            filerStr = fileNameStr.split('.')[0] # take off .txt
            classNumStr = int(fileStr('_')[0])
            vectorUnderTest = img2vector('.\testDigits'%s % fileNameStr)
            classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
            print ("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
            if (classifierResult != classNumstr): errorCount += 1.0
        print("\n the total number of errors is: %d" % errorCount)
        print("\n the total error rate is: %f" % (errorCount / float(mTest)))


if __name__ == '__main__':
    datingClassTest()
    handwritingClassTest()
