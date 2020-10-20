import operator
from math import log
import decisionTreePlot as dtPlot
from collections import Counter

def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels

def calcShannonEnt(dataSet):
    # list の長さを求める、学習に参加するデータ量を計算
    numEntries = len(dataSet)
    # 分類ラベルlabelが出現する回数
    labelCounts = {}
    # the the number of unique elements and their occurrence
    # featVec に dataSet を一行ずつ読み込ませる [1, 1, 'yes']......[1, 1, 'yes']......
    for featVec in dataSet:
        # 現在進行中データのラベルを保存、一行最後のデータ値がラベルになる
        currentLabel = featVec[-1]
        # 全ての可能性のある分類のために辞書を作成、現在進行中のキー値が存在しないとき、辞書を拡張して現在進行中のキー値を辞書に加える。すべてのキー値が現在進行中の分類が出現回数を記録
     　 if currentLabel not in labelCounts.keys():
           labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1

    # label ラベルの比率に対して、label ラベルのシャノンエントロピーを計算する
    shannonEnt = 0.0
    # labelCounts: {'yes': 2, 'no': 3}
    # key に labelCounts のキーを一つずつ読み込ませる
    for key in labelCounts:
        # ラベルの出現確率で、その分類が出現する確率を計算
        prob = float(labelCounts[key]) / numEntries
        # シャノンエントロピーを計算する、2を底とする対数を求める
        shannonEnt -= prob * math.log(prob, 2)
    return shannonEnt

def splitDataSet(dataSet, index, value):
    """
    Call:
       from: chooseBestFeatureToSplit()
       入力データ: (dataSet, 特徴値の列目, 重複要素排除後の特徴値)

     splitDataSet（dataSetをトラバーサルする，[index]が対応するcolnum列の值をvalueの行にする）
     index列をベースに分類を行う、index列のデータとvalueが等しいとき、indexを新しい作ったデータセットに分類する

     Args:
          dataSet データセット                                                 分類待ちのデータセット
          index    行が対応する[index]列   sample=[[1,1]
                       index=0(列) or 1(列)                    [1,0]]     分類データセットの特徴
          value    [index]列が対応するvalue                           特徴の値
      Returns:
          [index]列がvalueのデータセット【このデータセットはindex列を除外する必要がある】
     """
     retDataSet = []
     # dataSetの特徴部分を対象に
     for featVec in dataSet:
         # [index]列がvalueのデータセット【このデータセットはindex列を除外する必要がある】
         # [index]列の値が[value]なのか判断する
         if featVec[index] == value:
            # chop out index used for splitting
            # [:index] --前<index>行を意味する、indexが2とすれば、featVecの前<index>行を取る
            reducedFeatVec = featVec[:index]
            '''
            extendとappendの違い
            music_media.append(object) 配列に対象objectを追加
            music_media.extend(sequence) シリアルseqの内容の内容を配列に追加 ( += がlistでの使いが似ている， music_media += sequence)
            1.appendを使う際に、objectを対象として扱う、全体ををまとめてmusic_mediaに追加
            2.extendを使う際に、sequenceをシリアルとして扱う、このシリアルとmusic_mediaを合併させ、music_mediaの最後尾に置く
            music_media = []
            music_media.extend([1,2,3])
            print (music_media)
            ==>[1, 2, 3]

            music_media.append([4,5,6])
            print (music_media)
            ==>[1, 2, 3, [4, 5, 6]]

            music_media.extend([7,8,9])
            print music_media
            ==>[1, 2, 3, [4, 5, 6], 7, 8, 9]
            '''
            # [index+1:] --index の <index+1> 行を飛ばし、そのあとのデータを取る
            reducedFeatVec.extend(featVec[index+1:])
            # 結果値を収集する <index>列が<value>の行【この行はindex列を排除する必要がある】
            retDataSet.append(reducedFeatVec)
     return retDataSet

def chooseBestFeatureToSplit(dataSet):
    """chooseBestFeatureToSplit(最尤の特徴を選択)
    Args:
       dataSet データセット
    Returns:
       bestFeature 最尤の特徴列
    """
    # 一行目の Feature が何列あるのか確認する、最後の列はlabel列になる
    numFeatures = len(dataSet[0]) - 1
    # データセットの原始エントロピーを収集する
    baseEntropy = calcShannonEnt(dataSet)
    # 最尤の情報利得と最尤Featurnの番号
    bestInfoGain, baseFeature = 0.0, -1
    # iterate over all the features
    for i in range(numFeatures):
        # create a list of all the examples of this feature
        # 対応featureのすべてのデータを取得
        featList = [example [i] for example in dataSet]
        # get a set of unique value
        # 重複するデータを解き除く
        uniqueVals = set(featList)
        # 一時的な情報エントロピーを作成
        newEntropy = 0.0
        # 各列のvalue集合を遍歴、この列の情報エントロピーを計算
        # 現在進行中の特徴値におけるすべて唯一の属性値を遍歴する、それぞれの唯一属性に対してデータセットを作成する
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i , value)
            # 確率の計算
            prob = len(subDataSet) / float(len(dataSet))
            # 情報エントロピー
            newEntropy += prob * clacShannonEnt(subDataSet)
        # gain[情報利得] : データ分割前後の情報変化、情報エントロピー最大値を取得
        # 情報利得はエントロピーの減少あるいはデータ無秩序の度合いを意味する。最終的に、すべての特徴値中の情報利得を比較して、最も分割しやすい索引値を返す
        infoGain = baseEntropy - newEntropy
        print("infoGain=", infoGain,"bestFeature=", i, baseEntropy, newEntropy)
        if (infoGain > bestInfoGain):
           bestInfoGain = infoGain
           bestFeature = i
    return bestFeature

def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    # データセット最後列の最初の値が現れた回数＝データセットにおけるデータの数場合、つまりラベルが一種類しか存在しないことを意味する。その結果を返せばよい。
    # 1st 中止条件：全てのラベルが同じ、このラベルを返す
    # count() 関数は括弧中の値がlistに現れた回数を表す
    if classList.count(classList[0]) == len(classList):
       return classList[0]
    # データセットが一列のみ存在する場合、ラベル出現回数が一番多い分類を結果として扱う
    # 2nd 中止条件：全ての特徴を使い切っても、データセットを一分類のみが存在するグループにならない場合()
    if len(dataSet[0]) == 1:
       return majorityCnt(classList)

    # 最尤列を選び、最尤列が対応するラベルを取得（bestFeat: numFeaturesの第X列目）
    bestFeat = chooseBestFeatureToSplit(dataSet)
    # ラベルの名前を取得
    bestFeatLabel = labels[bestFeat]
    # myTreeを初期化する
    myTree = {bestFeatLabel: {}}
    # ! labels テーブルは可変対象、python関数中に引数として使うときアドレスを渡し、グローバルで変更可能
    # ラベルをリストから削除する
    del(labels[bestFeat])
    # 最尤特徴の列を取り出し、その枝で分類を行い
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        # 残されたラベルを求める
        subLabels = labels[:]
        # 現在選択された特徴が含まれた全ての特徴値を再帰的に処理、すべてのデータセットを分割するときに関数createTree()を呼び出す。
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
        # print('myTree', value, myTree)
    return myTree

def classify(inputTree, featLabels, testVec):
    """classify(入力された節点を与え、分類を行う)

    Args:
       inputTree  決定木モデル
       featLabels Featureラベルが対応する名前
       testVec      入力データをテストする

    Returns:
       classLabel  分類の結果値、labelと照合して名前を確認する
    """
    # treeの根節点が対応するキー値
    firstStr = list(inputTree.keys())[0]
    # キーを通して根節点が対応するvaluleを得る
    secondDict = inputTree[firstStr]
    # 根節点の名前で根節点がlabel内の前後順番を取得、これで入力されたtestVecがどうやって木を参照に分類を行うことがわかる
    featIndex = featLabels.index(firstStr)
    # データのテストを行う、根節点が対応するlabel位置を探す、これで、入力データの第何位から分類することがわかる
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    print('+++', firstStr, 'xxx', secondDict, '---', key, '>>>', valueOfFeat)
    # 枝分かれの終了を判断、valueOfFeatがdict型なのかを判断
    if isinstance(valueOfFeat, dict):
       classLabel = classify(valueOfFeat, featLabels, testVec)
    else:
       classLabel = valueOfFeat
    return classLabel

def fishTest():
    #  データとラベルを作成
    myDat, labels = createDataSet()
    # print myDat, labels

    # label分類のシャノンエントロピーを計算
    # calcShannonEnt(myDat)

    # 0列目の 1/0の列のデータセットを求める
    # print '1---', splitDataSet(myDat, 0, 1)
    # print '0---', splitDataSet(myDat, 0, 0)

    # print chooseBestFeatureToSplit(myDat)

    import copy
    myTree = createTree(myDat, copy.deepcopy(labels))
    print(myTree)
    # [1, 1]は取る予定の枝上の節点位置を意味する、結果値に対応する
    print(classify(myTree, labels, [1, 1]))

    # 木の高さを求める
    # print(get_tree_height(myTree))

    # 木を描く
    dtPlot.createPlot(myTree)

if __name__ == "__main__":
    fishTest()
