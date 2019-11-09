import numpy as np
# from torch.utils.serialization import load_lua
import joblib
import torchfile

def read_VT_clinical(datatype="original"):
    if datatype=='original':
        data_path='./data/clinical_dataset/qrsData.t7'
        
        #qrsData = load_lua(data_path)
        qrsData = torchfile.load(data_path)
        
        print("type(qrsData): ", type(qrsData))
        print("qrsData: ", qrsData.keys())
        
        trainX = None
        valX = None
        testX = None

        trainY = np.array(qrsData.train_y)
        valY = np.array(qrsData.val_y)
        testY = np.array(qrsData.test_y)
        
        print("train ids shape: ", trainY[:, 1].astype(int).shape)
        print("train ids: ", np.unique(trainY[:, 1].astype(int)))
        print("val ids: ", np.unique(valY[:, 1].astype(int)))
        print("test ids: ", np.unique(testY[:, 1].astype(int)))
        print("total patients: ", len(testY[:, 1].astype(int)))
        
        np.save('./data/clinical_dataset/trainY_id.npy', arr=trainY[:, 1].astype(int))
        np.save('./data/clinical_dataset/valY_id.npy', arr=valY[:, 1].astype(int))
        np.save('./data/clinical_dataset/tesY_id.npy', arr=testY[:, 1].astype(int))
        
        return trainX, valX, testX, trainY[:, 0].astype(int), valY[:, 0].astype(int), testY[:, 0].astype(int)
    if datatype == 'resplit':
        data_path = '/home/zl7904/Documents/projects/VT_test/qrsData_resplit'
        data = joblib.load(data_path)
        return data['X_tr'], data['X_vld'], data['X_test'], data['y_tr'], data['y_vld'], data['y_test']

read_VT_clinical()
