import torch
import numpy as np
import torch.utils.data as data_utils
import torch.utils.data as utils
import torchfile


class TMI_Dataset():
    def __init__(self, mode, root='/data', num_leads=12, dataset_type=''):
        self.dataset_type = dataset_type.lower()
        self.root = root
        print("Loading {} dataset.".format(self.dataset_type))
        if self.dataset_type == 'simulated':
            index_dict = {'train': 242, 'validation': 485, 'test': 728}
            self.data = torch.load(root + '/allData_{}.pth'.format(index_dict[mode]))
            self.np_data = self.data.data_tensor.numpy()
            # self.np_data = np.reshape(self.np_data, newshape=(self.np_data.shape[0], num_leads,
            #                                                   int(self.np_data.shape[-1]/num_leads)))

            self.np_targets = self.data.target_tensor.numpy()
        elif self.dataset_type == 'clinical':
            data_file_mapping = {'train': '/train_x.pt', 'validation': '/val_x.pt', 'test': '/test_x.pt'}
            label_file_mapping = {'train': '/train_y.pt', 'validation': '/val_y.pt', 'test': '/test_y.pt'}
            self.data = torch.load(root + data_file_mapping[mode]).permute(0, 2, 1)
            self.np_data = self.data.numpy()
            self.label = torch.load(root + label_file_mapping[mode]) - 1
            # self.label = torch.load(root + label_file_mapping[mode])
            self.np_targets = self.label.numpy()

        print("self.np_data.shape: ", self.np_data.shape)
        print("self.np_targets.shape: ", self.np_targets.shape)
        print()

    def __len__(self):
        return self.np_data.shape[0]

    def __getitem__ (self, index):
        if self.dataset_type == 'clinical':
            return torch.from_numpy(self.np_data[index, :, :]), torch.from_numpy(self.np_targets[index, :])

    def getRegularSet(self):
        signals = self.data.data_tensor
        labels = self.data.target_tensor

        return signals, labels

    def getSmallerSubset(self):
        signals = self.data[:][0]
        labels = self.data[:][1]

        return signals[:134865], labels[:134865]

    def getSmallerSubset_tx(self):
        """
        This is to create a training set with y_tx = 1, z_tx = 1
        """
        labels = self.data[:][1]
        labels = labels.numpy()
        index = np.where(labels[:,4] == - 5)
        index1 = np.where(labels[index[0], 5] == -20)

        signals1 = self.data[:][0].numpy()
        signals1 = signals1[index1]
        labels1 = labels[index1]

        dataIndex2 = torch.load(self.root + 'allData_{}.pth'.format(485))
        labels2 = dataIndex2[:][1]
        labels2 = labels2.numpy()
        index = np.where(labels2[:,4] == - 5)
        index2 = np.where(labels[index[0], 5] == -20)

        signals2 = dataIndex2[:][0].numpy()
        signals2 = signals2[index2]
        labels2 = labels2[index2]
        del dataIndex2

        dataIndex3 = torch.load(self.root + 'allData_{}.pth'.format(728))
        labels3 = dataIndex3[:][1]
        labels3 = labels3.numpy()
        index = np.where(labels3[:,4] == - 5)
        index3 = np.where(labels[index[0], 5] == -20)

        signals3 = dataIndex3[:][0].numpy()
        signals3 = signals3[index3]
        labels3 = labels3[index3]
        del dataIndex3

        signals = torch.FloatTensor(signals1)
        labels = torch.FloatTensor(labels1)

        signals = torch.cat([signals, torch.FloatTensor(signals2)], 0)
        labels = torch.cat([labels, torch.FloatTensor(labels2)], 0)

        signals = torch.cat([signals, torch.FloatTensor(signals3)], 0)
        labels = torch.cat([labels, torch.FloatTensor(labels3)], 0)

        return signals, labels

    def getSmallerSubset_tx_val(self):
        """
        This is to create a validation set with z_tx = 2, making the situation similar as previous experiment setup!
        """
        labels = self.data[:][1]
        labels = labels.numpy()
        index1 = np.where(labels[:, 5] == - 0)

        signals1 = self.data[:][0].numpy()
        signals1 = signals1[index1]
        labels1 = labels[index1]

        dataIndex2 = torch.load(self.root + 'allData_{}.pth'.format(485))
        labels2 = dataIndex2[:][1]
        labels2 = labels2.numpy()
        index2 = np.where(labels[:, 5] == - 0)

        signals2 = dataIndex2[:][0].numpy()
        signals2 = signals2[index2]
        labels2 = labels2[index2]
        del dataIndex2

        dataIndex3 = torch.load(self.root + 'allData_{}.pth'.format(728))
        labels3 = dataIndex3[:][1]
        labels3 = labels3.numpy()
        index3 = np.where(labels[:, 5] == - 0)

        signals3 = dataIndex3[:][0].numpy()
        signals3 = signals3[index3]
        labels3 = labels3[index3]
        del dataIndex3

        signals = torch.FloatTensor(signals1)
        labels = torch.FloatTensor(labels1)

        signals = torch.cat([signals, torch.FloatTensor(signals2)], 0)
        labels = torch.cat([labels, torch.FloatTensor(labels2)], 0)

        signals = torch.cat([signals, torch.FloatTensor(signals3)], 0)
        labels = torch.cat([labels, torch.FloatTensor(labels3)], 0)

        return signals, labels

    def getSmallerSubset_tx_test(self):
        """
        This is to create a test set with z_tx = 3, making the situation similar as previous experiment setup!
        """
        labels = self.data[:][1]
        labels = labels.numpy()
        index1 = np.where(labels[:, 5] == 20)

        signals1 = self.data[:][0].numpy()
        signals1 = signals1[index1]
        labels1 = labels[index1]

        dataIndex2 = torch.load(self.root + 'allData_{}.pth'.format(485))
        labels2 = dataIndex2[:][1]
        labels2 = labels2.numpy()
        index2 = np.where(labels[:, 5] == 20)

        signals2 = dataIndex2[:][0].numpy()
        signals2 = signals2[index2]
        labels2 = labels2[index2]
        del dataIndex2

        dataIndex3 = torch.load(self.root + 'allData_{}.pth'.format(728))
        labels3 = dataIndex3[:][1]
        labels3 = labels3.numpy()
        index3 = np.where(labels[:, 5] == 20)

        signals3 = dataIndex3[:][0].numpy()
        signals3 = signals3[index3]
        labels3 = labels3[index3]
        del dataIndex3

        signals = torch.FloatTensor(signals1)
        labels = torch.FloatTensor(labels1)

        signals = torch.cat([signals, torch.FloatTensor(signals2)], 0)
        labels = torch.cat([labels, torch.FloatTensor(labels2)], 0)

        signals = torch.cat([signals, torch.FloatTensor(signals3)], 0)
        labels = torch.cat([labels, torch.FloatTensor(labels3)], 0)

        return signals, labels

    def test_tx(self, testNum):
        """
        Here, I want to create different levels of dataset for z_tx = 1, y_tx = 1 (training set).
        Level 1 : test 1 --> y = 1, z = 2
        Level 1 : test 2 --> y = 2, z = 1

        Level 2 : test 3 --> y = 2, z = 2
        Level 2: test 4 --> y = 1, z = 3
        Level 2: test 5 --> y = 3, z = 1

        Level 3 : test 6 --> y = 2, z = 3
        Level 3 : test 7 --> y = 3, z = 2

        Level 4 : test 8 --> y = 3, z = 3
        """
        y = 10
        z = 10
        if testNum == 1:
            y = -5
            z = -0
        elif testNum == 2:
            y = 0
            z = -20
        elif testNum == 3:
            y = 0
            z = 0
        elif testNum == 4:
            y = -5
            z = 20
        elif testNum == 5:
            y = 5
            z = -20
        elif testNum == 6:
            y = -0
            z = 20
        elif testNum == 7:
            y = 5
            z = -0
        elif testNum == 8:
            y = 5
            z = 20

        labels = self.data[:][1]
        labels = labels.numpy()
        index = np.where(labels[:,4] == y)
        index1 = np.where(labels[index[0], 5] == z)

        signals1 = self.data[:][0].numpy()
        signals1 = signals1[index[0][index1]]
        labels1 = labels[index[0][index1]]


        dataIndex2 = torch.load(self.root + 'allData_{}.pth'.format(485))
        labels2 = dataIndex2[:][1]
        labels2 = labels2.numpy()
        index = np.where(labels2[:,4] == y)
        index2 = np.where(labels[index[0], 5] == z)

        signals2 = dataIndex2[:][0].numpy()
        signals2 = signals2[index[0][index2]]
        labels2 = labels2[index[0][index2]]

        del dataIndex2

        dataIndex3 = torch.load(self.root + 'allData_{}.pth'.format(728))
        labels3 = dataIndex3[:][1]
        labels3 = labels3.numpy()
        index = np.where(labels3[:,4] == y)
        index3 = np.where(labels[index[0], 5] == z)

        signals3 = dataIndex3[:][0].numpy()
        signals3 = signals3[index[0][index3]]
        labels3 = labels3[index[0][index3]]

        del dataIndex3

        signals = torch.FloatTensor(signals1)
        labels = torch.FloatTensor(labels1)

        signals = torch.cat([signals, torch.FloatTensor(signals2)], 0)
        labels = torch.cat([labels, torch.FloatTensor(labels2)], 0)

        signals = torch.cat([signals, torch.FloatTensor(signals3)], 0)
        labels = torch.cat([labels, torch.FloatTensor(labels3)], 0)

        return signals, labels

    def test(self, testNum):
        """
        Here, I want to create different levels of dataset.
        Level 1 : test 1 --> x = 1, y = 2
        Level 1 : test 2 --> x = 2, y = 1

        Level 2 : test 3 --> x = 2, y = 2

        Level 3 : test 4 --> x = 2, y = 3
        Level 3 : test 5 --> x = 3, y = 2

        Level 4 : test 6 --> x = 3, y = 3

        Level 5: test 7 --> x = 1, y = 3
        Level 5: test 8 --> x = 3, y = 1
        """
        index2 = 485
        index3 = 728

        if testNum == 1:
            signals = self.data[:][0]
            labels = self.data[:][1]
            return signals[134865:269730], labels[134865:269730]

        elif testNum == 2:
            dataNew = torch.load(self.root + 'allData_{}.pth'.format(index2))
            signals = dataNew[:][0]
            labels = dataNew[:][1]
            return signals[:134865], labels[:134865]

        elif testNum == 3:
            dataNew = torch.load(self.root + 'allData_{}.pth'.format(index2))
            signals = dataNew[:][0]
            labels = dataNew[:][1]
            return signals[134865:269730], labels[134865:269730]

        elif testNum == 4:
            dataNew = torch.load(self.root + 'allData_{}.pth'.format(index2))
            signals = dataNew[:][0]
            labels = dataNew[:][1]
            return signals[269730:], labels[269730:]

        elif testNum == 5:
            dataNew = torch.load(self.root + 'allData_{}.pth'.format(index3))
            signals = dataNew[:][0]
            labels = dataNew[:][1]
            return signals[134865:269730], labels[134865:269730]

        elif testNum == 6:
            dataNew = torch.load(self.root + 'allData_{}.pth'.format(index3))
            signals = dataNew[:][0]
            labels = dataNew[:][1]
            return signals[269730:], labels[269730:]

        elif testNum == 7:
            signals = self.data[:][0]
            labels = self.data[:][1]
            return signals[269730:], labels[269730:]

        elif testNum == 8:
            dataNew = torch.load(self.root + 'allData_{}.pth'.format(index3))
            signals = dataNew[:][0]
            labels = dataNew[:][1]
            return signals[:134865], labels[:134865]
