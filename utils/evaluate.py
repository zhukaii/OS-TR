import numpy as np


class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

    def Mean_IoU(self, output, target):
        output[output >= 0.5] = 1
        output[output < 0.5] = 0
        seg = output[:, 0, :, :].data.cpu().numpy().astype(np.float64)
        # print(scores[0].shape)
        label = target.data.cpu().numpy().astype(np.float64)
        # print((target * seg).sum())
        # print(label.size(), seg.size())
        overlap = np.sum(label * seg)
        all_region = np.sum(label) + np.sum(seg)
        Acc = (overlap / (all_region - overlap))
        return Acc

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))
        MIoU1 = np.nanmean(MIoU[1:])
        # MIoU1 = np.nanmean(MIoU[[6, 8, 9, 10, 11, 15, 19, 23, 26, 28]])
        return MIoU1, MIoU

    def FBIoU(self):
        binary_hist = np.array((self.confusion_matrix[0, 0], self.confusion_matrix[0, 1:].sum(),
                                self.confusion_matrix[1:, 0].sum(), self.confusion_matrix[1:, 1:].sum())).reshape((2, 2))
        MIoU = np.diag(binary_hist) / (binary_hist.sum(1) + binary_hist.sum(0) - np.diag(binary_hist))
        MIoU = np.nanmean(MIoU)
        return MIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class ** 2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image, image_class):
        assert gt_image.shape == pre_image.shape
        # for i in range(gt_image.shape[0]):
        #     gt_image[i][gt_image[i] == 1] = image_class[i]
        #     pre_image[i][pre_image[i] == 1] = image_class[i]
        image_class = image_class[:, np.newaxis, np.newaxis]
        gt_image = np.where(gt_image == 1, image_class, gt_image)
        pre_image = np.where(pre_image == 1, image_class, pre_image)
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

