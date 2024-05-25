import torch
import numpy as np
np.seterr(divide='ignore', invalid='ignore')

# NOTE: This will be the calculation of balanced accuracy for your classification task
# The balanced accuracy is defined as the average accuracy for each class. 
# The accuracy for an indiviual class is the ratio between correctly classified example to all examples of that class.
# The code in train.py will instantiate one instance of this class.
# It will call the reset methos at the beginning of each epoch. Use this to reset your
# internal states. The update method will be called multiple times during an epoch, once for each batch of the training.
# You will receive the network predictions, a Tensor of Size (BATCHSIZExCLASSES) containing the logits (output without Softmax).
# You will also receive the groundtruth, an integer (long) Tensor with the respective class index per example.
# For each class, count how many examples were correctly classified and how many total examples exist.
# Then, in the getBACC method, calculate the balanced accuracy by first calculating each individual accuracy
# and then taking the average.

# Balanced Accuracy
class BalancedAccuracy:
    def __init__(self, nClasses):
        # TODO: Setup internal variables
        # NOTE: It is good practive to all reset() from here to make sure everything is properly initialized
        self.nClasses = nClasses    
        self.reset()

    def reset(self):
        # TODO: Reset internal states.
        # Called at the beginning of each epoch
        self.number_of_predictions = 0
        self.array_truelabels = np.zeros(self.nClasses, dtype=int)
        self.array_truepositives = np.zeros(self.nClasses, dtype=int)
        self.array_falsepositives = np.zeros(self.nClasses, dtype=int)

    def update(self, predictions, groundtruth):
        # TODO: Implement the update of internal states
        # based on current network predictions and the groundtruth value.               
        # Predictions is a Tensor with logits (non-normalized activations)              
        # It is a BATCH_SIZE x N_CLASSES float Tensor. The argmax for each samples
        # indicated the predicted class.
        # Groundtruth is a BATCH_SIZE x 1 long Tensor. It contains the index of the
        # ground truth class.
        # Predictions = BATCH_SIZE x N_CLASSES
        # Groundtruth = BATCH_SIZE x 1

        self.number_of_predictions += len(groundtruth)
        list_groundtruth = groundtruth.tolist()
        predicted_labels = torch.argmax(predictions, dim=1)
        truepositives = (predicted_labels[groundtruth==predicted_labels]).tolist()
        falsepositives = (predicted_labels[groundtruth!=predicted_labels]).tolist()

        for idx in range(self.nClasses):
            self.array_truelabels[idx] += (list_groundtruth.count(idx))
            self.array_truepositives[idx] += (truepositives.count(idx))
            self.array_falsepositives[idx] += (falsepositives.count(idx))

    def getBACC(self):
        # TODO: Calculcate and return balanced accuracy 
        # based on current internal state

        print(f"array_truepositives: {self.array_truepositives}")
        print(f"array_truepositives: {self.array_truelabels}")
        #TP_rate = truepositive/real positive
        true_positive_rates = self.array_truepositives/self.array_truelabels

        # calculate real negatives:
        array_negatives = self.number_of_predictions - self.array_truelabels

        # calculate trueNegatives:
        array_truenegatives = array_negatives - self.array_falsepositives

        #TN_rate = truenegative/negative
        true_negative_rates = array_truenegatives/array_negatives

        #calculate balanced accuracy for each class
        array_balanced_acc = (true_negative_rates + true_positive_rates)/2

        # return mean bacc over all classes
        return np.mean(array_balanced_acc)
    
