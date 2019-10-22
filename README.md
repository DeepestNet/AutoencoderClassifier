# AutoencoderClassifier
Supervised and unsupervised training using CIFAR-10 dataset

-----------------------------------------------------------------------------------------------------------------------------

Dependencies:

python2.7

keras '2.1.5'

matplotlib '2.2.2'

sklearn '0.20.0'

tensorflow '1.4.0'


train.py is the training program that trains autoencoder, initial classifier and final classifier; also saves all the model, their corresponding weights and figures to disk.

test.py is the evaluation program that evaluates the performance of the classifier.


How to Run:

python train.py --dataset cifar-10-batches-py --model_ae model_autoencoder --model_cl model_classifier --images images

python test.py --dataset cifar-10-batches-py --model_cl model_classifier --images images
