# Optical-Character-Recognition
Optical Character Recognition

TRAINING: 

Trained on two images training_chars.png and training_chars2.png. 

Used the Tensorflow Inception model to train the data.

Steps:

To create training data, 

python create_train_data.py

Arrange the data as required by the inception model.

To train on the created data(remember to copy-paste images to create more than 30 images for each class):

python train.py --bottleneck_dir=bottlenecks --how_many_training_steps=500 --model_dir=inception --summaries_dir=training_summaries/basic --output_graph=retrained_graph.pb --output_labels=retrained_labels.txt --image_dir=train
  
To test the data: 

python test.py

