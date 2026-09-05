## purpose

this model allows for early and accurate detection of tumours 

## tech stack

CNN because input data is MRI images, CNN are well suited for extracting important features from spatial data
I chose 3 convolutional layers because I wanted enough depth to learn progressively complex features from MRI images without making the model unnecessarily large. The first layer can learn low-level features such as edges and intensity patterns, the second can combine them into textures and shapes, and the third can capture more complex tumor-related structures.
✅ The network learns increasingly abstract spatial features through successive convolutional layers, and the final layers use those learned representations for classification.

I chose PyTorch because I wanted direct control over the CNN architecture and, more importantly, I wanted to understand what was happening inside the model rather than treating it as a black box. I built a simple 3-layer convolutional architecture with ReLU and pooling myself so I could explicitly control things like the number of filters, kernel sizes, feature-map dimensions, activation functions, and training process.

## workflow

80-10-10
train.py     - we train the model - augment - flip & rotate, early stopping w patience, train for <100 epochs, BCE loss and adam optimizer, save parameters in best.pth file
test.py      - we test the model on test set - print accuracy and other metrics
predict.py   - input - mri.jpg - output - classfn

## architecture

3 x (Conv,ReLu,Pooling)
1 x FCC (Linear,ReLu,Dropout)
1 x Output layer (Linear,Sigmoid)

## problems faced

incremental layers approach -> not accurate enough
more layers -> overfitting / overlearning from noise
false positive -> acceptable because at least FN is low
