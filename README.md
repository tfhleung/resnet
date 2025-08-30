**PyTorch Implementation of ResNet**

This repo consists of an implementation of the ResNet model (He et. al 2015) using the PyTorch library.  The model was tested using a dataset of 7000 Pokemon (https://www.kaggle.com/datasets/lantian773030/pokemonclassification).  The model was also verified using the built-in versions of ResNet which achieved model accuracies of ~75%.  However, the accuracy improved significantly to ~95% when pretrained weights were used which suggests larger training datasets would be greatly beneficial to achieve more accurate results.

The ResNet is a modification of the original VGG-19 net with additional skip connections to help address the vanishing gradient problem.  Generally, networks become more difficult to train as the number of hidden layers is increased due to the vanishing gradient problem, where the computed gradient values become exponentially smaller as one progresses further through the network.  However, this problem can be alleviated by introducing so-called 'shortcut connections', which recasts the training problem to the optimization of a residual mapping instead of a direct mapping between the model predictions and ground truth.  The shortcut connections are constructed by adding the input directly to the output so that the output is formally written as $\mathcal{F}(x) + x$ and are relatively straighforward to implement.  The only complexity that arises is when a mismatch between the input and output dimensions occurs, in which case an additional convolution layer (with kernel size 1x1) is applied to the identity shortcut to ensure that the mapping is dimensionally consistent.

Two variants of the shortcut connections are implemented, the so-called 'basic' and 'bottleneck' blocks.  The basic block consists of a stack of two convolutional layers with kernel size of 3x3.  For very deep models, the bottleneck block is designed which consists of three convolutional layers with kernel sizes of 1x1, 3x3 and 1x1, respectively.  The 1x1 convolutional layers allow for a reduction in the number of model parameters so that a deeper model can developed without significantly increasing the number of parameters in the overall model.

This repo consists of data, model and training modules.  The data.py file consists a function for generating the data split for training, testing and validation as well as the custom dataset class.  After splitting the data, the dataset can be stored in a dictionary as the following:

    poke_ds = {'train': data.PokeData('train'),
            'val': data.PokeData('val'),
            'test': data.PokeData('test')}

The ResNet model is implemented with the resnet.py file and contains class definitions for the different building blocks.  For example, the model can be instanced by the following:

    poke_detector = ResNet(in_channels = 3, num_classes = 150, block = BasicBlock, num_layers = [3, 4, 6, 3])

The training module is contained with train.py and the training hyperparameters are configured using the Trainer class.  It can be instanced with the relevant hyperparameters as:
    
    trainer = Trainer(poke_ds, poke_detector101.to('cuda'), device = 'cuda', epochs = 50, batch_size = 16, lr = 1.e-3, num_workers = 4)

Finally, the training can be initiated by calling the train method by:   
    
    trainer.train()


Model predictions from ResNet50 using pretrained weights (IMAGENET1K_V2) are shown below:
<img width="9515" height="1901" alt="image" src="https://github.com/user-attachments/assets/2aba2003-bace-4999-b92a-162c22ee7857" />
