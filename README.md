**PyTorch Implementation of ResNet**

This repo consists of an implementation of the ResNet model (He et. al 2015) using the PyTorch library.  The model was tested using a dataset of 7000 Pokemon (https://www.kaggle.com/datasets/lantian773030/pokemonclassification).  The model was also verified using the built-in versions of ResNet which achieved model accuracies of ~75%.  However, the accuracy improved significantly to ~95% when pretrained weights were used which suggests larger training datasets would be greatly beneficial to achieve more accurate results.

The ResNet is a modification of the original VGG-19 net with additional skip connections to help address the vanishing gradient problem.  Generally, networks become more difficult to train as the number of hidden layers is increased due to the vanishing gradient problem, where the computed gradient values become exponentially smaller as one progresses further through the network.  However, this problem has can be alleviated by introducing so-called 'shortcut connections', which recasts the training problem to the optimization of a residual mapping instead of a direct mapping between the model predictions and ground truth.  The shortcut connections are constructed by adding the input directly to the output so that the output is formally written as $\mathcal{F}(x) + x$ and are relatively straighforward to implement.  The only complexity that arises is when a mismatch between the input and output dimensions occurs, in which case an additional convolution layer (with kernel size 1x1) is applied to the identity shortcut to ensure that the mapping is dimensionally consistent.

Two different types of buil

This repo consists of data, model and training modules.  The data.py file consists a function for generating the data split for training, testing and validation as well as the custom dataset class.  After spitting the data, the dataset can be stored in a dictionary as the following:

    poke_ds = {'train': data.PokeData('train'),
            'val': data.PokeData('val'),
            'test': data.PokeData('test')}

The ResNet model is implmented with with the resnet.py file and contains definitions for the relevant building blocks.  For example, the ResNet can be instanced by the following:

    poke_detector = ResNet(in_channels = 3, num_classes = 150, block = BasicBlock, num_layers = [3, 4, 6, 3])

The training module is contained with train.py and the training schedule is defined within the Traininer class.  It can be instanced with the following hyperparameters as the following:
    
    trainer = Trainer(poke_ds, poke_detector101.to('cuda'), device = 'cuda', epochs = 50, batch_size = 16, lr = 1.e-3, num_workers = 4)

Finally, the training can be initaiated by calling the train method by:   
    
    trainer.train()
