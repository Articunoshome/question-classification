# Documentation for all the functions and classes
* src/data/datamodel.py
    
    The module contains all the codes required to store the dataset as Torch DataLoader.
    * `QuestionDataset` inherits the torch `Dataset` class and accepts the text tokens, offsets and labels as torch tensors and stores them as data members.The member functions `__getitem__ ` and `__len__` methods are essential to create a generator.
* src/NN/classifier.py

    The module contains the classes for all accommodating a single model, and ensembling it.
    * `Classifier` inherits torch `nn.Module` and it accepts all the hyperparameters as well as module configuration flags.
    The type of embeddings used and loading of pre trained embeddings is also handled in the constructor of the class.
        * The `set_learning_params` will help to set the learning rate and gamma on the go and to create the Stochastic gradient descent, optimiser, and learning scheduler based on these values.
        * The `forward` function accepts an instance of data set and passes it through all the layers and activation functions and returns the ouput of the final Linear layer
        * `train_func` propogated the data through the layers and then calculates the loss and accuracies for each batch and optimises the weights using the optimiser's backpropogation algorithm. Returns the corresponding batch level loss and accuracies.
        * `test` function will calculate the validation loss for the batch
        * Given a sequence of sentence tokens `predict` function will ifentify the most propbable class and returns its encoded value.
        * `fit` function will iterate and feed the batches to train_func and saves the best accurate model based on validation accuracy.
    * `Ensemble` class is used to create a stack of a number of models. The constructor expects a list of `Classifier` objects, and stores them.
        * The `fit` function will individually call the the fit functions of the models and feed the data to them. It returns the the average of training and validation loss and accuracy of all the models.
        * The `predict` function accepts the single token sequence and call the predict of all the models individually. Returns the most prominent class back.

* src/preprocess/categorical.py
    
    The module handles handling of categorical data.
    * `LabelEncoder` Class takes a list of categorical data and identifies the unique values and stores them, so that they can be used in an index encoded format.
        * `build_labels` function accepts the list of labels and identify the unique ones and store them.
        * `convert_labels_to_encodings` given a label return the index of the label from our vocab(Encoding)
        * `convert_encodings_to_labels` Given an encoding retrieve the label based on index

* src/preprocessor/text.py
  
  This module contains the code for preprocessing input data and building vocabulary from it.
  
  * `VocabBuilder` is a class that is used to build the text vocabulary. Missing words from the text vacabulary are handled in the constructor of the class itself.
    
    * `_basic_english_normalise` function is used to tokenise the input string
 
    * `find_n_gram` is used function to split the word into ngrams

    * `find_average_vector` function is used to find the average of the embeddings of the individual ngrams to get the embedding of the word.

    * `get_token_embedding` function is used to get the embedding of the tokens. If the token is present in the word vocabulary and its embedding is present in the embedding layer, then we will retrieve the embedding of the token from the layer. 
    Otherwise, if the token can be split into ngrams, then we will calculate the embedding as the average of the embeddings of the individual ngrams but if the token cannot be split further, then we will represent it as '#UNK#' and use the vector corresponding to the unknown token.

    * `build_vocab_from_iterator` function is used to build the vocabulary from the token list that is passed to it by the `build_vocab` function. It checks whether each word satisfies the minimum frequency criterion and then removes the stop words.

    * `build_vocab` is a wrapper function and is used to create the token list from the input list of text. Also it loads the pre trained embeddings into the memory.

    * `convert_sentences_to_encoding` function is used to get the encoded tokens from the list of input sentences.
 
    * `convert_encoding_to_sentences` function is used to get the list of sentences from the encoded sentence tokens