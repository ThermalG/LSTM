from utils.const import META
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle

mplstyle.use('ggplot')

EPSILON = 1e-8


def load_dataset(index, norm_ts=False, verbose=True) -> tuple:
    """
    Loads a UCR Dataset indexed in META.

    Args:
        index: Integer. corresponds to line number in META.
        norm_ts: Bool. Determines whether to normalize the timeseries.
            If False, does not normalize the time series.
            If True, performs sample-wise normalization.
        verbose: Whether to describe the dataset being loaded.

    Returns:
        A tuple of shape (X_train, y_train, X_test, y_test).
    """
    assert index < len(META), "Index invalid. Could not load dataset at %d" % index
    if verbose:
        print("Loading train / test dataset...")

    if os.path.exists(META[index]['TrnPath']):
        df = pd.read_csv(META[index]['TrnPath'], header=None, encoding='latin-1')

    elif os.path.exists(META[index]['TrnPath'][1:]):
        df = pd.read_csv(META[index]['TrnPath'][1:], header=None, encoding='latin-1')

    else:
        raise FileNotFoundError('File %s not found!' % (META[index]['TrnPath']))
    
    # extract training labels and normalize them to [0, num_classes - 1]
    y_train = df[[0]].values
    num_classes = len(np.unique(y_train))
    y_train = (y_train - y_train.min()) / (y_train.max() - y_train.min()) * (num_classes - 1)
    
    # extract training data
    X_train = df.values[:, 1:]
    X_train = X_train[:, np.newaxis, :]

    # if required, normalize sequence values within each training sample
    if norm_ts:
        X_train_mean = np.mean(X_train, axis=-1, keepdims=True)
        X_train_std = np.std(X_train, axis=-1, keepdims=True)
        X_train = (X_train - X_train_mean) / (X_train_std + EPSILON)

    if verbose:
        print("Train dataset loaded.")

    if os.path.exists(META[index]['TstPath']):
        df = pd.read_csv(META[index]['TstPath'], header=None, encoding='latin-1')

    elif os.path.exists(META[index]['TstPath'][1:]):
        df = pd.read_csv(META[index]['TstPath'][1:], header=None, encoding='latin-1')
    else:
        raise FileNotFoundError('File %s not found!' % (META[index]['TstPath']))

    # extract testing labels and normalize them to [0, num_classes - 1]
    y_test = df[[0]].values
    num_classes = len(np.unique(y_test))
    y_test = (y_test - y_test.min()) / (y_test.max() - y_test.min()) * (num_classes - 1)

    # extract testing data
    X_test = df.values[:, 1:]
    X_test = X_test[:, np.newaxis, :]

    # if required, normalize sequence values within each testing sample
    if norm_ts:
        X_test_mean = np.mean(X_test, axis=-1, keepdims=True)
        X_test_std = np.std(X_test, axis=-1, keepdims=True)
        X_test = (X_test - X_test_mean) / (X_test_std + EPSILON)

    # output dataset info
    if verbose:
        print("Test dataset loaded.")
        print()
        print("Number of training samples: ", X_train.shape[0], "Number of testing samples: ", X_test.shape[0])
        print("Number of classes: ", num_classes, "Sequence length: ", X_train.shape[-1])
        print()

    return X_train, y_train, X_test, y_test


def plot_dataset(dataset_id, seed=None, limit=None, cutoff=None,
                 normalize_timeseries=False, plot_data=None,
                 type='Context', plot_classwise=False):
    """
    Util method to plot a dataset under several possibilities.

    Args:
        dataset_id: Integer id, refering to the dataset set inside
            `utils/const.py`.
        seed: Numpy Random seed.
        limit: Number of data points to be visualized. Min of 1.
        cutoff: Optional integer which slices of the first `cutoff` timesteps
            from the input signal.
        normalize_timeseries: Bool / Integer. Determines whether to normalize
            the timeseries.

            If False, does not normalize the time series.
            If True / int not equal to 2, performs standard sample-wise
                z-normalization.
            If 2: Performs full dataset z-normalization.
        plot_data: Additional data used for plotting in place of the
            loaded train set. Can be the test set or some other val set.
        type: Type of plot being built. Can be one of ['Context', any other string].
            Context is a specific keyword, used for Context from Attention LSTM.
            If any other string is provided, it is used in the title.
        plot_classwise: Bool flag. Wheter to visualize the samples
            seperated by class. When doing so, `limit` is multiplied by
            the number of classes so it is better to set `limit` to 1 in
            such cases
    """
    np.random.seed(seed)

    if plot_data is None:
        X_train, y_train, X_test, y_test = load_dataset(
            dataset_id,
            norm_ts=normalize_timeseries)

        sequence_length = X_train.shape[-1]

        if sequence_length != META[dataset_id]['Length']:
            if cutoff is None:
                choice = cutoff_choice(dataset_id, sequence_length)
            else:
                assert cutoff in ['pre', 'post'], 'Cutoff parameter value must be either "pre" or "post"'
                choice = cutoff

            if choice not in ['pre', 'post']:
                return
            else:
                X_train, X_test = X_test(X_train, X_test, choice, dataset_id, sequence_length)

        X_train_attention = None
        X_test_attention = None

    else:
        X_train, y_train, X_test, y_test, X_train_attention, X_test_attention = plot_data

    if limit is None:
        train_size = X_train.shape[0]
        test_size = X_test.shape[0]
    else:
        if not plot_classwise:
            train_size = limit
            test_size = limit
        else:
            assert limit == 1, 'If plotting classwise, limit must be 1 so as to ensure number of samples per class = 1'
            train_size = META[dataset_id]['Class'] * limit
            test_size = META[dataset_id]['Class'] * limit

    if not plot_classwise:
        train_idx = np.random.randint(0, X_train.shape[0], size=train_size)
        X_train = X_train[train_idx, 0, :]
        X_train = X_train.transpose((1, 0))

        if X_train_attention is not None:
            X_train_attention = X_train_attention[train_idx, 0, :]
            X_train_attention = X_train_attention.transpose((1, 0))
    else:
        classwise_train_list = []
        for y_ in sorted(np.unique(y_train[:, 0])):
            class_train_idx = np.where(y_train[:, 0] == y_)
            classwise_train_list.append(class_train_idx[:])

        classwise_sample_size_list = [len(x[0]) for x in classwise_train_list]
        size = min(classwise_sample_size_list)
        train_size = min([train_size // META[dataset_id]['Class'], size])

        for i in range(len(classwise_train_list)):
            classwise_train_idx = np.random.randint(0, len(classwise_train_list[i][0]), size=train_size)
            classwise_train_list[i] = classwise_train_list[i][0][classwise_train_idx]

        classwise_X_train_list = []
        classwise_X_train_attention_list = []

        for classwise_train_idx in classwise_train_list:
            classwise_X = X_train[classwise_train_idx, 0, :]
            classwise_X = classwise_X.transpose((1, 0))
            classwise_X_train_list.append(classwise_X)

            if X_train_attention is not None:
                classwise_X_attn = X_train_attention[classwise_train_idx, 0, :]
                classwise_X_attn = classwise_X_attn.transpose((1, 0))
                classwise_X_train_attention_list.append(classwise_X_attn)

        classwise_X_train_list = [np.asarray(x) for x in classwise_X_train_list]
        classwise_X_train_attention_list = [np.asarray(x) for x in classwise_X_train_attention_list]

        # classwise x train
        X_train = np.concatenate(classwise_X_train_list, axis=-1)

        # classwise x train attention
        if X_train_attention is not None:
            X_train_attention = np.concatenate(classwise_X_train_attention_list, axis=-1)

    if not plot_classwise:
        test_idx = np.random.randint(0, X_test.shape[0], size=test_size)
        X_test = X_test[test_idx, 0, :]
        X_test = X_test.transpose((1, 0))

        if X_test_attention is not None:
            X_test_attention = X_test_attention[test_idx, 0, :]
            X_test_attention = X_test_attention.transpose((1, 0))
    else:
        classwise_test_list = []
        for y_ in sorted(np.unique(y_test[:, 0])):
            class_test_idx = np.where(y_test[:, 0] == y_)
            classwise_test_list.append(class_test_idx[:])

        classwise_sample_size_list = [len(x[0]) for x in classwise_test_list]
        size = min(classwise_sample_size_list)
        test_size = min([test_size // META[dataset_id]['Class'], size])

        for i in range(len(classwise_test_list)):
            classwise_test_idx = np.random.randint(0, len(classwise_test_list[i][0]), size=test_size)
            classwise_test_list[i] = classwise_test_list[i][0][classwise_test_idx]

        classwise_X_test_list = []
        classwise_X_test_attention_list = []

        for classwise_test_idx in classwise_test_list:
            classwise_X = X_test[classwise_test_idx, 0, :]
            classwise_X = classwise_X.transpose((1, 0))
            classwise_X_test_list.append(classwise_X)

            if X_test_attention is not None:
                classwise_X_attn = X_test_attention[classwise_test_idx, 0, :]
                classwise_X_attn = classwise_X_attn.transpose((1, 0))
                classwise_X_test_attention_list.append(classwise_X_attn)

        classwise_X_test_list = [np.asarray(x) for x in classwise_X_test_list]
        classwise_X_test_attention_list = [np.asarray(x) for x in classwise_X_test_attention_list]

        # classwise x test
        X_test = np.concatenate(classwise_X_test_list, axis=-1)

        # classwise x test attention
        if X_test_attention is not None:
            X_test_attention = np.concatenate(classwise_X_test_attention_list, axis=-1)

    print('X_train shape : ', X_train.shape)
    print('X_test shape : ', X_test.shape)

    columns = ['Class %d' % (i + 1) for i in range(X_train.shape[1])]
    train_df = pd.DataFrame(X_train,
                            index=range(X_train.shape[0]),
                            columns=columns)

    test_df = pd.DataFrame(X_test,
                           index=range(X_test.shape[0]),
                           columns=columns)

    if plot_data is not None:
        rows = 2
        cols = 2
    else:
        rows = 1
        cols = 2

    fig, axs = plt.subplots(rows, cols, squeeze=False,
                            tight_layout=True, figsize=(8, 6))
    axs[0][0].set_title('Train dataset', size=16)
    axs[0][0].set_xlabel('timestep')
    axs[0][0].set_ylabel('value')
    train_df.plot(subplots=False,
                  legend='best',
                  ax=axs[0][0], )

    axs[0][1].set_title('Test dataset', size=16)
    axs[0][1].set_xlabel('timestep')
    axs[0][1].set_ylabel('value')
    test_df.plot(subplots=False,
                 legend='best',
                 ax=axs[0][1], )

    if plot_data is not None and X_train_attention is not None:
        columns = ['Class %d' % (i + 1) for i in range(X_train_attention.shape[1])]
        train_attention_df = pd.DataFrame(X_train_attention,
                                          index=range(X_train_attention.shape[0]),
                                          columns=columns)

        axs[1][0].set_title('Train %s Sequence' % (type), size=16)
        axs[1][0].set_xlabel('timestep')
        axs[1][0].set_ylabel('value')
        train_attention_df.plot(subplots=False,
                                legend='best',
                                ax=axs[1][0])

    if plot_data is not None and X_test_attention is not None:
        columns = ['Class %d' % (i + 1) for i in range(X_test_attention.shape[1])]
        test_df = pd.DataFrame(X_test_attention,
                               index=range(X_test_attention.shape[0]),
                               columns=columns)

        axs[1][1].set_title('Test %s Sequence' % (type), size=16)
        axs[1][1].set_xlabel('timestep')
        axs[1][1].set_ylabel('value')
        test_df.plot(subplots=False,
                     legend='best',
                     ax=axs[1][1])

    plt.show()


def cutoff_choice(dataset_id, sequence_length):
    """
    Helper to allow the user to select whether they want to cutoff timesteps or not,
    and in what manner (pre or post).

    Args:
        dataset_id: Dataset ID
        sequence_length: Length of the sequence originally.

    Returns:
        String choice of pre or post slicing.
    """
    print("Original sequence length was :", sequence_length, "New sequence Length will be : ",
          META[dataset_id]['Length'])
    choice = input('Options : \n'
                   '`pre` - cut the sequence from the beginning\n'
                   '`post`- cut the sequence from the end\n'
                   '`anything else` - stop execution\n'
                   'To automate choice: add flag `cutoff` = choice as above\n'
                   'Choice = ')

    choice = str(choice).lower()
    return choice


def cutoff_sequence(X_train, X_test, choice, dataset_id, sequence_length):
    """
    Slices of the first `cutoff` timesteps from the input signal.

    Args:
        X_train: Train sequences.
        X_test: Test sequences.
        choice: User's choice of slicing method.
        dataset_id: Integer id of the dataset set inside `utils/const.py`.
        sequence_length: Original length of the sequence.

    Returns:
        A tuple of (X_train, X_test) after slicing off the requisit number of
        timesteps.
    """
    assert META[dataset_id]['Length'] < sequence_length, "If sequence is to be cut, max sequence" \
                                                         "length must be less than original sequence length."
    cutoff = sequence_length - META[dataset_id]['Length']
    if choice == 'pre':
        if X_train is not None:
            X_train = X_train[:, :, cutoff:]
        if X_test is not None:
            X_test = X_test[:, :, cutoff:]
    else:
        if X_train is not None:
            X_train = X_train[:, :, :-cutoff]
        if X_test is not None:
            X_test = X_test[:, :, :-cutoff]
    print("New sequence length :", META[dataset_id]['Length'])
    return X_train, X_test