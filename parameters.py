from utility import get_args

config = get_args()
# ---------------- Pre-processing and loading--------------------------


text_size = 0.1
val_size = 0.1

# ---------------- directories and paths--------------------------
dataset = config.dataset
if dataset == 'auth':
    data_dir = '../project_data/authorship/'
elif dataset == 'ling':
    data_dir = '../project_data/linguistic_quality/'
elif dataset == 'sent':
    dir_name = '4-1-5 split' if config.original_split else '8-1-1 split'
    data_dir = '../project_data/sentiment/{}/'.format(dir_name)
elif dataset == 'yelp':
    data_dir = '../project_data/yelp/'
elif dataset == 'ag_news':
    data_dir = '../project_data/ag_news/'
three_parts_split = False

if config.use_paragraph:
        if config.fixed_length: train_test_val_data_dir = data_dir + 'train_val_test/fixed_length/'
        else: train_test_val_data_dir = data_dir + 'train_val_test/paragraph/'
else: train_test_val_data_dir = data_dir + 'train_val_test/document/'

word_seq_dir = train_test_val_data_dir + 'word_sequence/'
sent_split_dir = train_test_val_data_dir + 'sentence_split/3 parts/' if three_parts_split else train_test_val_data_dir + 'sentence_split/'

glove_dir = '../project_data/glove/'

dict_path = train_test_val_data_dir + 'dictionary.p'
glove_matrix_path = train_test_val_data_dir + 'weights_matrix.npy'

# ---------------- Training Parameters -----------------------
# training_data_file = "dataset_with_unk"
# 0: typical, 1: great and verygood

weighted_class = False

if config.debug:
    small_data = True
    batch_size = 3
    train_batch_size = batch_size
    test_batch_size = batch_size
    num_epoch = 2
    print_loss_every = 1
else:
    small_data = False
    batch_size = config.batch_size
    train_batch_size = batch_size
    test_batch_size = batch_size
    # num_epoch = 301 * 30
    num_epoch = config.num_epoch
    print_loss_every = 200
