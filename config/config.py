# class Config(object):
#     def __init__(self, test_model_path='checkpoints/resnet18_110.pth',lfw_test_list='/data/Datasets/lfw/lfw_test_pair.txt'):
#         self.env = 'default'
#         self.backbone = 'resnet18'
#         self.classify = 'softmax'
#         self.num_classes = 14017
#         self.metric = 'arc_margin'
#         self.easy_margin = False
#         self.use_se = True
#         self.loss = 'con_loss'

#         self.display = False
#         self.finetune = False

#         self.train_root = '/data/Datasets/webface/CASIA-maxpy-clean-crop-144/'
#         self.train_list = '/data/Datasets/webface/train_data_13938.txt'
#         self.val_list = '/data/Datasets/webface/val_data_13938.txt'

#         self.test_root = '/data1/Datasets/anti-spoofing/test/data_align_256'
#         self.test_list = 'test.txt'

#         self.lfw_root = '/data/Datasets/lfw/lfw-align-128'
#         self.lfw_test_list = lfw_test_list

#         self.checkpoints_path = 'r100_cas'
#         self.load_model_path = 'models/resnet18.pth'
#         self.test_model_path = test_model_path
#         self.save_interval = 10

#         self.train_batch_size = 16 # batch size
#         self.test_batch_size = 64

#         self.input_shape = (1, 128, 128)

#         self.optimizer = 'adam'

#         self.use_gpu = True  # use GPU or not
#         self.gpu_id = '0, 1'
#         self.num_workers = 4  # how many workers for loading data
#         self.print_freq = 100  # print info every N batch

#         self.debug_file = '/tmp/debug'  # if os.path.exists(debug_file): enter ipdb
#         self.result_file = 'result.csv'

#         self.max_epoch = 50
#         self.lr = 1e-1  # initial learning rate
#         self.lr_step = 10
#         self.lr_decay = 0.95  # when val_loss increase, lr = lr*lr_decay
#         self.weight_decay = 5e-4
#         self.momentum = 0.95

class Config(object):
    def __init__(self, test_model_path='checkpoints/resnet18_110.pth', lfw_test_list='/data/Datasets/lfw/lfw_test_pair.txt'):
        self.env = 'default'
        self.backbone = 'resnet18'
        self.classify = 'softmax'
        
        # ===== UPDATED: Number of unique identities in your AgeDB dataset =====
        # You need to count unique identities from your CSV
        # Based on your CSV, you should have around 500-600 identities
        self.num_classes = 570  # This is the standard number for AgeDB
        # If you want exact number, run: pd.read_csv('dataset/AgeDB.csv')['identity'].nunique()
        
        self.metric = 'arc_margin'
        self.easy_margin = False
        self.use_se = True
        self.loss = 'con_loss'

        self.display = False
        self.finetune = False

        # ===== UPDATED: Training data paths =====
        self.train_root = '/home/tawfik/git/AQUAFace/dataset/AgeDB'
        self.train_list = '/home/tawfik/git/AQUAFace/train_data/train_pairs.txt'
        self.val_list = '/home/tawfik/git/AQUAFace/train_data/val_pairs.txt'

        # Test data (not used for training, only for validation)
        self.test_root = '/home/tawfik/git/AQUAFace/dataset/AgeDB'
        self.test_list = 'test.txt'

        # LFW validation (optional, can skip if you don't have LFW)
        self.lfw_root = '/data/Datasets/lfw/lfw-align-128'
        self.lfw_test_list = lfw_test_list

        # ===== UPDATED: Checkpoint save path =====
        self.checkpoints_path = '/home/tawfik/git/AQUAFace/checkpoints'
        
        # ===== UPDATED: Pretrained model path =====
        self.load_model_path = '/home/tawfik/git/AQUAFace/pretrained_models/R18_MS1MV3.onnx'
        self.test_model_path = test_model_path
        
        self.save_interval = 10

        # ===== Training hyperparameters =====
        self.train_batch_size = 16  # Adjust based on your GPU memory
        self.test_batch_size = 64

        self.input_shape = (3, 112, 112)  # RGB images, 112x112

        self.optimizer = 'adam'

        self.use_gpu = True
        self.gpu_id = '0'  # Single GPU
        self.num_workers = 4  # Data loading workers
        self.print_freq = 100  # Print info every N batch

        self.debug_file = '/tmp/debug'
        self.result_file = 'result.csv'

        # ===== Training schedule =====
        self.max_epoch = 50
        self.lr = 1e-3  # Initial learning rate (reduced for fine-tuning)
        self.lr_step = 10
        self.lr_decay = 0.95
        self.weight_decay = 5e-4
        self.momentum = 0.95