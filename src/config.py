data_root = '../medicaldata/images/CASIA2_16'

# 先生方によるラベル付与の分割データ
#train_info_list = '../medicaldata/txt/as_oct_train_preprocess.csv'
#test_info_list = '../medicaldata/txt/as_oct_test.csv'

# 12月発表時に使用したデータ
train_info_list = '../medicaldata/txt/casia16_train_list.csv'
test_info_list = '../medicaldata/txt/casia16_test_list.csv'

#pickleの位置
normal_pkl = '../medicaldata/pkls/OCT_ViT_spin.pkl'

mae_path = './model/MAE_1ep.pth'

MODEL_DIR_PATH = './model/'
LOG_DIR_PATH = './log/'
n_per_unit = 1
image_size = 224
n_class = 2

# train_info_list = '../medicaldata/txt/casia16_train_list.csv'
# test_info_list = '../medicaldata/txt/casia16_test_list.csv'