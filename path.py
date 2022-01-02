import os

# event_type = "pulmonary"
event_type = "yidu"

current_dir = os.path.dirname(os.path.abspath(__file__))  # 获取当前地址
proj_path = current_dir

weights_path = proj_path + "/weights"
f1_report_path = proj_path + "/report/f1.csv"
log_path = proj_path + "/log/train_log.csv"
fig_path = proj_path + "/images"
label_dict_path = proj_path + "/data/%s_catagory.pkl" % event_type
categories_f1_path = proj_path + "/report/categories_f1.csv"

# NER
train_file_path = proj_path + "/data/%s.train" % event_type
test_file_path = proj_path + "/data/%s.test" % event_type
val_file_path = proj_path + "/data/%s.validate" % event_type

# Model Config
MODEL_TYPE = 'albert'

BASE_MODEL_DIR = proj_path + "/albert_tiny_google_zh"
BASE_CONFIG_NAME = proj_path + "/albert_tiny_google_zh/albert_config.json"
BASE_CKPT_NAME = proj_path + "/albert_tiny_google_zh/albert_model.ckpt"
