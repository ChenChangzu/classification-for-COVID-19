class DefaultConfigs(object):
    # 1.路径参数d
    data_root = "/home/image/DataDisk/Data/CHEN/COVID_19/COVID_19_crop_data"
    model_name = "ResNet18"
    weights = "./checkpoints/"
    best_models = weights + "best_model/"
    submit = "./figures/"
    logs = "./logs/"
    gpus = "0"
    augmen_level = "none"  # "none","light","medium","hard","hard2"

    # 2.训练参数
    fold = 42
    epochs = 200
    batch_size = 16
    patience = 200
    class_list = ["COVID_19", "NOCOVID_19","Health"]  # ["COVID_19", "NOCOVID_19", "Health"]
    seed = 888
    lr = 1e-4
    step_size = 30
    gamma = 0.1
    lr_decay = 1e-4
    momentum = 0.9
    weight_decay = 1e-4

    # 3.尺寸参数
    dpi = 600
    img_height = 256
    img_weight = 256

config = DefaultConfigs()
