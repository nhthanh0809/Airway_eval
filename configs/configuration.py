class Config():


    data_version = 'v1.0'
    root_data_dir = '/data/Airway/v1.0/'
    root_experiment_name = '_final
    visualization = False
    visualize_output_on_training = False
    transform = False
    multi_gpus = False
    n_gpu = 2
    gpu_id = 0
    resume = False
    base_number = 40
    resize_h = 512
    resize_w = 480
    sigma = 10
    loss_weight = [0.8,0.8, 10.0] ####### [landmark_loss, landmark_loss_refine, segment_loss]
    num_epochs = 60
    lr = 1e-3
    lr_step_milestones = [20,40]
    # lr_step_milestones = [0,1,2]
    debug_steps = 10
    validation_step = 5
    batch_size = 1
    save_weight_every_epoch = 10
