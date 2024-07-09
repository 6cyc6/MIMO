import sys
import os
import configargparse
import torch
from torch.utils.data import DataLoader

from mimo.model.vnn_occ_net import VNNOccNet
from mimo.training import losses, training, dataio

p = configargparse.ArgumentParser()

p.add_argument('--logging_root', type=str, default='./logging', help='root for logging')
p.add_argument('--obj_class', type=str, required=True,
               help='hammer, container, cup, bottle, mug, bowl, all')
p.add_argument('--shapenet', action='store_true', help='if shapenet object')
p.add_argument('--experiment_name', type=str, required=True,
               help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')

p.add_argument('--sidelength', type=int, default=128)

# General training options
p.add_argument('--batch_size', type=int, default=32)
p.add_argument('--lr', type=float, default=1e-4, help='learning rate. default=5e-5')
p.add_argument('--num_epochs', type=int, default=40001,
               help='Number of epochs to train for.')

p.add_argument('--epochs_til_ckpt', type=int, default=80,
               help='Time interval in seconds until checkpoint is saved.')
p.add_argument('--steps_til_validation', type=int, default=500,
               help='Time interval in seconds until tensorboard summary is saved.')

p.add_argument('--depth_aug', action='store_true', help='depth_augmentation')
p.add_argument('--multiview_aug', action='store_true', help='multiview_augmentation')
p.add_argument('--single_view_aug', action='store_true', help='single_view_augmentation')

p.add_argument('--checkpoint_path', default=None, help='Checkpoint to trained model.')
p.add_argument('--dgcnn', action='store_true', help='If you want to use a DGCNN encoder instead of pointnet (requires more GPU memory)')

p.add_argument('--schedule', action='store_true', help='learning rate decay')

opt = p.parse_args()

if opt.shapenet:
    train_dataset = dataio.JointShapenetTrainDataset(opt.sidelength,
                                                     obj_class=opt.obj_class,
                                                     depth_aug=opt.depth_aug,
                                                     multiview_aug=opt.multiview_aug,
                                                     single_view=opt.single_view_aug)
    val_dataset = dataio.JointShapenetTrainDataset(opt.sidelength, phase='val',
                                                   obj_class=opt.obj_class,
                                                   depth_aug=opt.depth_aug,
                                                   multiview_aug=opt.multiview_aug,
                                                   single_view=opt.single_view_aug)
else:
    train_dataset = dataio.JointNonShapenetTrainDataset(opt.sidelength,
                                                        obj_class=opt.obj_class,
                                                        depth_aug=opt.depth_aug,
                                                        multiview_aug=opt.multiview_aug,
                                                        single_view=opt.single_view_aug)
    val_dataset = dataio.JointNonShapenetTrainDataset(opt.sidelength, phase='val',
                                                      obj_class=opt.obj_class,
                                                      depth_aug=opt.depth_aug,
                                                      multiview_aug=opt.multiview_aug,
                                                      single_view=opt.single_view_aug)


train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True,
                              drop_last=True, num_workers=8, pin_memory=True)
val_dataloader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=True,
                            drop_last=True, num_workers=4, pin_memory=True)

model = VNNOccNet(latent_dim=256, sigmoid=True).cuda()

if opt.checkpoint_path is not None:
    model.load_state_dict(torch.load(opt.checkpoint_path))

# model_parallel = nn.DataParallel(model, device_ids=[0, 1, 2, 3])
model_parallel = model
# print(model)

# Define the loss
root_path = os.path.join(opt.logging_root, opt.experiment_name)
loss_fn = val_loss_fn = losses.occupancy

training.train_occ(model=model_parallel, train_dataloader=train_dataloader, val_dataloader=val_dataloader,
                   epochs=opt.num_epochs, lr=opt.lr,
                   steps_til_validation=opt.steps_til_validation, epochs_til_checkpoint=opt.epochs_til_ckpt,
                   loss_fn=loss_fn, val_loss_fn=val_loss_fn, clip_grad=False,
                   lr_schedule=opt.schedule,
                   model_dir=root_path, overwrite=True)

