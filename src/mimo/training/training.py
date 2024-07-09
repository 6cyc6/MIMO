'''Implements a generic training loop.
'''

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm
import time
import numpy as np
import os
import shutil
from collections import defaultdict
import torch.distributed as dist

from . import util


def average_gradients(model):
    """Averages gradients across workers"""
    size = float(dist.get_world_size())

    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
            param.grad.data /= size


def multiscale_training(model, lr, steps_til_summary, epochs_til_checkpoint, model_dir, loss_fn,
                        dataloader_callback, dataloader_iters, dataloader_params,
                        val_loss_fn=None, summary_fn=None, iters_til_checkpoint=None, clip_grad=False,
                        overwrite=True, optimizers=None, batches_per_validation=10, gpus=1, rank=0):

    for params, max_steps in zip(dataloader_params, dataloader_iters):
        train_dataloader, val_dataloader = dataloader_callback(*params)
        model_dir = os.path.join(model_dir, '_'.join(map(str, params)))

        model, optimizers = train(model, train_dataloader, epochs=10000, lr=lr, steps_til_summary=steps_til_summary,
                                  val_dataloader=val_dataloader, epochs_til_checkpoint=epochs_til_checkpoint, model_dir=model_dir, loss_fn=loss_fn,
                                  val_loss_fn=val_loss_fn, summary_fn=summary_fn, iters_til_checkpoint=iters_til_checkpoint,
                                  clip_grad=clip_grad, overwrite=overwrite, optimizers=optimizers, batches_per_validation=batches_per_validation,
                                  gpus=gpus, rank=rank, max_steps=max_steps)


def train_mfim(model, train_dataloader, epochs, lr, steps_til_validation, epochs_til_checkpoint, model_dir, loss_fn,
               val_dataloader=None, clip_grad=False, val_loss_fn=None,
               overwrite=True, optimizers=None, batches_per_validation=20, gpus=1, rank=0,
               mtl=False, lw=0.5, max_steps=None, lr_schedule=False):

    if optimizers is None:
        if mtl:
            log_var_occ = torch.zeros((1,), requires_grad=True, device="cuda")
            log_var_sdf = torch.zeros((1,), requires_grad=True, device="cuda")
            log_var_scf = torch.zeros((1,), requires_grad=True, device="cuda")
            log_var_inner = torch.zeros((1,), requires_grad=True, device="cuda")
            # log_var_occ = torch.tensor([-2.596], requires_grad=True, device="cuda")
            # log_var_sdf = torch.tensor([-4.098], requires_grad=True, device="cuda")
            # log_var_scf = torch.tensor([-4.872], requires_grad=True, device="cuda")
            params = ([p for p in model.parameters()] + [log_var_occ] + [log_var_sdf] + [log_var_scf] + [log_var_inner])
            optimizers = [torch.optim.Adam(lr=lr, params=params)]
        else:
            optimizers = [torch.optim.Adam(lr=lr, params=model.parameters())]

    if val_dataloader is not None:
        assert val_loss_fn is not None, "If validation set is passed, have to pass a validation loss_fn!"

    # learning rate decay
    if lr_schedule:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizers[0], step_size=40, gamma=0.2)

    if rank == 0:
        if os.path.exists(model_dir):
            if overwrite:
                shutil.rmtree(model_dir)
            else:
                val = input("The model directory %s exists. Overwrite? (y/n)" % model_dir)
                if val == 'y' or overwrite:
                    shutil.rmtree(model_dir)

        os.makedirs(model_dir)

        summaries_dir = os.path.join(model_dir, 'summaries')
        util.cond_mkdir(summaries_dir)

        checkpoints_dir = os.path.join(model_dir, 'checkpoints')
        util.cond_mkdir(checkpoints_dir)

        writer = SummaryWriter(summaries_dir)

    val_loss_occ_min = 1
    total_steps = 0

    # start training
    with tqdm(total=len(train_dataloader) * epochs) as pbar:
        train_losses = []
        train_occ_losses = []
        train_sdf_losses = []
        train_scf_losses = []
        train_inner_losses = []
        for epoch in range(epochs):
            # save model and training loss after a fixed number of epochs
            if not epoch % epochs_til_checkpoint and epoch and rank == 0:
                torch.save(model.state_dict(),
                           os.path.join(checkpoints_dir, 'model_final.pth'))
                np.savetxt(os.path.join(checkpoints_dir, 'train_losses_total_final.txt'),
                           np.array(train_losses))
                np.savetxt(os.path.join(checkpoints_dir, 'train_losses_occ_final.txt'),
                           np.array(train_occ_losses))
                np.savetxt(os.path.join(checkpoints_dir, 'train_losses_sdf_final.txt'),
                           np.array(train_sdf_losses))
                np.savetxt(os.path.join(checkpoints_dir, 'train_losses_scf_final.txt'),
                           np.array(train_scf_losses))
                np.savetxt(os.path.join(checkpoints_dir, 'train_losses_inner_final.txt'),
                           np.array(train_inner_losses))

            for step, (model_input, gt) in enumerate(train_dataloader):
                model_input = util.dict_to_gpu(model_input)
                gt = util.dict_to_gpu(gt)

                start_time = time.time()

                model_output = model(model_input)
                losses = loss_fn(model_output, gt)
                # losses = loss_fn(model_output, gt, model_input, model)

                train_loss = 0.
                for loss_name, loss in losses.items():
                    single_loss = loss.mean()

                    # save non-weighted loss for each head
                    if loss_name == "occ":
                        train_occ_losses.append(single_loss.item())
                    elif loss_name == "sdf":
                        train_sdf_losses.append(single_loss.item())
                    elif loss_name == "scf":
                        train_scf_losses.append(single_loss.item())
                    else:
                        train_inner_losses.append(single_loss.item())

                    if rank == 0:
                        writer.add_scalar("train_" + loss_name, single_loss, total_steps)

                    # compute weighted loss
                    if mtl:
                        if loss_name == "occ":
                            single_loss = torch.exp(-log_var_occ) * single_loss + log_var_occ
                        elif loss_name == "sdf":
                            single_loss = torch.exp(-log_var_sdf) * single_loss + log_var_sdf
                        elif loss_name == "scf":
                            single_loss = torch.exp(-log_var_scf) * single_loss + log_var_scf
                        else:
                            single_loss = torch.exp(-log_var_inner) * single_loss + log_var_inner
                    else:
                        if loss_name == "occ":
                            single_loss *= lw
                        if loss_name == "sdf":
                            single_loss *= lw
                        elif loss_name == "scf":
                            single_loss *= lw
                        else:
                            single_loss *= (1 - 3 * lw)

                    # write summery
                    # if rank == 0:
                    #     writer.add_scalar("train_weighted_" + loss_name, single_loss, total_steps)
                    train_loss += single_loss

                # save total weighted loss and write summery
                train_losses.append(train_loss.item())
                if rank == 0:
                    writer.add_scalar("total_train_loss", train_loss, total_steps)

                # calculate gradient
                for optim in optimizers:
                    optim.zero_grad()
                train_loss.backward()

                if gpus > 1:
                    average_gradients(model)

                if clip_grad:
                    if isinstance(clip_grad, bool):
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)

                for optim in optimizers:
                    optim.step()

                if rank == 0:
                    pbar.update(1)

                # run validation
                if not total_steps % steps_til_validation and rank == 0:
                    print("Epoch %d, Total loss %0.6f, iteration time %0.6f" % (epoch, train_loss, time.time() - start_time))
                    loss_sum = 0
                    if val_dataloader is not None:
                        print("Running validation set...")
                        with torch.no_grad():
                            model.eval()
                            val_losses = defaultdict(list)
                            for val_i, (model_input_i, gt_i) in enumerate(val_dataloader):
                                model_input_i = util.dict_to_gpu(model_input_i)
                                gt_i = util.dict_to_gpu(gt_i)

                                model_output_i = model(model_input_i)
                                val_loss = val_loss_fn(model_output_i, gt_i, val=True)

                                for name, value in val_loss.items():
                                    val_losses[name].append(value.cpu().numpy())

                                if val_i == batches_per_validation:
                                    break

                        for loss_name, loss in val_losses.items():
                            single_loss = np.mean(loss)

                            # save validation loss for each head
                            writer.add_scalar('val_' + loss_name, single_loss, total_steps)

                            if loss_name == "occ":
                                print(f"occ_loss: {single_loss}")
                            elif loss_name == "sdf":
                                print(f"sdf_loss: {single_loss}")
                            elif loss_name == "scf":
                                print(f"scf_loss: {single_loss}")
                            else:
                                print(f"inner_loss: {single_loss}")

                            # save best model with the lowest occupancy validation loss
                            if loss_name == "scf" and val_loss_occ_min > single_loss:
                                val_loss_occ_min = single_loss
                                torch.save(model.state_dict(),
                                           os.path.join(checkpoints_dir, 'best_model_epoch_%04d_iter_%06d.pth' % (epoch, total_steps)))
                                # np.savetxt(os.path.join(checkpoints_dir, 'train_losses_%04d_iter_%06d.pth' % (epoch, total_steps)),
                                #            np.array(train_losses))

                            # compute weighted validation loss
                            if mtl:
                                if loss_name == "occ":
                                    single_loss = torch.exp(-log_var_occ) * single_loss + log_var_occ
                                    print(f"log_var_occ: {log_var_occ.detach().cpu().numpy()}")
                                elif loss_name == "sdf":
                                    single_loss = torch.exp(-log_var_sdf) * single_loss + log_var_sdf
                                    print(f"log_var_sdf: {log_var_sdf.detach().cpu().numpy()}")
                                elif loss_name == "scf":
                                    single_loss = torch.exp(-log_var_scf) * single_loss + log_var_scf
                                    print(f"log_var_scf: {log_var_scf.detach().cpu().numpy()}")
                                else:
                                    single_loss = torch.exp(-log_var_inner) * single_loss + log_var_inner
                                    print(f"log_var_inner: {log_var_inner.detach().cpu().numpy()}")
                            else:
                                if loss_name == "occ":
                                    single_loss *= lw
                                if loss_name == "sdf":
                                    single_loss *= lw
                                elif loss_name == "scf":
                                    single_loss *= lw
                                else:
                                    single_loss *= (1 - 3 * lw)

                            if loss_name == "occ":
                                print(f"weighted_occ_loss: {single_loss.item()}")
                            elif loss_name == "sdf":
                                print(f"weighted_sdf_loss: {single_loss.item()}")
                            elif loss_name == "scf":
                                print(f"weighted_scf_loss: {single_loss.item()}")
                            else:
                                print(f"weighted_inner_loss: {single_loss.item()}")

                            # save weighted validation loss
                            # writer.add_scalar('val_weighted' + loss_name, single_loss, total_steps)

                            loss_sum += single_loss.item()

                        print(f"total_loss: {loss_sum}")
                        model.train()

                total_steps += 1
                if max_steps is not None and total_steps == max_steps:
                    break

            if max_steps is not None and total_steps == max_steps:
                break

            if lr_schedule:
                scheduler.step()

        # save final model
        torch.save(model.state_dict(),
                   os.path.join(checkpoints_dir, 'model_final.pth'))
        np.savetxt(os.path.join(checkpoints_dir, 'train_losses_total_final.txt'),
                   np.array(train_losses))
        np.savetxt(os.path.join(checkpoints_dir, 'train_losses_occ_final.txt'),
                   np.array(train_occ_losses))
        np.savetxt(os.path.join(checkpoints_dir, 'train_losses_sdf_final.txt'),
                   np.array(train_sdf_losses))
        np.savetxt(os.path.join(checkpoints_dir, 'train_losses_scf_final.txt'),
                   np.array(train_scf_losses))
        np.savetxt(os.path.join(checkpoints_dir, 'train_losses_inner_final.txt'),
                   np.array(train_scf_losses))

        return model, optimizers


def train_occ_sdf_scf(model, train_dataloader, epochs, lr, steps_til_validation, epochs_til_checkpoint, model_dir, loss_fn,
                      val_dataloader=None, clip_grad=False, val_loss_fn=None,
                      overwrite=True, optimizers=None, batches_per_validation=20, gpus=1, rank=0,
                      mtl=False, lw=0.5, max_steps=None, lr_schedule=False):

    if optimizers is None:
        if mtl:
            log_var_occ = torch.zeros((1,), requires_grad=True, device="cuda")
            log_var_sdf = torch.zeros((1,), requires_grad=True, device="cuda")
            log_var_scf = torch.zeros((1,), requires_grad=True, device="cuda")
            # log_var_occ = torch.tensor([-2.596], requires_grad=True, device="cuda")
            # log_var_sdf = torch.tensor([-4.098], requires_grad=True, device="cuda")
            # log_var_scf = torch.tensor([-4.872], requires_grad=True, device="cuda")
            params = ([p for p in model.parameters()] + [log_var_occ] + [log_var_sdf] + [log_var_scf])
            optimizers = [torch.optim.Adam(lr=lr, params=params)]
        else:
            optimizers = [torch.optim.Adam(lr=lr, params=model.parameters())]

    if val_dataloader is not None:
        assert val_loss_fn is not None, "If validation set is passed, have to pass a validation loss_fn!"

    # learning rate decay
    if lr_schedule:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizers[0], step_size=80, gamma=0.2)

    if rank == 0:
        if os.path.exists(model_dir):
            if overwrite:
                shutil.rmtree(model_dir)
            else:
                val = input("The model directory %s exists. Overwrite? (y/n)" % model_dir)
                if val == 'y' or overwrite:
                    shutil.rmtree(model_dir)

        os.makedirs(model_dir)

        summaries_dir = os.path.join(model_dir, 'summaries')
        util.cond_mkdir(summaries_dir)

        checkpoints_dir = os.path.join(model_dir, 'checkpoints')
        util.cond_mkdir(checkpoints_dir)

        writer = SummaryWriter(summaries_dir)

    val_loss_occ_min = 1
    total_steps = 0

    # start training
    with tqdm(total=len(train_dataloader) * epochs) as pbar:
        train_losses = []
        train_occ_losses = []
        train_sdf_losses = []
        train_scf_losses = []
        for epoch in range(epochs):
            # save model and training loss after a fixed number of epochs
            if not epoch % epochs_til_checkpoint and epoch and rank == 0:
                torch.save(model.state_dict(),
                           os.path.join(checkpoints_dir, 'model_final.pth'))
                np.savetxt(os.path.join(checkpoints_dir, 'train_losses_total_final.txt'),
                           np.array(train_losses))
                np.savetxt(os.path.join(checkpoints_dir, 'train_losses_occ_final.txt'),
                           np.array(train_occ_losses))
                np.savetxt(os.path.join(checkpoints_dir, 'train_losses_sdf_final.txt'),
                           np.array(train_sdf_losses))
                np.savetxt(os.path.join(checkpoints_dir, 'train_losses_scf_final.txt'),
                           np.array(train_scf_losses))

            for step, (model_input, gt) in enumerate(train_dataloader):
                model_input = util.dict_to_gpu(model_input)
                gt = util.dict_to_gpu(gt)

                start_time = time.time()

                model_output = model(model_input)
                losses = loss_fn(model_output, gt)
                # losses = loss_fn(model_output, gt, model_input, model)

                train_loss = 0.
                for loss_name, loss in losses.items():
                    single_loss = loss.mean()

                    # save non-weighted loss for each head
                    if loss_name == "occ":
                        train_occ_losses.append(single_loss.item())
                    elif loss_name == "sdf":
                        train_sdf_losses.append(single_loss.item())
                    else:
                        train_scf_losses.append(single_loss.item())

                    if rank == 0:
                        writer.add_scalar("train_" + loss_name, single_loss, total_steps)

                    # compute weighted loss
                    if mtl:
                        if loss_name == "occ":
                            single_loss = torch.exp(-log_var_occ) * single_loss + log_var_occ
                        elif loss_name == "sdf":
                            single_loss = torch.exp(-log_var_sdf) * single_loss + log_var_sdf
                        else:
                            single_loss = torch.exp(-log_var_scf) * single_loss + log_var_scf
                    else:
                        if loss_name == "occ":
                            single_loss *= lw
                        if loss_name == "sdf":
                            single_loss *= lw
                        else:
                            single_loss *= (1 - lw * 2)

                    # write summery
                    # if rank == 0:
                    #     writer.add_scalar("train_weighted_" + loss_name, single_loss, total_steps)
                    train_loss += single_loss

                # save total weighted loss and write summery
                train_losses.append(train_loss.item())
                if rank == 0:
                    writer.add_scalar("total_train_loss", train_loss, total_steps)

                # calculate gradient
                for optim in optimizers:
                    optim.zero_grad()
                train_loss.backward()

                if gpus > 1:
                    average_gradients(model)

                if clip_grad:
                    if isinstance(clip_grad, bool):
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)

                for optim in optimizers:
                    optim.step()

                if rank == 0:
                    pbar.update(1)

                # run validation
                if not total_steps % steps_til_validation and rank == 0:
                    print("Epoch %d, Total loss %0.6f, iteration time %0.6f" % (epoch, train_loss, time.time() - start_time))
                    loss_sum = 0
                    if val_dataloader is not None:
                        print("Running validation set...")
                        with torch.no_grad():
                            model.eval()
                            val_losses = defaultdict(list)
                            for val_i, (model_input_i, gt_i) in enumerate(val_dataloader):
                                model_input_i = util.dict_to_gpu(model_input_i)
                                gt_i = util.dict_to_gpu(gt_i)

                                model_output_i = model(model_input_i)
                                val_loss = val_loss_fn(model_output_i, gt_i, val=True)

                                for name, value in val_loss.items():
                                    val_losses[name].append(value.cpu().numpy())

                                if val_i == batches_per_validation:
                                    break

                        for loss_name, loss in val_losses.items():
                            single_loss = np.mean(loss)

                            # save validation loss for each head
                            writer.add_scalar('val_' + loss_name, single_loss, total_steps)

                            if loss_name == "occ":
                                print(f"occ_loss: {single_loss}")
                            elif loss_name == "sdf":
                                print(f"sdf_loss: {single_loss}")
                            else:
                                print(f"scf_loss: {single_loss}")

                            # save best model with the lowest occupancy validation loss
                            if loss_name == "scf" and val_loss_occ_min > single_loss:
                                val_loss_occ_min = single_loss
                                torch.save(model.state_dict(),
                                           os.path.join(checkpoints_dir, 'best_model_epoch_%04d_iter_%06d.pth' % (epoch, total_steps)))
                                # np.savetxt(os.path.join(checkpoints_dir, 'train_losses_%04d_iter_%06d.pth' % (epoch, total_steps)),
                                #            np.array(train_losses))

                            # compute weighted validation loss
                            if mtl:
                                if loss_name == "occ":
                                    single_loss = torch.exp(-log_var_occ) * single_loss + log_var_occ
                                    print(f"log_var_occ: {log_var_occ.detach().cpu().numpy()}")
                                elif loss_name == "sdf":
                                    single_loss = torch.exp(-log_var_sdf) * single_loss + log_var_sdf
                                    print(f"log_var_sdf: {log_var_sdf.detach().cpu().numpy()}")
                                else:
                                    single_loss = torch.exp(-log_var_scf) * single_loss + log_var_scf
                                    print(f"log_var_scf: {log_var_scf.detach().cpu().numpy()}")
                            else:
                                if loss_name == "occ":
                                    single_loss *= lw
                                if loss_name == "sdf":
                                    single_loss *= lw
                                else:
                                    single_loss *= (1 - lw * 2)

                            if loss_name == "occ":
                                print(f"weighted_occ_loss: {single_loss.item()}")
                            elif loss_name == "sdf":
                                print(f"weighted_sdf_loss: {single_loss.item()}")
                            else:
                                print(f"weighted_scf_loss: {single_loss.item()}")

                            # save weighted validation loss
                            # writer.add_scalar('val_weighted' + loss_name, single_loss, total_steps)

                            loss_sum += single_loss.item()

                        print(f"total_loss: {loss_sum}")
                        model.train()

                total_steps += 1
                if max_steps is not None and total_steps == max_steps:
                    break

            if max_steps is not None and total_steps == max_steps:
                break

            if lr_schedule:
                scheduler.step()

        # save final model
        torch.save(model.state_dict(),
                   os.path.join(checkpoints_dir, 'model_final.pth'))
        np.savetxt(os.path.join(checkpoints_dir, 'train_losses_total_final.txt'),
                   np.array(train_losses))
        np.savetxt(os.path.join(checkpoints_dir, 'train_losses_occ_final.txt'),
                   np.array(train_occ_losses))
        np.savetxt(os.path.join(checkpoints_dir, 'train_losses_sdf_final.txt'),
                   np.array(train_sdf_losses))
        np.savetxt(os.path.join(checkpoints_dir, 'train_losses_scf_final.txt'),
                   np.array(train_scf_losses))

        return model, optimizers


def train_sdf_scf(model, train_dataloader, epochs, lr, steps_til_validation, epochs_til_checkpoint, model_dir, loss_fn,
                  val_dataloader=None, clip_grad=False, val_loss_fn=None,
                  overwrite=True, optimizers=None, batches_per_validation=10, gpus=1, rank=0,
                  mtl=False, lw=0.5, max_steps=None, lr_schedule=False):

    if optimizers is None:
        if mtl:
            log_var_sdf = torch.zeros((1,), requires_grad=True, device="cuda")
            log_var_scf = torch.zeros((1,), requires_grad=True, device="cuda")
            params = ([p for p in model.parameters()] + [log_var_sdf] + [log_var_scf])
            optimizers = [torch.optim.Adam(lr=lr, params=params)]
        else:
            optimizers = [torch.optim.Adam(lr=lr, params=model.parameters())]

    if val_dataloader is not None:
        assert val_loss_fn is not None, "If validation set is passed, have to pass a validation loss_fn!"

    if lr_schedule:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizers[0], step_size=80, gamma=0.2)

    if rank == 0:
        if os.path.exists(model_dir):
            if overwrite:
                shutil.rmtree(model_dir)
            else:
                val = input("The model directory %s exists. Overwrite? (y/n)" % model_dir)
                if val == 'y' or overwrite:
                    shutil.rmtree(model_dir)

        os.makedirs(model_dir)

        summaries_dir = os.path.join(model_dir, 'summaries')
        util.cond_mkdir(summaries_dir)

        checkpoints_dir = os.path.join(model_dir, 'checkpoints')
        util.cond_mkdir(checkpoints_dir)

        writer = SummaryWriter(summaries_dir)

    val_loss_sdf_min = 1
    total_steps = 0

    # start training
    with tqdm(total=len(train_dataloader) * epochs) as pbar:
        train_losses = []
        train_sdf_losses = []
        train_scf_losses = []
        for epoch in range(epochs):
            # save model and training loss after a fixed number of epochs
            if not epoch % epochs_til_checkpoint and epoch and rank == 0:
                torch.save(model.state_dict(),
                           os.path.join(checkpoints_dir, 'model_epoch_%04d_iter_%06d.pth' % (epoch, total_steps)))
                np.savetxt(os.path.join(checkpoints_dir, 'train_losses_total_final.txt'),
                           np.array(train_losses))
                np.savetxt(os.path.join(checkpoints_dir, 'train_losses_sdf_final.txt'),
                           np.array(train_sdf_losses))
                np.savetxt(os.path.join(checkpoints_dir, 'train_losses_scf_final.txt'),
                           np.array(train_scf_losses))

            for step, (model_input, gt) in enumerate(train_dataloader):
                model_input = util.dict_to_gpu(model_input)
                gt = util.dict_to_gpu(gt)

                start_time = time.time()

                model_output = model(model_input)
                losses = loss_fn(model_output, gt)
                # losses = loss_fn(model_output, gt, model_input, model)

                train_loss = 0.
                for loss_name, loss in losses.items():
                    single_loss = loss.mean()

                    # save non-weighted loss for each head
                    if loss_name == "sdf":
                        train_sdf_losses.append(single_loss.item())
                    else:
                        train_scf_losses.append(single_loss.item())

                    if rank == 0:
                        writer.add_scalar("train_" + loss_name, single_loss, total_steps)

                    # compute weighted loss
                    if mtl:
                        if loss_name == "sdf":
                            single_loss = torch.exp(-log_var_sdf) * single_loss + log_var_sdf
                        else:
                            single_loss = torch.exp(-log_var_scf) * single_loss + log_var_scf
                    else:
                        if loss_name == "sdf":
                            single_loss *= lw
                        else:
                            single_loss *= (1 - lw)

                    # if rank == 0:
                    #     writer.add_scalar("train_weighted_" + loss_name, single_loss, total_steps)
                    train_loss += single_loss

                # save total weighted loss and write summery
                train_losses.append(train_loss.item())
                if rank == 0:
                    writer.add_scalar("total_train_loss", train_loss, total_steps)

                # calculate gradient
                for optim in optimizers:
                    optim.zero_grad()
                train_loss.backward()

                if gpus > 1:
                    average_gradients(model)

                if clip_grad:
                    if isinstance(clip_grad, bool):
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)

                for optim in optimizers:
                    optim.step()

                if rank == 0:
                    pbar.update(1)

                if not total_steps % steps_til_validation and rank == 0:
                    print("Epoch %d, Total loss %0.6f, iteration time %0.6f" % (epoch, train_loss, time.time() - start_time))
                    loss_sum = 0
                    if val_dataloader is not None:
                        print("Running validation set...")
                        with torch.no_grad():
                            model.eval()
                            val_losses = defaultdict(list)
                            for val_i, (model_input_i, gt_i) in enumerate(val_dataloader):
                                model_input_i = util.dict_to_gpu(model_input_i)
                                gt_i = util.dict_to_gpu(gt_i)

                                model_output_i = model(model_input_i)
                                val_loss = val_loss_fn(model_output_i, gt_i, val=True)

                                for name, value in val_loss.items():
                                    val_losses[name].append(value.cpu().numpy())

                                if val_i == batches_per_validation:
                                    break

                        for loss_name, loss in val_losses.items():
                            single_loss = np.mean(loss)

                            # save validation loss for each head
                            writer.add_scalar('val_' + loss_name, single_loss, total_steps)

                            if loss_name == "sdf":
                                print(f"sdf_loss: {single_loss}")
                            else:
                                print(f"scf_loss: {single_loss}")

                            # save best model with the lowest occupancy validation loss
                            if loss_name == "sdf" and val_loss_sdf_min > single_loss:
                                val_loss_sdf_min = single_loss
                                torch.save(model.state_dict(),
                                           os.path.join(checkpoints_dir, 'best_model_epoch_%04d_iter_%06d.pth' % (epoch, total_steps)))
                                # np.savetxt(os.path.join(checkpoints_dir, 'train_losses_%04d_iter_%06d.pth' % (epoch, total_steps)),
                                #            np.array(train_losses))

                            # compute weighted validation loss
                            if mtl:
                                if loss_name == "sdf":
                                    log_var_sdf_np = log_var_sdf.detach().cpu().numpy()
                                    single_loss = np.exp(-log_var_sdf_np) * single_loss + log_var_sdf_np
                                else:
                                    log_var_scf_np = log_var_scf.detach().cpu().numpy()
                                    single_loss = np.exp(-log_var_scf_np) * single_loss + log_var_scf_np
                            else:
                                if loss_name == "sdf":
                                    single_loss *= lw
                                else:
                                    single_loss *= (1 - lw)

                            if loss_name == "sdf":
                                print(f"weighted_sdf_loss: {single_loss}")
                            else:
                                print(f"weighted_scf_loss: {single_loss}")

                            # save weighted validation loss
                            # writer.add_scalar('val_weighted' + loss_name, single_loss, total_steps)

                            loss_sum += single_loss

                        print(f"total_loss: {loss_sum}")
                        model.train()

                total_steps += 1
                if max_steps is not None and total_steps == max_steps:
                    break

            if max_steps is not None and total_steps == max_steps:
                break

            if lr_schedule:
                scheduler.step()

        # save final model
        torch.save(model.state_dict(),
                   os.path.join(checkpoints_dir, 'model_final.pth'))
        np.savetxt(os.path.join(checkpoints_dir, 'train_losses_total_final.txt'),
                   np.array(train_losses))
        np.savetxt(os.path.join(checkpoints_dir, 'train_losses_sdf_final.txt'),
                   np.array(train_sdf_losses))
        np.savetxt(os.path.join(checkpoints_dir, 'train_losses_scf_final.txt'),
                   np.array(train_scf_losses))

        return model, optimizers


def train_occ_scf(model, train_dataloader, epochs, lr, steps_til_validation, epochs_til_checkpoint, model_dir, loss_fn,
                  val_dataloader=None, clip_grad=False, val_loss_fn=None,
                  overwrite=True, optimizers=None, batches_per_validation=10, gpus=1, rank=0,
                  mtl=False, lw=0.5, max_steps=None, lr_schedule=False):

    if optimizers is None:
        if mtl:
            log_var_occ = torch.zeros((1,), requires_grad=True, device="cuda")
            log_var_scf = torch.zeros((1,), requires_grad=True, device="cuda")
            params = ([p for p in model.parameters()] + [log_var_occ] + [log_var_scf])
            optimizers = [torch.optim.Adam(lr=lr, params=params)]
        else:
            optimizers = [torch.optim.Adam(lr=lr, params=model.parameters())]

    if val_dataloader is not None:
        assert val_loss_fn is not None, "If validation set is passed, have to pass a validation loss_fn!"

    if lr_schedule:
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizers[0], milestones=[200, 300], gamma=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizers[0], step_size=80, gamma=0.2)

    if rank == 0:
        if os.path.exists(model_dir):
            if overwrite:
                shutil.rmtree(model_dir)
            else:
                val = input("The model directory %s exists. Overwrite? (y/n)"%model_dir)
                if val == 'y' or overwrite:
                    shutil.rmtree(model_dir)

        os.makedirs(model_dir)

        summaries_dir = os.path.join(model_dir, 'summaries')
        util.cond_mkdir(summaries_dir)

        checkpoints_dir = os.path.join(model_dir, 'checkpoints')
        util.cond_mkdir(checkpoints_dir)

        writer = SummaryWriter(summaries_dir)

    val_loss_occ_min = 1
    total_steps = 0

    # start training
    with tqdm(total=len(train_dataloader) * epochs) as pbar:
        train_losses = []
        train_occ_losses = []
        train_scf_losses = []
        for epoch in range(epochs):
            # save model and training loss after a fixed number of epochs
            if not epoch % epochs_til_checkpoint and epoch and rank == 0:
                torch.save(model.state_dict(),
                           os.path.join(checkpoints_dir, 'model_final.pth'))
                np.savetxt(os.path.join(checkpoints_dir, 'train_losses_total_final.txt'),
                           np.array(train_losses))
                np.savetxt(os.path.join(checkpoints_dir, 'train_losses_occ_final.txt'),
                           np.array(train_occ_losses))
                np.savetxt(os.path.join(checkpoints_dir, 'train_losses_scf_final.txt'),
                           np.array(train_scf_losses))

            for step, (model_input, gt) in enumerate(train_dataloader):
                model_input = util.dict_to_gpu(model_input)
                gt = util.dict_to_gpu(gt)

                start_time = time.time()

                model_output = model(model_input)
                losses = loss_fn(model_output, gt)
                # losses = loss_fn(model_output, gt, model_input, model)

                train_loss = 0.
                for loss_name, loss in losses.items():
                    single_loss = loss.mean()

                    # save non-weighted loss for each head
                    if loss_name == "occ":
                        train_occ_losses.append(single_loss.item())
                    else:
                        train_scf_losses.append(single_loss.item())

                    if rank == 0:
                        writer.add_scalar("train_" + loss_name, single_loss, total_steps)

                    # compute weighted loss
                    if mtl:
                        if loss_name == "occ":
                            single_loss = torch.exp(-log_var_occ) * single_loss + log_var_occ
                        else:
                            single_loss = torch.exp(-log_var_scf) * single_loss + log_var_scf
                    else:
                        if loss_name == "scf":
                            single_loss *= lw
                        else:
                            single_loss *= (1 - lw)

                    # write summery
                    # if rank == 0:
                    #     writer.add_scalar("train_weighted_" + loss_name, single_loss, total_steps)
                    train_loss += single_loss

                # save total weighted loss and write summery
                train_losses.append(train_loss.item())
                if rank == 0:
                    writer.add_scalar("total_train_loss", train_loss, total_steps)

                # calculate gradient
                for optim in optimizers:
                    optim.zero_grad()
                train_loss.backward()

                if gpus > 1:
                    average_gradients(model)

                if clip_grad:
                    if isinstance(clip_grad, bool):
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)

                for optim in optimizers:
                    optim.step()

                if rank == 0:
                    pbar.update(1)

                # run validation
                if not total_steps % steps_til_validation and rank == 0:
                    print("Epoch %d, Total loss %0.6f, iteration time %0.6f" % (epoch, train_loss, time.time() - start_time))
                    loss_sum = 0
                    if val_dataloader is not None:
                        print("Running validation set...")
                        with torch.no_grad():
                            model.eval()
                            val_losses = defaultdict(list)
                            for val_i, (model_input_i, gt_i) in enumerate(val_dataloader):
                                model_input_i = util.dict_to_gpu(model_input_i)
                                gt_i = util.dict_to_gpu(gt_i)

                                model_output_i = model(model_input_i)
                                val_loss = val_loss_fn(model_output_i, gt_i, val=True)

                                for name, value in val_loss.items():
                                    val_losses[name].append(value.cpu().numpy())

                                if val_i == batches_per_validation:
                                    break

                        for loss_name, loss in val_losses.items():
                            single_loss = np.mean(loss)

                            # save validation loss for each head
                            writer.add_scalar('val_' + loss_name, single_loss, total_steps)

                            if loss_name == "occ":
                                print(f"occ_loss: {single_loss}")
                            else:
                                print(f"scf_loss: {single_loss}")

                            # save best model with the lowest occupancy validation loss
                            if loss_name == "occ" and val_loss_occ_min > single_loss:
                                val_loss_occ_min = single_loss
                                torch.save(model.state_dict(),
                                           os.path.join(checkpoints_dir, 'best_model_epoch_%04d_iter_%06d.pth' % (epoch, total_steps)))
                                # np.savetxt(os.path.join(checkpoints_dir, 'train_losses_%04d_iter_%06d.pth' % (epoch, total_steps)),
                                #            np.array(train_losses))

                            # compute weighted validation loss
                            if mtl:
                                if loss_name == "occ":
                                    log_var_occ_np = log_var_occ.detach().cpu().numpy()
                                    single_loss = np.exp(-log_var_occ_np) * single_loss + log_var_occ_np
                                else:
                                    log_var_scf_np = log_var_scf.detach().cpu().numpy()
                                    single_loss = np.exp(-log_var_scf_np) * single_loss + log_var_scf_np
                            else:
                                if loss_name == "scf":
                                    single_loss *= lw
                                else:
                                    single_loss *= (1 - lw)

                            if loss_name == "occ":
                                print(f"weighted_occ_loss: {single_loss}")
                            else:
                                print(f"weighted_scf_loss: {single_loss}")

                            # save weighted validation loss
                            # writer.add_scalar('val_weighted' + loss_name, single_loss, total_steps)

                            loss_sum += single_loss

                        print(f"total_loss: {loss_sum}")
                        model.train()

                total_steps += 1
                if max_steps is not None and total_steps == max_steps:
                    break

            if max_steps is not None and total_steps == max_steps:
                break

            if lr_schedule:
                scheduler.step()

            # save final model
            torch.save(model.state_dict(),
                       os.path.join(checkpoints_dir, 'model_final.pth'))
            np.savetxt(os.path.join(checkpoints_dir, 'train_losses_total_final.txt'),
                       np.array(train_losses))
            np.savetxt(os.path.join(checkpoints_dir, 'train_losses_occ_final.txt'),
                       np.array(train_occ_losses))
            np.savetxt(os.path.join(checkpoints_dir, 'train_losses_scf_final.txt'),
                       np.array(train_scf_losses))

        return model, optimizers


def train(model, train_dataloader, epochs, lr, steps_til_summary, epochs_til_checkpoint, model_dir, loss_fn,
          summary_fn=None, iters_til_checkpoint=None, val_dataloader=None, clip_grad=False, val_loss_fn=None,
          overwrite=True, optimizers=None, batches_per_validation=10, gpus=1, rank=0, max_steps=None):

    if optimizers is None:
        optimizers = [torch.optim.Adam(lr=lr, params=model.parameters())]

    if val_dataloader is not None:
        assert val_loss_fn is not None, "If validation set is passed, have to pass a validation loss_fn!"

    if rank == 0:
        if os.path.exists(model_dir):
            if overwrite:
                shutil.rmtree(model_dir)
            else:
                val = input("The model directory %s exists. Overwrite? (y/n)"%model_dir)
                if val == 'y' or overwrite:
                    shutil.rmtree(model_dir)

        os.makedirs(model_dir)

        summaries_dir = os.path.join(model_dir, 'summaries')
        util.cond_mkdir(summaries_dir)

        checkpoints_dir = os.path.join(model_dir, 'checkpoints')
        util.cond_mkdir(checkpoints_dir)

        writer = SummaryWriter(summaries_dir)

    total_steps = 0
    with tqdm(total=len(train_dataloader) * epochs) as pbar:
        train_losses = []
        for epoch in range(epochs):
            if not epoch % epochs_til_checkpoint and epoch and rank == 0:
                torch.save(model.state_dict(),
                           os.path.join(checkpoints_dir, 'model_epoch_%04d_iter_%06d.pth' % (epoch, total_steps)))
                np.savetxt(os.path.join(checkpoints_dir, 'train_losses_%04d_iter_%06d.pth' % (epoch, total_steps)),
                           np.array(train_losses))

            for step, (model_input, gt) in enumerate(train_dataloader):
                model_input = util.dict_to_gpu(model_input)
                gt = util.dict_to_gpu(gt)

                start_time = time.time()

                model_output = model(model_input)
                losses = loss_fn(model_output, gt)
                # losses = loss_fn(model_output, gt, model_input, model)

                train_loss = 0.
                for loss_name, loss in losses.items():
                    single_loss = loss.mean()

                    if rank == 0:
                        writer.add_scalar(loss_name, single_loss, total_steps)
                    train_loss += single_loss

                train_losses.append(train_loss.item())
                if rank == 0:
                    writer.add_scalar("total_train_loss", train_loss, total_steps)

                if not total_steps % steps_til_summary and rank == 0:
                    torch.save(model.state_dict(),
                               os.path.join(checkpoints_dir, 'model_current.pth'))
                    summary_fn(model, model_input, gt, model_output, writer, total_steps)

                for optim in optimizers:
                    optim.zero_grad()
                train_loss.backward()

                if gpus > 1:
                    average_gradients(model)

                if clip_grad:
                    if isinstance(clip_grad, bool):
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)

                for optim in optimizers:
                    optim.step()

                if rank == 0:
                    pbar.update(1)

                if not total_steps % steps_til_summary and rank == 0:
                    print("Epoch %d, Total loss %0.6f, iteration time %0.6f" % (epoch, train_loss, time.time() - start_time))

                    if val_dataloader is not None:
                        print("Running validation set...")
                        with torch.no_grad():
                            model.eval()
                            val_losses = defaultdict(list)
                            for val_i, (model_input, gt) in enumerate(val_dataloader):
                                model_input = util.dict_to_gpu(model_input)
                                gt = util.dict_to_gpu(gt)

                                model_output = model(model_input)
                                val_loss = val_loss_fn(model_output, gt, val=True)

                                for name, value in val_loss.items():
                                    val_losses[name].append(value.cpu().numpy())

                                if val_i == batches_per_validation:
                                    break

                        for loss_name, loss in val_losses.items():
                            single_loss = np.mean(loss)
                            summary_fn(model, model_input, gt, model_output, writer, total_steps, 'val_')
                            writer.add_scalar('val_' + loss_name, single_loss, total_steps)

                        model.train()

                if (iters_til_checkpoint is not None) and (not total_steps % iters_til_checkpoint) and rank == 0:
                    torch.save(model.state_dict(),
                               os.path.join(checkpoints_dir, 'model_epoch_%04d_iter_%06d.pth' % (epoch, total_steps)))
                    np.savetxt(os.path.join(checkpoints_dir, 'train_losses_%04d_iter_%06d.pth' % (epoch, total_steps)),
                               np.array(train_losses))

                total_steps += 1
                if max_steps is not None and total_steps==max_steps:
                    break

            if max_steps is not None and total_steps==max_steps:
                break

        torch.save(model.state_dict(),
                   os.path.join(checkpoints_dir, 'model_final.pth'))
        np.savetxt(os.path.join(checkpoints_dir, 'train_losses_final.txt'),
                   np.array(train_losses))

        return model, optimizers


def train_occ(model, train_dataloader, epochs, lr, steps_til_validation, epochs_til_checkpoint, model_dir, loss_fn,
              val_dataloader=None, clip_grad=False, val_loss_fn=None,
              overwrite=True, optimizers=None, batches_per_validation=10, gpus=1, rank=0,
              max_steps=None, lr_schedule=False):

    if optimizers is None:
        optimizers = [torch.optim.Adam(lr=lr, params=model.parameters())]

    if val_dataloader is not None:
        assert val_loss_fn is not None, "If validation set is passed, have to pass a validation loss_fn!"

    if lr_schedule:
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizers[0], milestones=[200, 300], gamma=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizers[0], step_size=80, gamma=0.2)

    if rank == 0:
        if os.path.exists(model_dir):
            if overwrite:
                shutil.rmtree(model_dir)
            else:
                val = input("The model directory %s exists. Overwrite? (y/n)"%model_dir)
                if val == 'y' or overwrite:
                    shutil.rmtree(model_dir)

        os.makedirs(model_dir)

        summaries_dir = os.path.join(model_dir, 'summaries')
        util.cond_mkdir(summaries_dir)

        checkpoints_dir = os.path.join(model_dir, 'checkpoints')
        util.cond_mkdir(checkpoints_dir)

        writer = SummaryWriter(summaries_dir)

    val_loss_occ_min = 1
    total_steps = 0

    # start training
    with tqdm(total=len(train_dataloader) * epochs) as pbar:
        train_losses = []
        for epoch in range(epochs):
            # save model and training loss after a fixed number of epochs
            if not epoch % epochs_til_checkpoint and epoch and rank == 0:
                torch.save(model.state_dict(),
                           os.path.join(checkpoints_dir, 'model_final.pth'))
                np.savetxt(os.path.join(checkpoints_dir, 'train_losses_total_final.txt'),
                           np.array(train_losses))

            for step, (model_input, gt) in enumerate(train_dataloader):
                model_input = util.dict_to_gpu(model_input)
                gt = util.dict_to_gpu(gt)

                start_time = time.time()

                model_output = model(model_input)
                losses = loss_fn(model_output, gt)
                # losses = loss_fn(model_output, gt, model_input, model)

                train_loss = 0.
                for loss_name, loss in losses.items():
                    single_loss = loss.mean()
                    train_loss += single_loss

                # save total weighted loss and write summery
                train_losses.append(train_loss.item())
                if rank == 0:
                    writer.add_scalar("train_loss", train_loss, total_steps)

                # calculate gradient
                for optim in optimizers:
                    optim.zero_grad()
                train_loss.backward()

                if gpus > 1:
                    average_gradients(model)

                if clip_grad:
                    if isinstance(clip_grad, bool):
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)

                for optim in optimizers:
                    optim.step()

                if rank == 0:
                    pbar.update(1)

                # run validation
                if not total_steps % steps_til_validation and rank == 0:
                    print("Epoch %d, Total loss %0.6f, iteration time %0.6f" % (epoch, train_loss, time.time() - start_time))
                    loss_sum = 0
                    if val_dataloader is not None:
                        print("Running validation set...")
                        with torch.no_grad():
                            model.eval()
                            val_losses = defaultdict(list)
                            for val_i, (model_input_i, gt_i) in enumerate(val_dataloader):
                                model_input_i = util.dict_to_gpu(model_input_i)
                                gt_i = util.dict_to_gpu(gt_i)

                                model_output_i = model(model_input_i)
                                val_loss = val_loss_fn(model_output_i, gt_i, val=True)

                                for name, value in val_loss.items():
                                    val_losses[name].append(value.cpu().numpy())

                                if val_i == batches_per_validation:
                                    break

                        for loss_name, loss in val_losses.items():
                            single_loss = np.mean(loss)
                            loss_sum += single_loss

                        print(f"total_loss: {loss_sum}")
                        writer.add_scalar('val_loss', loss_sum, total_steps)

                        # save best model with the lowest occupancy validation loss
                        if val_loss_occ_min > loss_sum:
                            val_loss_occ_min = loss_sum
                            torch.save(model.state_dict(),
                                       os.path.join(checkpoints_dir,
                                                    'best_model_epoch_%04d_iter_%06d.pth' % (epoch, total_steps)))
                            # np.savetxt(os.path.join(checkpoints_dir, 'train_losses_%04d_iter_%06d.pth' % (epoch, total_steps)),
                            #            np.array(train_losses))
                        model.train()

                total_steps += 1
                if max_steps is not None and total_steps == max_steps:
                    break

            if max_steps is not None and total_steps == max_steps:
                break

            if lr_schedule:
                scheduler.step()

            # save final model
            torch.save(model.state_dict(),
                       os.path.join(checkpoints_dir, 'model_final.pth'))
            np.savetxt(os.path.join(checkpoints_dir, 'train_losses_total_final.txt'),
                       np.array(train_losses))

        return model, optimizers


def train_sdf(model, train_dataloader, epochs, lr, steps_til_validation, epochs_til_checkpoint, model_dir, loss_fn,
              val_dataloader=None, clip_grad=False, val_loss_fn=None,
              overwrite=True, optimizers=None, batches_per_validation=10, gpus=1, rank=0,
              max_steps=None, lr_schedule=False):

    if optimizers is None:
        optimizers = [torch.optim.Adam(lr=lr, params=model.parameters())]

    if val_dataloader is not None:
        assert val_loss_fn is not None, "If validation set is passed, have to pass a validation loss_fn!"

    if lr_schedule:
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizers[0], milestones=[200, 300], gamma=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizers[0], step_size=80, gamma=0.2)

    if rank == 0:
        if os.path.exists(model_dir):
            if overwrite:
                shutil.rmtree(model_dir)
            else:
                val = input("The model directory %s exists. Overwrite? (y/n)"%model_dir)
                if val == 'y' or overwrite:
                    shutil.rmtree(model_dir)

        os.makedirs(model_dir)

        summaries_dir = os.path.join(model_dir, 'summaries')
        util.cond_mkdir(summaries_dir)

        checkpoints_dir = os.path.join(model_dir, 'checkpoints')
        util.cond_mkdir(checkpoints_dir)

        writer = SummaryWriter(summaries_dir)

    val_loss_sdf_min = 1
    total_steps = 0

    # start training
    with tqdm(total=len(train_dataloader) * epochs) as pbar:
        train_losses = []
        for epoch in range(epochs):
            # save model and training loss after a fixed number of epochs
            if not epoch % epochs_til_checkpoint and epoch and rank == 0:
                torch.save(model.state_dict(),
                           os.path.join(checkpoints_dir, 'model_final.pth'))
                np.savetxt(os.path.join(checkpoints_dir, 'train_losses_total_final.txt'),
                           np.array(train_losses))

            for step, (model_input, gt) in enumerate(train_dataloader):
                model_input = util.dict_to_gpu(model_input)
                gt = util.dict_to_gpu(gt)

                start_time = time.time()

                model_output = model(model_input)
                losses = loss_fn(model_output, gt)
                # losses = loss_fn(model_output, gt, model_input, model)

                train_loss = 0.
                for loss_name, loss in losses.items():
                    single_loss = loss.mean()
                    train_loss += single_loss

                # save total weighted loss and write summery
                train_losses.append(train_loss.item())
                if rank == 0:
                    writer.add_scalar("train_loss", train_loss, total_steps)

                # calculate gradient
                for optim in optimizers:
                    optim.zero_grad()
                train_loss.backward()

                if gpus > 1:
                    average_gradients(model)

                if clip_grad:
                    if isinstance(clip_grad, bool):
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)

                for optim in optimizers:
                    optim.step()

                if rank == 0:
                    pbar.update(1)

                # run validation
                if not total_steps % steps_til_validation and rank == 0:
                    print("Epoch %d, Total loss %0.6f, iteration time %0.6f" % (epoch, train_loss, time.time() - start_time))
                    loss_sum = 0
                    if val_dataloader is not None:
                        print("Running validation set...")
                        with torch.no_grad():
                            model.eval()
                            val_losses = defaultdict(list)
                            for val_i, (model_input_i, gt_i) in enumerate(val_dataloader):
                                model_input_i = util.dict_to_gpu(model_input_i)
                                gt_i = util.dict_to_gpu(gt_i)

                                model_output_i = model(model_input_i)
                                val_loss = val_loss_fn(model_output_i, gt_i, val=True)

                                for name, value in val_loss.items():
                                    val_losses[name].append(value.cpu().numpy())

                                if val_i == batches_per_validation:
                                    break

                        for loss_name, loss in val_losses.items():
                            single_loss = np.mean(loss)
                            loss_sum += single_loss

                        print(f"total_loss: {loss_sum}")
                        writer.add_scalar('val_loss', loss_sum, total_steps)

                        # save best model with the lowest occupancy validation loss
                        if val_loss_sdf_min > loss_sum:
                            val_loss_sdf_min = loss_sum
                            torch.save(model.state_dict(),
                                       os.path.join(checkpoints_dir,
                                                    'best_model_epoch_%04d_iter_%06d.pth' % (epoch, total_steps)))
                            # np.savetxt(os.path.join(checkpoints_dir, 'train_losses_%04d_iter_%06d.pth' % (epoch, total_steps)),
                            #            np.array(train_losses))
                        model.train()

                total_steps += 1
                if max_steps is not None and total_steps == max_steps:
                    break

            if max_steps is not None and total_steps == max_steps:
                break

            if lr_schedule:
                scheduler.step()

            # save final model
            torch.save(model.state_dict(),
                       os.path.join(checkpoints_dir, 'model_final.pth'))
            np.savetxt(os.path.join(checkpoints_dir, 'train_losses_total_final.txt'),
                       np.array(train_losses))

        return model, optimizers


def train_scf(model, train_dataloader, epochs, lr, steps_til_validation, epochs_til_checkpoint, model_dir, loss_fn,
              val_dataloader=None, clip_grad=False, val_loss_fn=None,
              overwrite=True, optimizers=None, batches_per_validation=10, gpus=1, rank=0,
              max_steps=None, lr_schedule=False):

    if optimizers is None:
        optimizers = [torch.optim.Adam(lr=lr, params=model.parameters())]

    if val_dataloader is not None:
        assert val_loss_fn is not None, "If validation set is passed, have to pass a validation loss_fn!"

    if lr_schedule:
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizers[0], milestones=[200, 300], gamma=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizers[0], step_size=80, gamma=0.2)

    if rank == 0:
        if os.path.exists(model_dir):
            if overwrite:
                shutil.rmtree(model_dir)
            else:
                val = input("The model directory %s exists. Overwrite? (y/n)"%model_dir)
                if val == 'y' or overwrite:
                    shutil.rmtree(model_dir)

        os.makedirs(model_dir)

        summaries_dir = os.path.join(model_dir, 'summaries')
        util.cond_mkdir(summaries_dir)

        checkpoints_dir = os.path.join(model_dir, 'checkpoints')
        util.cond_mkdir(checkpoints_dir)

        writer = SummaryWriter(summaries_dir)

    val_loss_scf_min = 1
    total_steps = 0

    # start training
    with tqdm(total=len(train_dataloader) * epochs) as pbar:
        train_losses = []
        for epoch in range(epochs):
            # save model and training loss after a fixed number of epochs
            if not epoch % epochs_til_checkpoint and epoch and rank == 0:
                torch.save(model.state_dict(),
                           os.path.join(checkpoints_dir, 'model_final.pth'))
                np.savetxt(os.path.join(checkpoints_dir, 'train_losses_total_final.txt'),
                           np.array(train_losses))

            for step, (model_input, gt) in enumerate(train_dataloader):
                model_input = util.dict_to_gpu(model_input)
                gt = util.dict_to_gpu(gt)

                start_time = time.time()

                model_output = model(model_input)
                losses = loss_fn(model_output, gt)
                # losses = loss_fn(model_output, gt, model_input, model)

                train_loss = 0.
                for loss_name, loss in losses.items():
                    single_loss = loss.mean()
                    train_loss += single_loss

                # save total weighted loss and write summery
                train_losses.append(train_loss.item())
                if rank == 0:
                    writer.add_scalar("train_loss", train_loss, total_steps)

                # calculate gradient
                for optim in optimizers:
                    optim.zero_grad()
                train_loss.backward()

                if gpus > 1:
                    average_gradients(model)

                if clip_grad:
                    if isinstance(clip_grad, bool):
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)

                for optim in optimizers:
                    optim.step()

                if rank == 0:
                    pbar.update(1)

                # run validation
                if not total_steps % steps_til_validation and rank == 0:
                    print("Epoch %d, Total loss %0.6f, iteration time %0.6f" % (epoch, train_loss, time.time() - start_time))
                    loss_sum = 0
                    if val_dataloader is not None:
                        print("Running validation set...")
                        with torch.no_grad():
                            model.eval()
                            val_losses = defaultdict(list)
                            for val_i, (model_input_i, gt_i) in enumerate(val_dataloader):
                                model_input_i = util.dict_to_gpu(model_input_i)
                                gt_i = util.dict_to_gpu(gt_i)

                                model_output_i = model(model_input_i)
                                val_loss = val_loss_fn(model_output_i, gt_i, val=True)

                                for name, value in val_loss.items():
                                    val_losses[name].append(value.cpu().numpy())

                                if val_i == batches_per_validation:
                                    break

                        for loss_name, loss in val_losses.items():
                            single_loss = np.mean(loss)
                            loss_sum += single_loss

                        print(f"total_loss: {loss_sum}")
                        writer.add_scalar('val_loss', loss_sum, total_steps)

                        # save best model with the lowest occupancy validation loss
                        if val_loss_scf_min > loss_sum:
                            val_loss_scf_min = loss_sum
                            torch.save(model.state_dict(),
                                       os.path.join(checkpoints_dir,
                                                    'best_model_epoch_%04d_iter_%06d.pth' % (epoch, total_steps)))
                            # np.savetxt(os.path.join(checkpoints_dir, 'train_losses_%04d_iter_%06d.pth' % (epoch, total_steps)),
                            #            np.array(train_losses))
                        model.train()

                total_steps += 1
                if max_steps is not None and total_steps == max_steps:
                    break

            if max_steps is not None and total_steps == max_steps:
                break

            if lr_schedule:
                scheduler.step()

            # save final model
            torch.save(model.state_dict(),
                       os.path.join(checkpoints_dir, 'model_final.pth'))
            np.savetxt(os.path.join(checkpoints_dir, 'train_losses_total_final.txt'),
                       np.array(train_losses))

        return model, optimizers
