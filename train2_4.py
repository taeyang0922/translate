import os
import torch
from prarm2_1 import rec_args
import torch.utils.data as data
from prarm2_1 import device
from processing2_2 import WMRDataset
from backbone2_3 import RecModelBuilder



# train
def rec_train():
    # dataset
    dataset = WMRDataset(rec_args.train_dir, max_len=rec_args.max_len, resize_shape=(rec_args.height, rec_args.width), train=True)
    train_dataloader = data.DataLoader(dataset, batch_size=rec_args.batch_size, num_workers=rec_args.num_workers, shuffle=True, pin_memory=True, drop_last=False)

    # model
    model = RecModelBuilder(rec_num_classes=rec_args.voc_size, sDim=rec_args.decoder_sdim)
    model = model.to(device)
    model.train()

    # Optimizer
    param_groups = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adadelta(param_groups, lr=rec_args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=rec_args.milestones, gamma=0.1)


    os.makedirs(rec_args.save_dir, exist_ok=True)
    # do train
    step = 0
    for epoch in range(rec_args.max_epoch):
        current_lr = optimizer.param_groups[0]['lr']

        for i, batch in enumerate(train_dataloader):
            step += 1
            batch = [v.to(device) for v in batch]
            loss = model(batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            # print
            if step % rec_args.print_interval == 0:
                print('step: {:4d}\tepoch: {:4d}\tloss: {:.4f}'.format(step, epoch, loss.item()))
        scheduler.step()

        # save
        if epoch % rec_args.save_interval == 0:
            save_name = 'checkpoint_' + str(epoch)
            torch.save(model.state_dict(), os.path.join(rec_args.save_dir, save_name))


    torch.save(model.state_dict(), rec_args.saved_model_path)