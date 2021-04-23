import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def trainer(args, dataloader, model, loss_func, io):
    model = nn.DataParallel(model)
    model.cuda()

    if not args.load_pretrained and os.path.exists('./checkpoints/{}/models/gnn_{}.pth'.format(args.exp_name, args.iteration)):
        model.module.load_state_dict(torch.load('./checkpoints/{}/models/gnn_{}.pth'.format(args.exp_name, args.iteration)))
        print('Load state dict successfully.')


    weight_decay = 1e-6
    opt_gnn = optim.Adam(model.parameters(), lr=args.lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(opt_gnn, step_size=args.dec_lr, gamma=args.gamma)

    for batch_idx in range(args.iterations):

        counter = 0
        total_loss = 0
        # Train
        for data in tqdm(dataloader):
            opt_gnn.zero_grad()
            batch_img, label_link = data[0], data[1].float().cuda()
            batch_img = [b.float().cuda() for b in batch_img]
            out_metric, out_logits, W_logits = model(batch_img)
            loss = loss_func(W_logits, label_link)
            loss.backward()
            opt_gnn.step()
            scheduler.step()


            counter += 1
            total_loss += loss.item()

        display_str = 'Train Iter: {}'.format(batch_idx)
        display_str += '\tLoss_d_metric: {:.6f}'.format(total_loss / counter)

        io.cprint(display_str)

        # Save model
        if (batch_idx + 1) % args.save_interval == 0:
            torch.save(model.module.state_dict(), 'checkpoints/{}/models/gnn_{}.pth'.format(args.exp_name,batch_idx+1))

        # val model


