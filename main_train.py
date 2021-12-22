import numpy as np
import cv2
import models
import torch
import torch.nn as nn
import data
from config import config as cfg
from torch.autograd import Variable
import torch.utils.data
import os
import matplotlib.pyplot as plt

if __name__ == "__main__":
    print('training start')

    print('start loading data')
    data_set = data.read_data()
    print('data set num:{}'.format(len(data_set)))
    print('training data ready')
    train_data_loader = torch.utils.data.DataLoader(dataset=data_set,batch_size=cfg['batch_size'],shuffle=True)

    model = models.U_net(n=16)

    #model_ckpt = torch.load(cfg["ckpt"])
    #model.load_state_dict(model_ckpt, strict=True)

    model.to(cfg['device'])

    loss_Fun = nn.MSELoss()
    lr = cfg['learning_rate']
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    start_epoch = 0
    save_epoch = cfg['save_epoch']
    min_avg_loss = None

    test_img = cv2.imread("test/test.png")
    show_img = test_img
    show_bi = data.binary_plus(test_img)
    test_data = torch.from_numpy(data.img_to_data(test_img)).unsqueeze(0).float()
    model.train()

    loss_list = []
    lr_set = []
    lr_i=0
    not_min_num = 0
    batchNorm2d = nn.BatchNorm2d(4)
    for epoch in range(start_epoch, cfg['epochs'], 1):
        total_loss = 0.0
        min_loss = None
        max_loss = None
        i = 0
        for i,(x,y) in enumerate(train_data_loader):
            if lr_i<len(lr_set) and epoch==lr_set[lr_i][0] and i == lr_set[lr_i][1]:
                lr = lr_set[lr_i][2]
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                lr_i+=1

            img_data = x.to(torch.float32).to(cfg['device'])
            label = y.to(torch.float32).to(cfg['device'])
            img_np,img_bi = data.data_to_img(img_data[0].detach().numpy())

            pre = model(img_data)

            heatmaps = pre.detach().numpy()[0]
            points = data.heatmap_to_point(heatmaps,img_data.shape)
            pre_out = torch.from_numpy(points).unsqueeze(0)

            pre_norm = batchNorm2d(pre)
            label_norm = batchNorm2d(label)

            loss = loss_Fun(pre_norm, label_norm)

            optimizer.zero_grad()
            loss.backward()
            loss_list.append(loss.item())
            optimizer.step()

            total_loss += loss.item()
            if min_loss==None or loss < min_loss:
                min_loss = loss
            if max_loss==None or loss > max_loss:
                max_loss = loss

            #out = data.show_result_on_img(img_np, data.heatmap_to_point(y[0],img_data.shape), (255,0,255))
            #out = data.show_result_on_img(out,points)
            #cv2.imwrite("{}/{}_{}.png".format(cfg["show_output_path"],epoch,i),out)

            if (i+1)%10 == 0:
                test_pre = model(test_data)
                test_heatmaps = test_pre.detach().numpy()[0]
                test_points = data.heatmap_to_point(test_heatmaps,show_img.shape)

                show_img = data.show_result_on_img(show_img, test_points)
                show_bi = data.show_result_on_img(show_bi, test_points)
                cv2.imwrite("{}/0000_show.png".format(cfg["show_output_path"]), show_img)
                cv2.imwrite("{}/0000_bi.png".format(cfg["show_output_path"]), show_bi)
                show_img = cv2.cvtColor(show_img,cv2.COLOR_RGB2GRAY)
                show_bi = cv2.cvtColor(show_bi,cv2.COLOR_RGB2GRAY)

                for h_i in range(test_heatmaps.shape[0]):
                    output_heatmap = test_heatmaps[h_i]
                    output_heatmap = output_heatmap - output_heatmap.min()
                    output_heatmap = output_heatmap / output_heatmap.max() * 255
                    cv2.imwrite("{}/000_heatmaps_{}.png".format(cfg["show_output_path"], h_i), output_heatmap)
                print("epoch:{} |lr:{} |batch:{}/{} |avg_loss:{:.6f}".format(epoch,lr,i+1,len(data_set),total_loss / (i + 1)))

        avg_loss = total_loss / (i + 1)

        if min_avg_loss==None or avg_loss < min_avg_loss:
            min_avg_loss = avg_loss
            torch.save(model.state_dict(), cfg["save_min_loss_file"])
        else:
            not_min_num += 1

        if not_min_num > 3 and lr >cfg['min_learning_rate']:
            lr_set.append((epoch+1,0,lr/10))
            model_ckpt = torch.load(cfg["save_min_loss_file"])
            model.load_state_dict(model_ckpt, strict=True)
            not_min_num = 0

        print("not_min_num:{:d} |lr_i:{:d} |lr_set_len:{:d}".format(not_min_num, lr_i, len(lr_set)))
        print('Epoch {:d}, photo number {:d}, avg loss {:.6f}, min loss {:.6f}, max loss {:.6f}, min avg loss {:.6f}'.format(epoch, i + 1, avg_loss, min_loss, max_loss, min_avg_loss))
        print('-------------------')

        #save check point of model
        save_name = "{}_epoch_{}.pth".format(cfg["save_file"],epoch)
        if (epoch) % save_epoch == 0:
            torch.save(model.state_dict(), save_name)
    plt.plot(loss_list)
    plt.show()



