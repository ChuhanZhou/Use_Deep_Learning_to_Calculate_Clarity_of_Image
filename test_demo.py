import cv2
import torch

import models
import data
import get_clarity_of_image
from config import config as cfg

if __name__ == "__main__":
    test_path_list = open("test/path_list.txt","r").readlines()
    path_points_file_path = "test/path_points_list.txt"
    path_points_file = open(path_points_file_path,"w")

    model = models.U_net(n=16)
    model_ckpt = torch.load(cfg["ckpt"])
    model.load_state_dict(model_ckpt, strict=True)
    model.to(cfg['device'])

    for path in test_path_list:
        path = path.split("\n")[0].split(" ")[0]
        name = path.split("/")[-1].split(".")[0]
        test_img = img = cv2.imread(path)

        sp = test_img.shape

        resize_max = -1

        if sp[0] > 1024 and sp[1] > 1024:
            resize_max = 1024
        elif sp[0]%16!=0 or sp[1]%16!=0:
            resize_max = int(max(sp[0], sp[1])/16)*16

        if resize_max!= -1:
            max_pect = resize_max / max(sp[0], sp[1])
            min_pect = int(min(sp[0], sp[1]) * max_pect / 16) * 16 / min(sp[0], sp[1])
            if sp[0] > sp[1]:
                test_size = (int(min_pect * sp[1]), int(max_pect * sp[0]))
            else:
                test_size = (int(max_pect * sp[1]), int(min_pect * sp[0]))
            test_img = cv2.resize(test_img, test_size)

        gray = cv2.cvtColor(test_img, cv2.COLOR_RGB2GRAY)
        test_binary = data.binary_plus(gray)
        test_data = torch.from_numpy(data.img_to_data(test_img)).unsqueeze(0).float()

        pect = model(test_data)
        heatmaps = pect.detach().numpy()[0]
        points = data.heatmap_to_point(heatmaps, test_img.shape)
        points_o = data.mapping_points(img, test_img, points)

        show_img = data.show_result_on_img(img, points_o)
        cv2.imwrite("running_output/{}.png".format(name), show_img)

        out = path
        for point in points_o:
            out = "{} {} {}".format(out, point[1], point[0])
        print(out)
        path_points_file.write("{}\n".format(out))
    path_points_file.close()

    get_clarity_of_image.run(path_points_file_path,"result",need_flip=False)