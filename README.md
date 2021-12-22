# Use_Deep_Learning_to_Calculate_Clarity_of_Image

The net of model is U-net.

Training data set is non-public.

Train:

1. Put training data into ./train/

2. Write label into ./train/train_data.txt
(type of each label: <img_path> x1 y1 x2 y2 ... xn yn)

3. Run ./main_train.py

Test:

1. Run ./test_demo.py

Test step：

1. Using deep learning model to mark the key points.

2. Cut the area base on key points to calculate clarity.
