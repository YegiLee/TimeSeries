import torch
import yolo
import cv2
from torchvision import transforms
import matplotlib.pyplot as plt
from yolo.visualize import factor, patches, xyxy2xywh
import MCamera

#MPEG-IoMT MAbnormalityDetector


# COCO dataset, 80 classes
classes = ("person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
           "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
           "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
           "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
           "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
           "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
           "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
           "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
           "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
           "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush")

# abnormality detection
def abnormality_detection(image, target, classes, target_object_id, target_object_size, save):
    target=target[0]
    image = image.clone()

    image = image.clamp(0, 1)
    H, W = image.shape[-2:]
    fig = plt.figure(figsize=(W / 160, H / 160))
    ax = fig.add_subplot(111)

    im = image.cpu().numpy()
    ax.imshow(im.transpose(1, 2, 0))  # RGB
    ax.set_title("W: {}   H: {}".format(W, H))
    ax.axis("off")

    if target:
        if "labels" in target:
            if classes is None:
                raise ValueError("'classes' should not be None when 'target' has 'labels'!")

            tags = {l: i for i, l in enumerate(tuple(set(target["labels"].tolist())))}

        index = 0
        if "boxes" in target:
            boxes = target["boxes"]
            boxes = xyxy2xywh(boxes).cpu().detach()
            for i, b in enumerate(boxes):
                area = b[2] * b[3]
                if target["labels"][i].item()==target_object_id:
                    if "labels" in target:
                        l = target["labels"][i].item()
                        index = tags[l]
                        txt = classes[l]
                        if area>=target_object_size:
                            abnormal="Abnomaly detected"
                            abnormal_flag=True
                        else:
                            abnormal="No abnomaly detected"
                            abnormal_flag=False
                        if "scores" in target:
                            s = target["scores"][i]
                            s = round(s.item() * 100)
                            txt = "{} {}%".format(txt, s)
                        ax.text(b[0], b[1], txt, fontsize=10, color=factor(index),
                            horizontalalignment="left", verticalalignment="bottom",
                            bbox=dict(boxstyle="square", fc="black", lw=1, alpha=1))

                        ax.text(b[0]+b[2], b[1]+b[3], abnormal, fontsize=15, color="white",
                            horizontalalignment="center", verticalalignment="bottom",
                            bbox=dict(boxstyle="square", fc="black", lw=1, alpha=1))

                    rect = patches.Rectangle(b[:2], b[2], b[3], linewidth=2, edgecolor=factor(index), facecolor="none")
                    ax.add_patch(rect)

    if save:
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig(save)
        plt.cla()
        plt.close(fig)

    #plt.show()
    return abnormal_flag


def getDetectedAbnormality():
    # get video url from getVideoUrl()
    print("+++++++++ MAbnormalityDetector requests getVideoURL() to MCamera +++++++++\n")
    video_url = MCamera.getVideoURL()

    # run abnormality detection using yolo v5
    # load and prepare model
    ckpt_path = "./yolo/yolov5s_official_2cf45318.pth"
    model = yolo.YOLOv5(80, img_sizes=672, score_thresh=0.3)
    model.eval()

    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint)
    model.head.merge = False

    # read video and detect target object
    target_object_id = 0        # person
    target_object_size = 130000
    num_of_tar_det_frame = 10

    video = cv2.VideoCapture(video_url)
    sucess, frame = video.read()  # read 0 frame
    count = 0
    abnormal_count=0
    print(f'Loading video and run anomaly detection')

    while sucess:

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = transforms.ToTensor()(frame)
        images = [img]
        results, losses = model(images)  # prediction

        abnormal_flag=abnormality_detection(images[0], results, classes, target_object_id, target_object_size,
                              save="./inference/inference_%06d.png" % count)

        if abnormal_flag==True:
            abnormal_count += 1
            print("*", end='')
        else:
            abnormal_count = 0
            print(".", end='')

        if abnormal_count>=num_of_tar_det_frame:
            print(f'\nstart frame of abnormal {count-abnormal_count+1}')
            print(f'end frame of abnormal {count}\n')

            detected_time='2022.10.8 0:00'
            xml=f'''
<mtdl:analysedData xsi:type="maov:LivestockAbnormalityDetectionType">
    <maov:DetectedTime>{detected_time}</maov:DetectedTime>
    <maov:AbnormalityDetection>true</maov:AbnormalityDetection> 
    <maov:DetectedVideoURL>{video_url}</maov:DetectedVideoURL>
</mtdl:analysedData>'''

            print(f'---MAbnormalityDetector getDetectedAbnormality() return--- {xml} \n\n')

            return [xml, detected_time]
            break

        # read next frame
        sucess, frame = video.read()
        count += 1
