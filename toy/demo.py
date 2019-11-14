"""
Dual Spatial Pyramid for exposing face warp artifacts in DeepFake videos (DSP-FWA)
"""
import torch
import torch.nn.functional as F
import cv2, os, dlib
import numpy as np
from py_utils.face_utils import lib
from py_utils.vid_utils import proc_vid as pv
from py_utils.DL.pytorch_utils.models.classifier \
    import VGG, ResNet, SqueezeNet, DenseNet, InceptionNet, SPPNet, MobileNet, FPN

sample_num = 10
# Employ dlib to extract face area and landmark points
pwd = os.path.dirname(os.path.abspath(__file__))
front_face_detector = dlib.get_frontal_face_detector()
lmark_predictor = dlib.shape_predictor(pwd + '/../dlib_model/shape_predictor_68_face_landmarks.dat')


def im_test(net, im, args):
    face_info = lib.align(im[:, :, (2,1,0)], front_face_detector, lmark_predictor)
    # Samples
    if len(face_info) != 1:
        prob = -1
    else:
        _, point = face_info[0]
        rois = []
        for i in range(sample_num):
            roi, _ = lib.cut_head([im], point, i)
            rois.append(cv2.resize(roi[0], (args.input_size, args.input_size)))

        # vis_ = np.concatenate(rois, 1)
        # cv2.imwrite('vis.jpg', vis_)

        bgr_mean = np.array([103.939, 116.779, 123.68])
        bgr_mean = bgr_mean[np.newaxis, :, np.newaxis, np.newaxis]
        bgr_mean = torch.from_numpy(bgr_mean).float().cuda()

        rois = torch.from_numpy(np.array(rois)).float().cuda()
        rois = rois.permute((0, 3, 1, 2))
        prob = net(rois - bgr_mean)
        prob = F.softmax(prob, dim=1)
        prob = prob.data.cpu().numpy()
        prob = 1 - np.mean(np.sort(prob[:, 0])[np.round(sample_num / 2).astype(int):])
    return prob, face_info


def draw_face_score(im, face_info, prob):
    if len(face_info) == 0:
        return im

    _, points = np.array(face_info[0])
    x1 = np.min(points[:, 0])
    x2 = np.max(points[:, 0])
    y1 = np.min(points[:, 1])
    y2 = np.max(points[:, 1])

    # Real: (0, 255, 0), Fake: (0, 0, 255)
    color = (0, prob * 255, (1 - prob) * 255)
    cv2.rectangle(im, (x1, y1), (x2, y2), color, 10)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(im, '{:.3f}'.format(prob), (x1, y1 - 10), font, 1, color, 3, cv2.LINE_AA)
    return im
    
    
def setup(args):
    num_class = 2
    if args.arch.lower() == 'vgg16':
        net = VGG(args.layers, num_class)
    elif args.arch.lower() == 'resnet':
        net = ResNet(args.layers, num_class)
    elif args.arch.lower() == 'densenet':
        net = DenseNet(args.layers, num_class)
    elif args.arch.lower() == 'inceptionnet':
        net = InceptionNet(num_class=num_class)
    elif args.arch.lower() == 'squeezenet':
        net = SqueezeNet(num_class=num_class)
    elif args.arch.lower() == 'mobilenet':
        net = MobileNet(num_class=num_class, version=2)
    elif args.arch.lower() == 'fpn':
        net = FPN(backbone=args.layers, num_class=num_class)
    elif args.arch.lower() == 'sppnet':
        net = SPPNet(backbone=args.layers, num_class=num_class)
    net = net.cuda()
    net.eval()
    return net


def main(args):
    net = setup(args)
    model_path = os.path.join(args.save_dir, args.ckpt_name)
    if os.path.isfile(model_path):
        print("=> loading checkpoint '{}'".format(model_path))
        checkpoint = torch.load(model_path)
        start_epoch = checkpoint['epoch']
        net.load_state_dict(checkpoint['net'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(model_path, start_epoch))
    else:
        raise ValueError("=> no checkpoint found at '{}'".format(model_path))

    f_path = args.input
    print('Testing: ' + f_path)
    suffix = f_path.split('.')[-1]
    if suffix.lower() in ['jpg', 'png', 'jpeg', 'bmp', 'tif', 'nef', 'raf']:
        im = cv2.imread(f_path)
        if im is None:
            prob = -1
        else:
            prob, face_info = im_test(net, im, args)
        print(prob)

    elif suffix.lower() in ['mp4', 'avi', 'mov']:
        # Parse video
        imgs, frame_num, fps, width, height = pv.parse_vid(f_path)
        probs = []
        for fid, im in enumerate(imgs):
            print('Frame: ' + str(fid))
            prob, face_info = im_test(net, im, args)
            probs.append(prob)

        return probs


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='')
    parser.add_argument('--arch', type=str, default='resnet',
                        help='VGG, ResNet, SqueezeNet, DenseNet, InceptionNet')
    parser.add_argument('--layers', type=int, default='50')
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--save_dir', type=str, default='')
    parser.add_argument('--ckpt_name', type=str, default='')
    args = parser.parse_args()
    main(args)