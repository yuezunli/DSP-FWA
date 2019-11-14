

import cv2, os
from pathlib import Path
import numpy as np

# We only use opencv3
if not (cv2.__version__).startswith('3.'):
    raise ValueError('Only opencv 3. is supported!')


def crop_video(pathIn, pathOut, pos, size):
    """
    Crop video
    :param pathIn:
    :param pathOut:
    :param pos: (left, top, right, bottom)
    :return:
    """

    imgs, frame_num, fps, width, height = parse_vid(pathIn)

    for i, image in enumerate(imgs):
        y1 = np.int32(pos[0])
        x1 = np.int32(pos[1])
        y2 = np.int32(pos[2])
        x2 = np.int32(pos[3])
        roi = image[y1:y2, x1:x2, :]
        if size is not 'None':
            roi = cv2.resize(roi, (size[1], size[0]))
        imgs[i] = roi

    gen_vid(pathOut, imgs, fps, width, height)


def get_video_dims(video_path):
    vidcap = cv2.VideoCapture(video_path)
    width = np.int32(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))  # float
    height = np.int32(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float
    vidcap.release()
    return width, height


def get_video_frame_nums(video_path):
    vidcap = cv2.VideoCapture(video_path)
    frame_num = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    vidcap.release()
    return frame_num


def get_video_fps(video_path):
    vidcap = cv2.VideoCapture(video_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    vidcap.release()
    return fps


def parse_vid(video_path):
    vidcap = cv2.VideoCapture(video_path)
    frame_num = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    width = np.int32(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)) # float
    height = np.int32(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float
    imgs = []
    while True:
        success, image = vidcap.read()
        if success:
            imgs.append(image)
        else:
            break

    vidcap.release()
    if len(imgs) != frame_num:
        frame_num = len(imgs)
    return imgs, frame_num, fps, width, height


def parse_vid_into_imgs(video_path, folder, im_name='{:05d}.jpg'):
    imgs, frame_num, fps, width, height = parse_vid(video_path)
    for id, im in enumerate(imgs):
        im_name = im_name.format(id)
        cv2.imwrite(folder + '/' + im_name, im)
    print('Save original images to folder {}'.format(folder))


def gen_vid(video_path, imgs, fps, width=None, height=None):
    # Combine video
    ext = Path(video_path).suffix
    if ext == '.mp4':
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Be sure to use lower case
    elif ext == '.avi':
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')  #*'XVID')
    else:
        # if not .mp4 or avi, we force it to mp4
        video_path = video_path.replace(ext, '.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Be sure to use lower case
    if width is None or height is None:
        height, width= imgs[0].shape[:2]
    else:
        imgs_ = [cv2.resize(img, (width, height)) for img in imgs]
        imgs = imgs_

    out = cv2.VideoWriter(video_path, fourcc, fps, (np.int32(width), np.int32(height)))

    for image in imgs:
        out.write(np.uint8(image))  # Write out frame to video

    # Release everything if job is finished
    out.release()
    print('The output video is ' + video_path)


def gen_vid_from_folder(video_path, img_dir, ext, fps, width=None, height=None):
    imgs_path = sorted(Path(img_dir).glob('*' + ext))
    imgs = [cv2.imread(str(p)) for p in imgs_path]
    gen_vid(video_path, imgs, fps, width, height)


def resize_video(video_path, w=None, h=None, scale=1., out_path=None):
    imgs, frame_num, fps, width, height = parse_vid(video_path)
    # Resize imgs
    if w is None or h is None:
        width, height = int(width * scale), int(height * scale)
        for i, im in enumerate(imgs):
            im = cv2.resize(im, None, None, fx=scale, fy=scale)
            imgs[i] = im
    else:
        width, height = w, h
        for i, im in enumerate(imgs):
            im = cv2.resize(im, (w, h))
            imgs[i] = im
    if out_path:
        gen_vid(out_path, imgs, fps, width, height)
    return imgs, frame_num, fps, width, height


def extract_key_frames(video_path, len_window=50):
    """
    The frames which the average interframe difference are local maximum are
    considered to be key frames.
    It should be noted that smoothing the average difference value before
    calculating the local maximum can effectively remove noise to avoid
    repeated extraction of frames of similar scenes.
    """
    imgs, frame_num, fps, width, height = parse_vid(video_path)
    frame_diffs = []
    for i in range(1, len(imgs)):
        curr_frame = cv2.cvtColor(imgs[i], cv2.COLOR_BGR2LUV)
        prev_frame = cv2.cvtColor(imgs[i - 1], cv2.COLOR_BGR2LUV)
        # logic here
        diff = cv2.absdiff(curr_frame, prev_frame)
        diff_sum = np.sum(diff)
        diff_sum_mean = diff_sum / (diff.shape[0] * diff.shape[1])
        frame_diffs.append(diff_sum_mean)
    from scipy.signal import argrelextrema
    # compute keyframe
    diff_array = np.array(frame_diffs)

    def smooth(x, window_len=13, window='hanning'):
        s = np.r_[2 * x[0] - x[window_len:1:-1],
                  x, 2 * x[-1] - x[-1:-window_len:-1]]
        if window == 'flat':  # moving average
            w = np.ones(window_len, 'd')
        else:
            w = getattr(np, window)(window_len)
        y = np.convolve(w / w.sum(), s, mode='same')
        return y[window_len - 1:-window_len + 1]

    sm_diff_array = smooth(diff_array, len_window)
    frame_indexes = np.asarray(argrelextrema(sm_diff_array, np.greater))[0]
    key_frames = []
    for i in frame_indexes:
        key_frames.append(imgs[i])
    return key_frames




