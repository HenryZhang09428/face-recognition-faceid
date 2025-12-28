import cv2
import torch
from PIL import Image
from utils.mtcnn import MTCNN


class preprocessor:

    def __init__(self, device=torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')):
        self.mtcnn = MTCNN(margin=20, keep_all=True, device=device)
        self.device = device

    def get_faces_from_image(self, path=None, img=None):
        """
        Args:
            path {str} -- 图片的路径
            img {ndarray} -- 照片
        Returns:
            一个n*3*160*160的tensor，代表一共找到了n张脸，每张脸有rgb三通道;
            一个n*4的numpy数组，代表人脸的anchor
            若没有脸，返回None, None
        """
        if img is None:
            img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img)
        faces, boxes = self.mtcnn(img)
        if faces is None:
            return None, None
        return faces.to(self.device), boxes

    def get_faces_from_video(self, path, gap=10):
        """
        Args:
            path: 视频的路径
            gap: 每隔多少帧进行检测

        Returns: 一个list，其中每个元素为一个n*3*160*160的tensor，代表在这一帧一共找到了n张脸，每张脸有rgb三通道

        """
        v_cap = cv2.VideoCapture(path)
        v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        batch_size = 16
        frames = []
        faces = []
        for _ in range(v_len):
            success, frame = v_cap.read()
            assert success is True
            if _ % gap != 0:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))

            if len(frames) >= batch_size:
                faces.extend(self.mtcnn(frames)[0])
                frames = []

        faces.extend(self.mtcnn(frames)[0])
        return faces


if __name__ == '__main__':
    """直接运行此文件进行测试"""
    p = preprocessor()
    print(p.get_faces_from_image('./data/multiface.jpg')[0].size())
    print(p.get_faces_from_image('./data/multiface.jpg')[1])
    # print(len(p.get_faces_from_video('./data/video.mp4', gap=10)[0]))
    # print(p.get_faces_from_video('./data/video.mp4', gap=10))
