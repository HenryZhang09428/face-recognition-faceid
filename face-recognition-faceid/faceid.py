import faceid_core
import cv2
import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np

core = faceid_core.faceid_core(db_dir='./database.pt')
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
font = ImageFont.truetype('C:/Windows/Fonts/Arial.ttf', 20)

def videocapture():
    cap = cv2.VideoCapture(0)  # 生成读取摄像头对象
    while cap.isOpened():
        ret, frame = cap.read()  # 读取摄像头画面
        frame = core.mark_image(frame)
        cv2.imshow('video', frame)  # 显示画面
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def register():
    cap = cv2.VideoCapture(0)
    embeddings_list = []
    idx = -1
    while cap.isOpened():
        ret, frame = cap.read()
        img_draw = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_draw)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        embeddings, _ = core.get_embeddings_and_boxes(img=frame)
        if embeddings is None:
            draw.text((30, 400), 'No faces detected!', fill=(233, 0, 0), font=font)
            cv2.imshow('faceid_register', cv2.cvtColor(np.array(img_draw), cv2.COLOR_RGB2BGR))
            continue
        elif embeddings.size()[0] != 1:
            draw.text((30, 400), 'Please make sure there is ONLY ONE person!', fill=(233, 0, 0), font=font)
            cv2.imshow('faceid_register', cv2.cvtColor(np.array(img_draw), cv2.COLOR_RGB2BGR))
            continue
        else:
            draw.text((30, 400), 'Recognizing...', fill=(0, 233, 0), font=font)
            cv2.imshow('faceid_register', cv2.cvtColor(np.array(img_draw), cv2.COLOR_RGB2BGR))

        embeddings_list.append(embeddings[0])
        if len(embeddings_list) == 30:
            center = sum(embeddings_list) / len(embeddings_list)
            embeddings_list.sort(key=lambda x: torch.sum(torch.pow(x - center, 2)))
            s = torch.tensor(0, dtype=torch.float, device=device)
            for embeddings in embeddings_list:
                s += torch.sum(torch.pow(embeddings - center, 2))
            if s <= 2.0:
                idx = core.get_id(embeddings_list[0])
                if idx != 0:
                    print("您已存在，您的ID为{0:}".format(idx))
                    break
                else:
                    idx = core.write_db(embeddings_list[0], op=0)
                    for i in range(1, 5):
                        core.write_db(embeddings_list[i], op=idx)
                    print("注册成功，您的ID为{0:}".format(idx))
                    break
            else:
                img_draw = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(img_draw)
                draw.text((30, 400), 'Please look at the camera and stay still, retrying!', fill=(233, 0, 0), font=font)
                cv2.imshow('faceid_register', cv2.cvtColor(np.array(img_draw), cv2.COLOR_RGB2BGR))
                cv2.waitKey(2000)
            embeddings_list = []
    cap.release()
    cv2.destroyAllWindows()
    return idx


def verify():
    cap = cv2.VideoCapture(0)
    embeddings_list = []
    idx = -1
    while cap.isOpened():
        ret, frame = cap.read()
        img_draw = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_draw)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        embeddings, _ = core.get_embeddings_and_boxes(img=frame)
        if embeddings is None:
            draw.text((30, 400), 'No faces detected!', fill=(233, 0, 0), font=font)
            cv2.imshow('faceid_verify', cv2.cvtColor(np.array(img_draw), cv2.COLOR_RGB2BGR))
            continue
        elif embeddings.size()[0] != 1:
            draw.text((30, 400), 'Please make sure there is ONLY ONE person!', fill=(233, 0, 0), font=font)
            cv2.imshow('faceid_verify', cv2.cvtColor(np.array(img_draw), cv2.COLOR_RGB2BGR))
            continue
        else:
            draw.text((30, 400), 'Recognizing...', fill=(0, 233, 0), font=font)
            cv2.imshow('faceid_verify', cv2.cvtColor(np.array(img_draw), cv2.COLOR_RGB2BGR))

        embeddings_list.append(embeddings[0])
        if len(embeddings_list) == 30:
            center = sum(embeddings_list) / len(embeddings_list)
            s = torch.tensor(0, dtype=torch.float, device=device)
            for embeddings in embeddings_list:
                s += torch.sum(torch.pow(embeddings - center, 2))
            if s <= 2.0:
                idx = core.get_id(embeddings_list[0])
                break
            else:
                img_draw = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(img_draw)
                draw.text((30, 400), 'Please look at the camera and stay still, retrying!', fill=(233, 0, 0), font=font)
                cv2.imshow('faceid_verify', cv2.cvtColor(np.array(img_draw), cv2.COLOR_RGB2BGR))
                cv2.waitKey(2000)
            embeddings_list = []
    cap.release()
    cv2.destroyAllWindows()
    return idx


if __name__ == '__main__':
    register()
    # videocapture()
    print(verify())
    pass
