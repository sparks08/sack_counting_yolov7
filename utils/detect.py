import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import requests
import json
import tqdm
from scripts.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
from utils.count_objects import CountObjects
from threading import Timer
from os import environ 


class RepeatedTimer(object):
    def __init__(self, interval, function, *args, **kwargs):
        self._timer = None
        self.interval = interval
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self.is_running = False
        self.start()

    def _run(self):
        self.is_running = False
        self.start()
        self.function(*self.args, **self.kwargs)

    def start(self):
        if not self.is_running:
            self._timer = Timer(self.interval, self._run)
            self._timer.start()
            self.is_running = True

    def stop(self):
        self._timer.cancel()
        self.is_running = False


class YoloV7:
    def __init__(self, weights='./models/sackcounter_v3.pt', device='', project='.', name='inference', save_txt=False,
                 trace=True, imgsz=640, classify=False) -> None:
        #./models/objectcounter.pt
        # Initialize
        set_logging(1)
        device = select_device(device)
        self.half = device.type != 'cpu'  # half precision only supported on CUDA

        # Directories
        self.save_txt = save_txt
        self.save_dir = Path(increment_path(Path(project) / name, exist_ok=True))  # increment run
        (self.save_dir / 'labels' if save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        self.stride = int(model.stride.max())  # model stride
        self.imgsz = check_img_size(imgsz, s=self.stride)  # check img_size

        if trace:
            model = TracedModel(model, device, imgsz)

        if self.half:
            model.half()  # to FP16

        # Second-stage classifier
        self.classify = classify
        if self.classify:
            self.modelc = load_classifier(name='resnet101', n=2)  # initialize
            self.modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(
                device).eval()

        # Run inference
        if device.type != 'cpu':
            model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(device).type_as(next(model.parameters())))  # run once

        # Get names and colors
        self.names = model.module.names if hasattr(model, 'module') else model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]
        self.device = device
        self.model = model

        self.close = False



    def detect(self, uuid="", source="http://197.211.126.24:5080/LiveApp/streams/535687921618306764690790.m3u8",
               timeout=0, idle=900, view_img=False, nosave=True, count=True,
               intermittent=True, augment=False, conf_thres=0.45, iou_thres=0.80, classes=['sack'], agnostic_nms=False,
               save_conf=False, line_begin=(0, 400), line_end=(1500, 400)):
        if timeout:
            hard_timeout = time.time() + timeout

        idle_timeout = False
        save_img = not nosave and not source.endswith('.txt')  # save inference images
        webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
            ('rtsp://', 'rtmp://', 'http://', 'https://'))

        # Set Dataloader
        vid_path, vid_writer = None, None
        if webcam:
            view_img = view_img and check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=self.imgsz, stride=self.stride)
        else:
            dataset = LoadImages(source, img_size=self.imgsz, stride=self.stride)

        count_object = CountObjects(line_begin, line_end, self.names, self.colors, idle)
        if intermittent and count:
            thread = RepeatedTimer(10, self._update, uuid, count_object)

        objects_detected = {}
        try:
            old_img_w = old_img_h = self.imgsz
            old_img_b = 1

            start_time = time.time()
            for path, img, im0s, vid_cap in tqdm.tqdm(dataset):
                img = torch.from_numpy(img).to(self.device)
                img = img.half() if self.half else img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)

                # Warmup
                if self.device.type != 'cpu' and (
                        old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
                    old_img_b = img.shape[0]
                    old_img_h = img.shape[2]
                    old_img_w = img.shape[3]
                    for i in range(3):
                        self.model(img, augment=augment)[0]

                # Inference
                t1 = time_synchronized()
                pred = self.model(img, augment=augment)[0]
                t2 = time_synchronized()

                # Apply NMS
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=agnostic_nms)
                t3 = time_synchronized()

                # Apply Classifier
                if self.classify:
                    pred = apply_classifier(pred, self.modelc, img, im0s)

                # Process detections
                for i, det in enumerate(pred):  # detections per image
                    if webcam:  # batch_size >= 1
                        p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
                    else:
                        p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

                    p = Path(p)  # to Path
                    save_path = f"{self.save_dir}/predicted_{p.name}"  # img.jpg
                    txt_path = str(self.save_dir / 'labels' / p.stem) + (
                        '' if dataset.mode == 'image' else f'_{frame}')  # img.txt
                    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                    objects_in_frame = []
                    if len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                        # Print results
                        # for c in det[:, -1].unique():
                        #     n = (det[:, -1] == c).sum()  # detections per class
                        #     s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                        # Write results
                        for *xyxy, conf, cls in reversed(det):
                            objects_in_frame.append([*torch.tensor(xyxy).tolist(), self.names[int(cls)], conf.item()])
                            objects_detected.setdefault(self.names[int(cls)], 0)
                            if self.save_txt:  # Write to file
                                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(
                                    -1).tolist()  # normalized xywh
                                line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                                with open(txt_path + '.txt', 'a') as f:
                                    f.write(('%g ' * len(line)).rstrip() % line + '\n')

                            if save_img or view_img:  # Add bbox to image
                                label = f'{self.names[int(cls)]} {conf:.2f}'
                                plot_one_box(xyxy, im0, label=label, color=self.colors[int(cls)], line_thickness=1)

                        if count:
                            im0, idle_timeout = count_object.count_objects_crossing_the_virtual_line(im0,
                                                                                                     objects_in_frame,
                                                                                                     targeted_classes=classes)

                    # # Print time (inference + NMS)
                    # print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

                    # Stream results
                    if view_img:
                        cv2.imshow(str(p), im0)
                        cv2.waitKey(1)  # 1 millisecond

                    # Save results (image with detections)
                    if save_img:
                        if count:
                            # Add line to the image
                            im0 = cv2.line(im0, line_begin, line_end, color=(0, 0, 255), thickness=3)
                            im0 = cv2.putText(im0, f"SACKS: {count_object.counter}", (450, 30),
                                              cv2.FONT_HERSHEY_SIMPLEX, 1,
                                              (0, 0, 255), 3)
                        if dataset.mode == 'image':
                            cv2.imwrite(save_path, im0)
                        else:  # 'video' or 'stream'
                            if vid_path != save_path:  # new video
                                vid_path = save_path
                                if isinstance(vid_writer, cv2.VideoWriter):
                                    vid_writer.release()  # release previous video writer
                                if vid_cap:  # video
                                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                else:  # stream
                                    fps, w, h = 30, im0.shape[1], im0.shape[0]
                                    save_path += '.mp4'
                                vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                            vid_writer.write(im0)
                if idle_timeout or (timeout and hard_timeout < time.time()):
                    print('Timeout Reached')
                    dataset.stopIteration = True

            if self.save_txt or save_img:
                s = f"\n{len(list(self.save_dir.glob('labels/*.txt')))} labels saved to {self.save_dir / 'labels'}" if self.save_txt else ''
                print(f"Results saved to {self.save_dir}{s}")

            print(f'Done. ({time.time() - start_time:.3f}s), total objects detected: {count_object.counter}')

        except Exception as ex:
            print(ex)
            print(f'Exception - {ex}')
        finally:
            if intermittent:
                thread.stop()
            if not count:
                count_object.counter = objects_detected
            self._update(uuid, count_object)
            dataset.stopIteration = True

    def _update(self, uuid, count_object):
        MAX_RETRIES = 3
        try:
            while MAX_RETRIES:
                MAX_RETRIES -= 1
                response = requests.request("PATCH",
                                            environ.get('UPDATE_URL',
                                                        'https://6t4ilb6t53.execute-api.af-south-1.amazonaws.com/dev/updatetrip'),
                                            data=json.dumps({"uuid": uuid, "detection": count_object.counter}),
                                            headers={"Content-Type": "application/json"})
                if 'timeout' in response.content.decode():
                    continue
                break
            if response.status_code != 200:
                raise Exception(f"Update Server Error {response.content}")
        except Exception as ex:
            print(str(ex))
            print(f"uuid: {uuid}, detection: {count_object.counter}")


def aux(uuid="", source="", timeout=0, idle=900, view_img=False, nosave=True, count=True, intermittent=True,
        augment=False, conf_thres=0.45, iou_thres=0.45, classes=[], agnostic_nms=False, save_conf=False,
        line_begin=(0, 400), line_end=(1500, 400)):
    with torch.no_grad():
        model = YoloV7()
        model.detect(uuid=uuid, source=source, timeout=timeout, idle=idle, view_img=view_img, nosave=nosave,
                     count=count, intermittent=intermittent, augment=augment, conf_thres=conf_thres,
                     iou_thres=iou_thres, classes=classes, agnostic_nms=agnostic_nms, save_conf=save_conf,
                     line_begin=line_begin, line_end=line_end)


if __name__ == '__main__':
    with torch.no_grad():
        model = YoloV7()
        model.detect("http://197.211.126.24:5080/LiveApp/streams/535687921618306764690790.m3u8", nosave=False, count=True)
