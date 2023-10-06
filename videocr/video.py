from __future__ import annotations
from typing import List
import cv2
import numpy as np
import subprocess as sp
import os
import re

from . import utils
from .models import PredictedFrames, PredictedSubtitle
from .opencv_adapter import Capture
from paddleocr import PaddleOCR
import ass


class Video:
    path: str
    lang: str
    use_fullframe: bool
    det_model_dir: str
    rec_model_dir: str
    num_frames: int
    fps: float
    width: int
    height: int
    ocr: PaddleOCR
    pred_frames: List[PredictedFrames]
    pred_subs: List[PredictedSubtitle]

    def __init__(self, path: str, det_model_dir: str, rec_model_dir: str):
        self.path = path
        self.det_model_dir = det_model_dir
        self.rec_model_dir = rec_model_dir
        with Capture(path) as v:
            self.num_frames = int(v.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = v.get(cv2.CAP_PROP_FPS)
            self.width = int(v.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(v.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def run_ocr(self, use_gpu: bool, lang: str, time_start: str, time_end: str,
                conf_threshold: int, use_fullframe: bool, brightness_threshold: int, similar_image_threshold: int, similar_pixel_threshold: int, frames_to_skip: int,
                crop_x: int, crop_y: int, crop_width: int, crop_height: int) -> None:
        conf_threshold_percent = float(conf_threshold/100)
        self.lang = lang
        self.use_fullframe = use_fullframe
        self.pred_frames = []
        ocr = PaddleOCR(lang=self.lang, rec_model_dir=self.rec_model_dir, det_model_dir=self.det_model_dir, use_gpu=use_gpu)

        ocr_start = utils.get_frame_index(time_start, self.fps) if time_start else 0
        ocr_end = utils.get_frame_index(time_end, self.fps) if time_end else self.num_frames

        if ocr_end < ocr_start:
            raise ValueError('time_start is later than time_end')
        num_ocr_frames = ocr_end - ocr_start

        crop_x_end = None
        crop_y_end = None
        if crop_x and crop_y and crop_width and crop_height:
            crop_x_end = crop_x + crop_width
            crop_y_end = crop_y + crop_height

        # get frames from ocr_start to ocr_end
        with Capture(self.path) as v:
            v.set(cv2.CAP_PROP_POS_FRAMES, ocr_start)
            prev_grey = None
            predicted_frames = None
            modulo = frames_to_skip + 1
            last_init = 0
            for i in range(num_ocr_frames):
                print(f"Frame #{i}")
                if i % modulo == 0:
                    frame = v.read()[1]
                    if not self.use_fullframe:
                        if crop_x_end and crop_y_end:
                            frame = frame[crop_y:crop_y_end, crop_x:crop_x_end]
                        else:
                            # only use bottom third of the frame by default
                            frame = frame[self.height // 3:, :]

                    if brightness_threshold:
                        frame = cv2.bitwise_and(frame, frame, mask=cv2.inRange(frame, (brightness_threshold, brightness_threshold, brightness_threshold), (255, 255, 255)))

                    if similar_image_threshold:
                        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        if prev_grey is not None:
                            _, absdiff = cv2.threshold(cv2.absdiff(prev_grey, grey), similar_pixel_threshold, 255, cv2.THRESH_BINARY)
                            if np.count_nonzero(absdiff) < similar_image_threshold:
                                predicted_frames.end_index = i + ocr_start
                                prev_grey = grey
                                continue

                        prev_grey = grey

                    # check GPU memory every 500 frames, flush if 90 % full
                    if (use_gpu == True) and (i > last_init + 500):
                        gpu_state = sp.check_output(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv"], text=True)
                        mem = re.findall(r'(\d+) MiB', gpu_state)
                        if int(mem[0]) > int(mem[1]) * 0.9:
                            ocr = None
                            ocr = PaddleOCR(lang=self.lang, rec_model_dir=self.rec_model_dir, det_model_dir=self.det_model_dir, use_gpu=use_gpu)
                        last_init = i
                    
                    predicted_frames = PredictedFrames(i + ocr_start, ocr.ocr(frame, cls=False), conf_threshold_percent)
                    self.pred_frames.append(predicted_frames)
                else:
                    v.read()
        

    def get_subtitles(self, sim_threshold: int, ass_out: bool, ass_base: str) -> str:
        self._generate_subtitles(sim_threshold)
        if ass_out:
            with open(ass_base, encoding='utf_8_sig') as f:
                doc = ass.parse(f)

            ass_width = doc.info['PlayResX']
            ass_height = doc.info['PlayResY']

            for i, sub in enumerate(self.pred_subs):
                pos = ""
                if sub.pos_max_y > self.height - self.height * 0.15:
                    line_type = 'dialogue'
                    line_style = 'Default'
                elif sub.pos_min_y < self.height * 0.15:
                    line_type = 'dialogue'
                    line_style = 'Top'
                else:
                    line_type = 'comment'
                    line_style = 'Sign'
                    pos = f"{{\\pos({sub.pos_x*(ass_width/self.width)},{sub.pos_y*(ass_height/self.height)})}}"
                doc.events.add_line(line_type, '0,{},{},{},,0,0,0,,{}{}'.format(
                    utils.get_timestamp(sub.index_start, self.fps, True),
                    utils.get_timestamp(sub.index_end, self.fps, True),
                    line_style,
                    pos, sub.text)+f"{{DEBUG: WIDTH={self.width}, HEIGHT={self.height}, POS_X={sub.pos_x}, POS_Y={sub.pos_y}, POS_MIN_Y={sub.pos_min_y}, POS_MAX_Y={sub.pos_max_y}}}")
            return doc
        else:
            return ''.join(
                '{}\n{} --> {}\n{}\n\n'.format(
                    i,
                    utils.get_timestamp(sub.index_start, self.fps, False),
                    utils.get_timestamp(sub.index_end, self.fps, False),
                    sub.text)
                for i, sub in enumerate(self.pred_subs))

    def _generate_subtitles(self, sim_threshold: int) -> None:
        self.pred_subs = []

        if self.pred_frames is None:
            raise AttributeError(
                'Please call self.run_ocr() first to perform ocr on frames')

        max_frame_merge_diff = int(0.09 * self.fps)
        for frame in self.pred_frames:
            self._append_sub(PredictedSubtitle([frame], sim_threshold), max_frame_merge_diff)
        self.pred_subs = [sub for sub in self.pred_subs if len(sub.frames[0].lines) > 0]

    def _append_sub(self, sub: PredictedSubtitle, max_frame_merge_diff: int) -> None:
        if len(sub.frames) == 0:
            return

        # merge new sub to the last subs if they are not empty, similar and within 0.09 seconds apart
        if self.pred_subs:
            last_sub = self.pred_subs[-1]
            if len(last_sub.frames[0].lines) > 0 and sub.index_start - last_sub.index_end <= max_frame_merge_diff and last_sub.is_similar_to(sub):
                del self.pred_subs[-1]
                sub = PredictedSubtitle(last_sub.frames + sub.frames, sub.sim_threshold)

        self.pred_subs.append(sub)
