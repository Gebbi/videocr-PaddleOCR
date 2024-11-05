from __future__ import annotations
from typing import List
from dataclasses import dataclass
from statistics import mean
from thefuzz import fuzz

@dataclass
class PredictedText:
    __slots__ = 'bounding_box', 'confidence', 'text'
    bounding_box: list
    confidence: float
    text: str

@dataclass
class PredictedTextWindow:
    __slots__ = 'pos_x', 'pos_y', 'pos_min_y', 'pos_max_y', 'confidence', 'words', 'text'
    pos_x: float
    pos_y: float
    pos_min_y: float
    pos_max_y: float
    confidence: float  # total confidence of all words
    words: List[PredictedText]
    text: str


class PredictedFrames:
    start_index: int  # 0-based index of the frame
    end_index: int
    windows: List[PredictedTextWindow]
    pos_min_x = None
    pos_max_x = None
    pos_min_y = None
    pos_max_y = None
    pos_x = None
    pos_y = None

    def _create_text(self, line):
        line_words = []
        for word in line:
            line_words.append(word.text)
        return ' '.join(line_words)

    def __init__(self, index: int, pred_data: list[list], conf_threshold: float):
        self.start_index = index
        # self.end_index = index
        # self.lines = []
        self.windows = []

        # print(f"pred_data: {pred_data}")
        # print(f"pred_data[0]: {pred_data[0]}")

        total_conf = 0
        word_count = 0
        current_line = []
        current_line_max_y = None
        reset = False
        for l in pred_data[0]:
            # print(f"l: {l}")
            if len(l) < 2:
                continue
            bounding_box = l[0]
            text = l[1][0]
            conf = l[1][1]

            # print(f"text: {l[1][0]}")

            # set pos data
            max_x = max(bounding_box[0][0], bounding_box[1][0], bounding_box[2][0], bounding_box[3][0])
            min_x = min(bounding_box[0][0], bounding_box[1][0], bounding_box[2][0], bounding_box[3][0])
            max_y = max(bounding_box[0][1], bounding_box[1][1], bounding_box[2][1], bounding_box[3][1])
            min_y = min(bounding_box[0][1], bounding_box[1][1], bounding_box[2][1], bounding_box[3][1])

            if self.pos_max_x is None or self.pos_max_x > max_x or reset:
                self.pos_max_x = max_x
            if self.pos_min_x is None or self.pos_min_x > min_x or reset:
                self.pos_min_x = min_x
            if self.pos_max_y is None or self.pos_max_y > max_y or reset:
                self.pos_max_y = max_y
            if self.pos_min_y is None or self.pos_min_y > min_y or reset:
                self.pos_min_y = min_y

            self.pos_x = (self.pos_min_x + self.pos_max_x)/2
            self.pos_y = (self.pos_min_y + self.pos_max_y)/2

            # word predictions with low confidence will be filtered out
            if conf >= conf_threshold:
                total_conf += conf
                word_count += 1

                # add word to current line or create a new line
                if current_line_max_y is None:
                    # print(f"Line start with: {text}")
                    current_line_max_y = max_y
                    current_line.append(PredictedText(bounding_box, conf, text))
                else:
                    height = max_y - min_y
                    height_overlap_allowance = height * 0.1
                    if min_y >= current_line_max_y - height_overlap_allowance: # new line 
                        # print(f"Creating new line with text: {text}")
                        self.windows.append(PredictedTextWindow(self.pos_x, self.pos_y, self.pos_min_y, self.pos_max_y, total_conf/word_count, current_line, self._create_text(current_line)))
                        reset = True

                        current_line = [PredictedText(bounding_box, conf, text)]
                        total_conf = conf
                        word_count = 1
                        current_line_max_y = max_y
                    else:
                        current_line.append(PredictedText(bounding_box, conf, text))
                        current_line_max_y = max(current_line_max_y, max_y)
            else:
                reset = True

        if len(current_line) > 0 and word_count > 0:
            # print(f"Finalize: {text}")
            # self.pos_max_x = max_x
            # self.pos_min_x = min_x
            # self.pos_max_y = max_y
            # self.pos_min_y = min_y
            self.windows.append(PredictedTextWindow((min_x +  max_x)/2, (min_y + max_y)/2, min_y, max_y, total_conf/word_count, current_line, self._create_text(current_line)))

        # print(f"Windows: {self.windows}")

        # if self.lines:
            # self.confidence = total_conf/word_count
            # for line in self.lines:
                # line.sort(key=lambda word: word.bounding_box[0][0])
        # elif len(pred_data[0]) == 0:
            # self.confidence = 100
        # else:
            # self.confidence = 0

        # for line in self.lines:
            # line_words = []
            # for word in line:
                # line_words.append(word.text)
                # self.words.append(word)
            # line_text = ' '.join(line_words)
            # self.windows.append(PredictedTextWindow())
        # self.text = '\n'.join(lines)

    def is_similar_to(self, other: PredictedFrames, threshold=70) -> bool:
        return fuzz.partial_ratio(self.text, other.text) >= threshold

class PredictedSubtitleGroup:
    subtitles: List[PredictedSubtitle]

    def __init__(self, subtitles: List[PredictedSubtitle]):
        self.subtitles = subtitles

        for sub in self.subtitles:
            self.merge_lines(sub)

    def merge_lines(self, sub):
        for other in self.subtitles:
            if other == sub:
                continue
            # print(f"sub: {sub}")
            # print(f"other: {other}")
            # print(f"sub.pos_x in range: {int(sub.pos_x) in range(int(other.pos_x-10), int(other.pos_x+10))}")
            # print(f"sub.pos_max_y in range: {int(sub.pos_max_y) in range(int(other.pos_min_y), int(other.pos_min_y*2) - int(other.pos_max_y))} | sub.pos_max_y: {int(sub.pos_max_y)} | other.pos_min_y: {int(other.pos_min_y)} | other.pos_min_y*2 - other.pos_max_y: {int(other.pos_min_y*2) - int(other.pos_max_y)}")
            if int(sub.pos_x) in range(int(other.pos_x-100), int(other.pos_x+100)) and int(sub.pos_max_y) in range(int(other.pos_min_y*2) - int(other.pos_max_y), int(other.pos_min_y)):
                sub.text = '\n'.join([sub.text, other.text])
                sub.window.words = sub.window.words + other.window.words
                sub.window.confidence = (sub.window.confidence + other.window.confidence) / 2
                sub.pos_max_y = other.pos_max_y
                sub.pos_y = (sub.pos_min_y + other.pos_max_y)/2
                sub.index_start = other.index_start
                self.subtitles.remove(other)

    def update(self, temp_subtitle_group, fps):
        max_frame_merge_diff = int(0.09 * fps)
        orphans = []
        for sub in temp_subtitle_group.subtitles:
            merged = False
            for other in self.subtitles:
                # print(f"sub: {sub}")
                # print(f"sub type: {type(sub)}")
                # print(f"other: {other}")
                # print(f"other type: {type(other)}")
                if int(sub.pos_x) in range(int(other.pos_x-20), int(other.pos_x+20)) and int(sub.pos_y) in range(int(other.pos_y-10), int(other.pos_y+10)) and sub.is_similar_to(other) and sub.index_start - other.index_end <= max_frame_merge_diff:
                    other.index_end = sub.index_end
                    # prefer lines with more words and similar confidence to fix missing space issues
                    max_word_sub = max(sub, other, key=lambda f: (len(f.window.words), -f.window.confidence))
                    max_conf_sub = max(sub, other, key=lambda f: f.window.confidence)
                    if max_word_sub.window.confidence >= max_conf_sub.window.confidence - 1:
                        other.text = max_word_sub.text
                        other.window.words = max_word_sub.window.words
                        other.window.confidence = max_word_sub.window.confidence
                    else:
                        other.text = max_conf_sub.text
                        other.window.words = max_conf_sub.window.words
                        other.window.confidence = max_conf_sub.window.confidence
                    merged = True
            if not merged:
                orphans.append(sub)
        if len(orphans) < len(temp_subtitle_group.subtitles):
            self.subtitles = self.subtitles + orphans
            return True
        else:
            return False


class PredictedSubtitle:
    sim_threshold: int
    text: str
    pos_min_y: int
    pos_max_y: int
    pos_x: int
    pos_y: int
    index_start: int
    index_end: int
    window: PredictedTextWindow

    def __init__(self, text, pos_min_y, pos_max_y, pos_x, pos_y, index_start, index_end, sim_threshold: int, window: PredictedTextWindow):
        self.sim_threshold = sim_threshold
        self.text = text
        self.pos_min_y = pos_min_y
        self.pos_max_y = pos_max_y
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.index_start = index_start
        self.index_end = index_end
        self.window = window

        # if self.frames:
        #     # prefer lines with more words and similar confidence to fix missing space issues
        #     max_word_frame = max(self.frames, key=lambda f: (len(f.words), -f.confidence))
        #     max_conf_frame = max(self.frames, key=lambda f: f.confidence)
        #     if max_word_frame.confidence >= max_conf_frame.confidence - 1:
        #         self.text = max_word_frame.text
        #     else:
        #         self.text = max_conf_frame.text
            
        #     self.pos_min_y = max(self.frames, key=lambda f: f.confidence).pos_min_y
        #     self.pos_max_y = max(self.frames, key=lambda f: f.confidence).pos_max_y
        #     self.pos_x = max(self.frames, key=lambda f: f.confidence).pos_x
        #     self.pos_y = max(self.frames, key=lambda f: f.confidence).pos_y
        # else:
        #     self.text = ''

    def is_similar_to(self, other: PredictedSubtitle) -> bool:
        text = self.text.split('\n')
        other_text = other.text.split('\n')
        sim = []
        for i, line in enumerate(text):
            if len(other_text) >= i+1:
                sim.append(fuzz.partial_ratio(line.replace(' ', ''), other_text[i].replace(' ', '')))
            else:
                sim.append(0)
        return mean(sim) >= self.sim_threshold

    def __repr__(self):
        return f"start: {self.index_start}, end: {self.index_end}, pos_min_y: {self.pos_min_y}, pos_max_y: {self.pos_max_y}, pos_x: {self.pos_x}, pos_y: {self.pos_y}, text: {self.text}"

class MergeDebug:
    start_index: int
    text: str
    last_text: str

    def __init__(self, start_index: int, text: str, last_text: str):
        self.start_index = start_index
        self.text = text
        self.last_text = last_text
