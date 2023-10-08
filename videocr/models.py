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


class PredictedFrames:
    start_index: int  # 0-based index of the frame
    end_index: int
    words: List[PredictedText]
    confidence: float  # total confidence of all words
    text: str
    pos_min_x = None
    pos_max_x = None
    pos_min_y = None
    pos_max_y = None
    pos_x = None
    pos_y = None

    def __init__(self, index: int, pred_data: list[list], conf_threshold: float):
        self.start_index = index
        self.end_index = index
        self.lines = []
        self.words = []

        total_conf = 0
        word_count = 0
        current_line = []
        current_line_max_y = None
        for l in pred_data[0]:
            if len(l) < 2:
                continue
            bounding_box = l[0]
            text = l[1][0]
            conf = l[1][1]

            # set pos data
            max_x = max(bounding_box[0][0], bounding_box[1][0], bounding_box[2][0], bounding_box[3][0])
            min_x = min(bounding_box[0][0], bounding_box[1][0], bounding_box[2][0], bounding_box[3][0])
            max_y = max(bounding_box[0][1], bounding_box[1][1], bounding_box[2][1], bounding_box[3][1])
            min_y = min(bounding_box[0][1], bounding_box[1][1], bounding_box[2][1], bounding_box[3][1])

            if self.pos_max_x is None or self.pos_max_x > max_x:
                self.pos_max_x = max_x
            if self.pos_min_x is None or self.pos_min_x > min_x:
                self.pos_min_x = min_x
            if self.pos_max_y is None or self.pos_max_y > max_y:
                self.pos_max_y = max_y
            if self.pos_min_y is None or self.pos_min_y > min_y:
                self.pos_min_y = min_y

            self.pos_x = (self.pos_min_x + self.pos_max_x)/2
            self.pos_y = (self.pos_min_y + self.pos_max_y)/2

            # word predictions with low confidence will be filtered out
            if conf >= conf_threshold:
                total_conf += conf
                word_count += 1

                # add word to current line or create a new line
                if current_line_max_y is None:
                    current_line_max_y = max_y
                    current_line.append(PredictedText(bounding_box, conf, text))
                else:
                    height = max_y - min_y
                    height_overlap_allowance = height * 0.1
                    if min_y >= current_line_max_y - height_overlap_allowance: # new line
                        self.lines.append(current_line)
                        current_line = [PredictedText(bounding_box, conf, text)]
                        current_line_max_y = max_y
                    else:
                        current_line.append(PredictedText(bounding_box, conf, text))
                        current_line_max_y = max(current_line_max_y, max_y)

        if len(current_line) > 0:
            self.lines.append(current_line)
        if self.lines:
            self.confidence = total_conf/word_count
            for line in self.lines:
                line.sort(key=lambda word: word.bounding_box[0][0])
        elif len(pred_data[0]) == 0:
            self.confidence = 100
        else:
            self.confidence = 0
        lines = []
        for line in self.lines:
            line_words = []
            for word in line:
                line_words.append(word.text)
                self.words.append(word)
            lines.append(' '.join(line_words))
        self.text = '\n'.join(lines)

    def is_similar_to(self, other: PredictedFrames, threshold=70) -> bool:
        return fuzz.partial_ratio(self.text, other.text) >= threshold


class PredictedSubtitle:
    frames: List[PredictedFrames]
    sim_threshold: int
    text: str
    pos_min_y: int
    pos_max_y: int
    pos_x: int
    pos_y: int

    def __init__(self, frames: List[PredictedFrames], sim_threshold: int):
        self.frames = [f for f in frames if f.confidence > 0]
        self.frames.sort(key=lambda frame: frame.start_index)
        self.sim_threshold = sim_threshold

        if self.frames:
            # prefer lines with more words and similar confidence to fix missing space issues
            max_word_frame = max(self.frames, key=lambda f: (len(f.words), -f.confidence))
            max_conf_frame = max(self.frames, key=lambda f: f.confidence)
            if max_word_frame.confidence >= max_conf_frame.confidence - 1:
                self.text = max_word_frame.text
            else:
                self.text = max_conf_frame.text
            
            self.pos_min_y = max(self.frames, key=lambda f: f.confidence).pos_min_y
            self.pos_max_y = max(self.frames, key=lambda f: f.confidence).pos_max_y
            self.pos_x = max(self.frames, key=lambda f: f.confidence).pos_x
            self.pos_y = max(self.frames, key=lambda f: f.confidence).pos_y
        else:
            self.text = ''

    @property
    def index_start(self) -> int:
        if self.frames:
            return self.frames[0].start_index
        return 0

    @property
    def index_end(self) -> int:
        if self.frames:
            return self.frames[-1].end_index
        return 0

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
        return '{} - {}. {}'.format(self.index_start, self.index_end, self.text)

class MergeDebug:
    start_index: int
    text: str
    last_text: str

    def __init__(self, start_index: int, text: str, last_text: str):
        self.start_index = start_index
        self.text = text
        self.last_text = last_text
