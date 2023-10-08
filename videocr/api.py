import os
import re
import openai
import ass
from .video import Video

def get_subtitles(
        video_path: str, lang='ch', ass_out=False, ass_base=f"{os.path.dirname(__file__)}/../base_example.ass", time_start='0:00', time_end='',
        conf_threshold=75, sim_threshold=80, use_fullframe=False,
        det_model_dir=None, rec_model_dir=None, use_gpu=False,
        brightness_threshold=None, similar_image_threshold=100, similar_pixel_threshold=25, frames_to_skip=1,
        crop_x=None, crop_y=None, crop_width=None, crop_height=None, debug=False, debug_str=None) -> str:
    
    v = Video(video_path, det_model_dir, rec_model_dir)
    v.run_ocr(use_gpu, lang, time_start, time_end, conf_threshold, use_fullframe,
        brightness_threshold, similar_image_threshold, similar_pixel_threshold, frames_to_skip,
        crop_x, crop_y, crop_width, crop_height, debug)
    return v.get_subtitles(sim_threshold, ass_out, ass_base, debug, debug_str)


def save_subtitles_to_file(
        video_path: str, file_path='subtitle.ass', lang='ch', ass_base=f"{os.path.dirname(__file__)}/../base_example.ass",
        time_start='0:00', time_end='', conf_threshold=75, sim_threshold=80,
        use_fullframe=False, det_model_dir=None, rec_model_dir=None, use_gpu=False,
        brightness_threshold=None, similar_image_threshold=100, similar_pixel_threshold=25, frames_to_skip=1,
        crop_x=None, crop_y=None, crop_width=None, crop_height=None, debug=False) -> None:
    debug_str, sub_format = os.path.splitext(file_path)
    ass_out = sub_format == '.ass'
    subs = get_subtitles(
            video_path, lang, ass_out, ass_base, time_start, time_end, conf_threshold,
            sim_threshold, use_fullframe, det_model_dir, rec_model_dir, use_gpu,
            brightness_threshold, similar_image_threshold, similar_pixel_threshold, frames_to_skip,
            crop_x, crop_y, crop_width, crop_height, debug, debug_str)
    with open(file_path, 'w+', encoding='utf-8') as f:
        if ass_out:
            subs.dump_file(f)
        else:
            f.write(subs)

def fix_subtitles(api_key: str, subtitle_file: str, lang='english', model='gpt-3.5-turbo') -> None:
    openai.api_key = api_key
    sub_name, sub_format = os.path.splitext(subtitle_file)
    if sub_format == '.ass':
        with open(subtitle_file, encoding='utf_8_sig') as f:
            doc = ass.parse(f)
        original_text = []
        lines = ""
        next_split = 100
        for i in range(len(doc.events)):
            event = doc.events[i]
            if i > next_split:
                next_split += 100
                original_text.append(lines)
                lines = ""
            if (type(event) is ass.line.Dialogue) and (event.style != "Sign"):
                lines += f"{i}|{re.sub(r'{[^}]*}', '', event.text)}\n"
        original_text.append(lines)

        for text in original_text:
            print("Creating ChatGPT request...")
            prompt = f"[No prose] Fix the following lines (spelling, grammar, wording), written in {lang}. Keep this line format without changing the line number: number|text. Keep any occurence of \\N at the same position. Before the first line, add [START], after the last line, add [END]. The original lines:\n\n{text}\n\nThe fixed lines:"
            print(prompt)
            chat = openai.ChatCompletion.create(
                model=model,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            response = chat.choices[0]["message"]["content"]
            print(response)

            fixed_lines = re.search(r'\[START\](.*?)\[END\]', response, re.DOTALL)
            if fixed_lines:
                fixes = fixed_lines.group(1)
                for fix in fixes.split('\n'):
                    line = re.match(r'^(\d+)\|(.*)$', fix)
                    if line:
                        doc.events[int(line.group(1))].text = line.group(2)

        with open(f"{sub_name}_fixed.ass", 'w+', encoding='utf-8') as f:
            doc.dump_file(f)
    else:
        print("This script currently only supports .ass files for text correction.")
