import os
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
