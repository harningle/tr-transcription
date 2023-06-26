# -*- coding: utf-8 -*-
"""Detect TR from driver's onboard and transcribe it"""

import argparse
import json
import os
import subprocess

from pyannote.audio import Pipeline
from pydub import AudioSegment
import torch
from tqdm.auto import tqdm
import whisper


def main(in_video: str, out_file: str, ass: bool = False):
    """Detect and extract the TR clips from the onboard video, and then transcribe them and make
    a subtitle

    :param in_video: str: Path to the onboard video
    :param out_file: str: Path to the output transcription file
    :param ass: bool: Whether to create an .ass subtitle file (default is no)
    """

    # Extract audio from onboard video
    os.makedirs('temp', exist_ok=True)
    subprocess.run(['ffmpeg', '-y', '-i', in_video, '-vn', 'temp/audio.wav'],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print('Onboard audio extracted\n')

    # Detect video activity, i.e. TR happens from x second to y second
    pipeline = Pipeline.from_pretrained('pyannote/voice-activity-detection',
                                        use_auth_token=os.environ['hf_token'])
    pipeline.to(torch.device(0))
    tr_clips = pipeline('temp/audio.wav')
    print('All TR detected\n')

    # Transcribe each TR
    raw_audio = AudioSegment.from_wav('temp/audio.wav')
    model = whisper.load_model('large')
    transcript = []
    for clip, _ in tqdm(tr_clips.itertracks(yield_label=False), desc='Transcribing'):

        # Cut the TR clip (+1 sec. before and after, to allow for some buffer)
        clip_start = clip.start*1000 - 1000 if clip.start > 1 else 0
        duration = raw_audio.duration_seconds
        clip_end = clip.end*1000 + 1000 if clip.end < duration - 1 else duration*1000
        tr = raw_audio[clip_start:clip_end]  # Sec. to millisec
        tr.export('temp/tr.wav', format='wav')

        # Transcribe the TR clip
        result = model.transcribe('temp/tr.wav', language='English')
        text = result['text'].strip()
        if text:  # Sec. here instead of millisec.
            transcript.append({'start': clip_start/1000 + 0.9, 'end': clip.end, 'text': text})
    os.remove('temp/tr.wav')

    # Save the transcript to a json file
    with open(out_file, 'w', encoding='utf8') as f:
        json.dump(transcript, f)

    # Create .ass
    if ass:
        make_ass(out_file)
    print('Done!')
    pass


def sec2hmmssms(sec: float) -> str:
    """Convert from seconds to h:mm:ss.ms format

    :param sec: float: Seconds. Should be strictly less than 10 hours?
    :return: str: time in h:mm:ss.ms format

    Examples:

    >>> sec2hmmssms(209.6678)
    '0:03:30.67'

    >>> sec2hmmssms(60)
    '0:01:00.00'
    """

    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    ms = round((s - int(s)) * 100)
    return f'{round(h)}:{round(m):02d}:{s:02.0f}.{ms:02d}'


def make_ass(transcript: str):
    """Make .ass subtitle from the transcription

    :param transcript: str: Path to the .json transcription file
    """

    # Subtitle font format
    subtitle = '[Script Info]\n' \
               'Title: Default Aegisub file\n' \
               'ScriptType: v4.00+\n' \
               'WrapStyle: 0\n' \
               'ScaledBorderAndShadow: yes\n' \
               'YCbCr Matrix: None\n\n' \
               '[V4+ Styles]\n' \
               'Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, ' \
               'BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, ' \
               'BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n' \
               'Style: English,Arial,14,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,' \
               '100,0,0,1,0.5,0,2,10,10,10,1\n\n' \
               '[Events]\n' \
               'Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n'

    # Add each TR to the subtitle
    with open(transcript, 'r', encoding='utf8') as f:
        trs = json.load(f)
    for tr in trs:
        start = sec2hmmssms(tr['start'])
        end = sec2hmmssms(tr['end'])
        subtitle += f'Dialogue: 0,{start},{end},English,,0,0,0,,{tr["text"]}\n'

    # Save .ass
    with open(transcript.removesuffix('.json') + '.ass', 'w', encoding='utf8') as f:
        f.write(subtitle)
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transcribe the TR from onboard video')
    parser.add_argument('--in_video', '-i', type=str, help='Path to the onboard video')
    parser.add_argument('--out_file', '-o', type=str, default='transcript.json',
                        help='Path to the output transcription file')
    parser.add_argument('--ass', action='store_true', help='Create .ass subtitle')
    args = parser.parse_args()
    main(args.in_video, args.out_file, args.ass)
