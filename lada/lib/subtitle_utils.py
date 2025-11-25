# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import logging
import pathlib
import re
from typing import List, Tuple

logger = logging.getLogger(__name__)

Subtitle = Tuple[float, float, str]

def parse_srt(content: str) -> List[Subtitle]:
    content = content.lstrip('\ufeff')  # Remove BOM
    subtitles = []
    lines = content.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.isdigit():
            i += 1
            if i < len(lines) and '-->' in lines[i]:
                time_line = lines[i]
                start, end = time_line.split(' --> ')
                def parse_time(t: str) -> float:
                    h, m, s = t.split(':')
                    if ',' in s:
                        s, ms = s.split(',')
                    elif '.' in s:
                        s, ms = s.split('.')
                    else:
                        ms = '0'
                    return int(h)*3600 + int(m)*60 + int(s) + int(ms)/1000
                start_time = parse_time(start)
                end_time = parse_time(end)
                i += 1
                text = ''
                while i < len(lines) and lines[i].strip():
                    text += lines[i] + '\n'
                    i += 1
                text = text.strip()
                subtitles.append((start_time, end_time, text))
            else:
                i += 1
        else:
            i += 1
    return subtitles

def try_open_subtitle_file(file_path: str) -> List[Subtitle] | None:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            # Check for SRT format
            if re.search(r'\d+\n\d{2}:\d{2}:\d{2}[,.]\d{3} --> \d{2}:\d{2}:\d{2}[,.]\d{3}', content):
                return parse_srt(content)
            else:
                return None
    except Exception as e:
        logger.debug(f"Error reading subtitle file {file_path}: {e}")
        return None

def find_subtitle_file(video_path: str) -> str | None:
    video_path_obj = pathlib.Path(video_path)
    srt_path = video_path_obj.with_suffix('.srt')
    if srt_path.exists():
        subtitles = try_open_subtitle_file(str(srt_path))
        if subtitles is not None:
            logger.info(f"Found valid subtitle file: {srt_path}")
            return str(srt_path)
    return None