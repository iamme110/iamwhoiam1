# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import logging
import pathlib
from typing import Optional

from gi.repository import Gst

logger = logging.getLogger(__name__)

def find_and_add_subtitle_elements(pipeline: Gst.Pipeline, video_file_path: str, video_sink: Gst.Element) -> Optional[tuple[Gst.Element, Gst.Element, Gst.Element]]:
    """
    Find SRT subtitle file and add subparse + textoverlay elements to the pipeline.

    Args:
        pipeline: The GStreamer pipeline to add elements to
        video_file_path: Path to the video file
        video_sink: The video sink element to connect subtitle overlay to

    Returns:
        Tuple of (filesrc, subparse, textoverlay) elements if subtitles were found and added,
        None otherwise
    """
    try:
        # Look for SRT file with the same name as video file
        video_path = pathlib.Path(video_file_path)
        srt_path = video_path.with_suffix('.srt')

        if not srt_path.exists():
            logger.debug(f"No SRT subtitle file found at: {srt_path}")
            return None

        logger.info(f"Found SRT subtitle file: {srt_path}")

        # Create textoverlay element to render subtitles (this is the correct element)
        textoverlay = Gst.ElementFactory.make('textoverlay', None)
        if not textoverlay:
            logger.error("Failed to create textoverlay element")
            return None

        # Configure textoverlay with correct properties from documentation
        textoverlay.set_property('font-desc', 'Sans 18')  # Font description
        textoverlay.set_property('halignment', 'center')  # Horizontal alignment
        textoverlay.set_property('valignment', 'bottom') # Vertical alignment
        textoverlay.set_property('shaded-background', True) # Show background

        # Create subparse to parse the subtitle data
        subparse = Gst.ElementFactory.make('subparse', None)
        if not subparse:
            logger.error("Failed to create subparse element")
            return None

        # Create filesrc to read the SRT file
        filesrc = Gst.ElementFactory.make('filesrc', None)
        if not filesrc:
            logger.error("Failed to create filesrc element for subtitles")
            return None
        filesrc.set_property('location', str(srt_path.resolve()))

        # Add all elements to pipeline
        pipeline.add(textoverlay)
        pipeline.add(subparse)
        pipeline.add(filesrc)

        # Link the subtitle pipeline: filesrc -> subparse -> textoverlay (text sink)
        filesrc.link(subparse)
        subparse.link(textoverlay)

        return filesrc, subparse, textoverlay

    except Exception as e:
        logger.error(f"Error setting up subtitle elements: {e}")
        return None

def remove_subtitle_elements(pipeline: Gst.Pipeline, subtitle_elements: Optional[tuple[Gst.Element, Gst.Element, Gst.Element]]) -> None:
    """
    Remove subtitle elements from pipeline if they exist.

    Args:
        pipeline: The GStreamer pipeline
        subtitle_elements: Tuple of (filesrc, subparse, textoverlay) elements to remove
    """
    if subtitle_elements:
        filesrc, subparse, textoverlay = subtitle_elements
        try:
            filesrc.set_state(Gst.State.NULL)
            subparse.set_state(Gst.State.NULL)
            textoverlay.set_state(Gst.State.NULL)
            pipeline.remove(filesrc)
            pipeline.remove(subparse)
            pipeline.remove(textoverlay)
            logger.debug("Removed subtitle elements from pipeline")
        except Exception as e:
            logger.error(f"Error removing subtitle elements: {e}")