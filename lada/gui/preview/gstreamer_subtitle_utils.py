# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import logging
from typing import Optional

from gi.repository import Gst

logger = logging.getLogger(__name__)

def create_subtitle_elements(pipeline: Gst.Pipeline, subtitle_path: str) -> Optional[tuple[Gst.Element, Gst.Element, Gst.Element]]:
    """
    Create and configure subtitle elements for the pipeline.

    Args:
        pipeline: The GStreamer pipeline to add elements to
        subtitle_path: Path to the subtitle file

    Returns:
        Tuple of (filesrc, subparse, textoverlay) elements if successful, None otherwise
    """
    try:
        # Create textoverlay element
        textoverlay = Gst.ElementFactory.make('textoverlay', None)
        if not textoverlay:
            logger.error("Failed to create textoverlay element")
            return None

        # Configure textoverlay
        textoverlay.set_property('font-desc', 'Sans 18')
        textoverlay.set_property('halignment', 'center')
        textoverlay.set_property('valignment', 'bottom')
        textoverlay.set_property('shaded-background', True)

        # Create subparse to parse the subtitle data
        subparse = Gst.ElementFactory.make('subparse', None)
        if not subparse:
            logger.error("Failed to create subparse element")
            return None

        # Create filesrc to read the subtitle file
        filesrc = Gst.ElementFactory.make('filesrc', None)
        if not filesrc:
            logger.error("Failed to create filesrc element for subtitles")
            return None
        filesrc.set_property('location', subtitle_path)

        # Add all elements to pipeline
        pipeline.add(textoverlay)
        pipeline.add(subparse)
        pipeline.add(filesrc)

        # Link the subtitle pipeline: filesrc -> subparse -> textoverlay (text sink)
        filesrc.link(subparse)
        subparse.link(textoverlay)

        return filesrc, subparse, textoverlay

    except Exception:
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
        except Exception:
            pass