# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import argparse
import hashlib
import json
import os
import pathlib
import shutil
import sys
import tempfile
import textwrap
from fractions import Fraction

import torch

from lada import MODEL_WEIGHTS_DIR, VERSION
from lada.cli import utils
from lada.utils import audio_utils, video_utils
from lada.restorationpipeline.frame_restorer import FrameRestorer
from lada.restorationpipeline import load_models
from lada.utils.video_utils import get_video_meta_data, VideoWriter

def setup_argparser() -> argparse.ArgumentParser:
    examples_header_text = _("Examples:")

    example1_text = _("Restore video with default settings:")
    example1_command = _("%(prog)s --input input.mp4")

    example2_text = _("Restore all videos found in the specified directory and save them to a different folder:")
    example2_command = _("%(prog)s --input path/to/input/dir/ --output /path/to/output/dir/")

    example3_text = _("Use a GPU-accelerated codec for encoding the restored video:")
    example3_command = _("%(prog)s --input input.mp4 --codec hevc_nvenc --crf 20")

    parser = argparse.ArgumentParser(
        usage=_('%(prog)s [options]'),
        description=_("Restore pixelated adult videos (JAV)"),
        epilog=_(textwrap.dedent(f'''\
            {examples_header_text}
                * {example1_text}
                    {example1_command}
                * {example2_text}
                     {example2_command}
                * {example3_text}
                    {example3_command}
            ''')),
        formatter_class=utils.TranslatableHelpFormatter,
        add_help=False)

    group_general = parser.add_argument_group(_('General'))
    group_general.add_argument('--input', type=str, help=_('Path to pixelated video file or directory containing video files'))
    group_general.add_argument('--output', type=str, help=_('Path used to save output file(s). If path is a directory then file name will be chosen automatically (see --output-file-pattern). If no output path was given then the directory of the input file will be used'))
    group_general.add_argument('--output-file-pattern', type=str, default="{orig_file_name}.restored.mp4", help=_("Pattern used to determine output file name(s). Used when input is a directory, or a file but no output path was specified. Must include the placeholder '{orig_file_name}'. (default: %(default)s)"))
    group_general.add_argument('--device', type=str, default="cuda:0", help=_('Device used for running Restoration and Detection models. Use "cpu" or "cuda". If you have multiple GPUs you can select a specific one via index e.g. "cuda:0" (default: %(default)s)'))
    group_general.add_argument('--fp16', action=argparse.BooleanOptionalAction, default=torch.cuda.is_available(), help=_("Use FP16 precision for restoration and detection models. Reduces memory usage. (default: True if CUDA is available)"))
    group_general.add_argument('--list-devices', action='store_true', help=_("List available devices and exit"))
    group_general.add_argument('--version', action='store_true', help=_("Display version and exit"))
    group_general.add_argument('--help', action='store_true', help=_("Show this help message and exit"))

    export = parser.add_argument_group(_('Export (Encoder settings)'))
    export.add_argument('--codec', type=str, default="h264", help=_('FFmpeg video codec. E.g. "h264, "hevc" or "hevc_nvenc". Use "--list-codecs" to see what\'s available. (default: %(default)s)'))
    export.add_argument('--list-codecs', action='store_true', help=_("List available codecs and hardware devices / GPUs for hardware-accelerated video encoding"))
    export.add_argument('--crf', type=int, default=None, help=_('Constant rate factor (CRF). Quality setting of the video encoder. Lower values will result in higher quality but larger file sizes. If you have selected GPU codecs "h264_nvenc" or "hevc_nvenc" then the option "qp" will be used instead as those encoders don\'t support the "crf" option. (default: %(default)s)'))
    export.add_argument('--preset', type=str, default=None, help=_('Encoder preset. Mostly affects file-size and speed. (default: %(default)s)'))
    export.add_argument('--moov-front',  default=False, action=argparse.BooleanOptionalAction, help=_("Sets ffmpeg mov flags 'frag_keyframe+empty_moov+faststart'. Enables playing the output video while it's being written (default: %(default)s)"))
    export.add_argument('--custom-encoder-options', type=str, help=_("Pass arbitrary encoder options. Pass it like you'd specify them using ffmpeg. For example: --custom-encoder-options \"-rc-lookahead 32 -rc vbr_hq\". Official FFmpeg Codecs Documentation: https://ffmpeg.org/ffmpeg-codecs.html"))

    group_restoration = parser.add_argument_group(_('Mosaic Restoration'))
    group_restoration.add_argument('--mosaic-restoration-model', type=str, default="basicvsrpp", help=_("Model used to restore mosaic clips (default: %(default)s)"))
    group_restoration.add_argument('--list-mosaic-restoration-models', action='store_true', help=_("List available restoration models found in model weights directory and exit (default location is './model_weights' if not overwritten by environment variable LADA_MODEL_WEIGHTS_DIR)"))
    group_restoration.add_argument('--mosaic-restoration-model-path', type=str, default=os.path.join(MODEL_WEIGHTS_DIR, 'lada_mosaic_restoration_model_generic_v1.2.pth'), help=_("Path to restoration model weights file (default: %(default)s)"))
    group_restoration.add_argument('--mosaic-restoration-config-path', type=str, default=None, help=_("Path to restoration model configuration file. You'll not have to set this unless you're training your own custom models"))
    group_restoration.add_argument('--max-clip-length', type=int, default=180, help=_('Maximum number of frames for restoration. Higher values improve temporal stability. Lower values reduce memory footprint. If set too low flickering could appear (default: %(default)s)'))

    group_detection = parser.add_argument_group(_('Mosaic Detection'))
    group_detection.add_argument('--mosaic-detection-model-path', type=str, default=os.path.join(MODEL_WEIGHTS_DIR, 'lada_mosaic_detection_model_v3.1_fast.pt'), help=_("Path to restoration model weights file (default: %(default)s)"))
    group_detection.add_argument('--list-mosaic-detection-models', action='store_true', help=_("List available detection models found in model weights directory and exit (default location is './model_weights' if not overwritten by environment variable LADA_MODEL_WEIGHTS_DIR)"))

    group_splitting = parser.add_argument_group(_('Video Splitting (Crash Recovery)'))
    group_splitting.add_argument('--video-splitting-enabled', action=argparse.BooleanOptionalAction, default=False, help=_("Enable video splitting for crash recovery. Splits video into parts before processing and merges them after completion. Useful for long videos to avoid losing progress on crashes. (default: %(default)s)"))
    group_splitting.add_argument('--video-part-duration', type=int, default=10, help=_('Duration of each video part in minutes when splitting is enabled. (default: %(default)s)'))
    group_splitting.add_argument('--resume', action='store_true', help=_("Resume interrupted video splitting export from where it left off. Only works with --video-splitting-enabled true"))
    group_splitting.add_argument('--force-fresh-start', action='store_true', help=_("Ignore any existing resume information and start fresh. Only relevant with --video-splitting-enabled true"))

    return parser

def process_video_file(input_path: str, output_path: str, device: torch.device, mosaic_restoration_model, mosaic_detection_model,
                        mosaic_restoration_model_name, preferred_pad_mode, max_clip_length, codec, crf, moov_front, preset, custom_encoder_options,
                        video_splitting_enabled: bool = False, video_part_duration: int = 600, resume: bool = False, force_fresh_start: bool = False):
    video_metadata = get_video_meta_data(input_path)

    if video_splitting_enabled:
        # Use video splitting logic for crash recovery
        file_hash = hashlib.md5(input_path.encode()).hexdigest()[:8]
        parts_dir = os.path.join(tempfile.gettempdir(), f"lada_parts_{file_hash}")
        os.makedirs(parts_dir, exist_ok=True)

        # Check for existing resume information if resume is requested or not force fresh start
        resume_info = None
        if resume or (not force_fresh_start and not resume):
            resume_info = check_for_resume_info(input_path)
            if resume_info:
                print(f"Found resume information: part {resume_info.frame_num + 1}/{resume_info.frame_num + 1}, {resume_info.total_processing_time_s:.1f}s processing time")
            elif resume:
                print("No resume information found, starting fresh")
                resume = False

        # If force fresh start is requested, clear any existing resume info
        if force_fresh_start:
            clear_resume_info(input_path)
            resume_info = None
            print("Force fresh start requested, cleared any existing resume information")

        try:
            # Split video into parts
            part_files = video_utils.split_video_by_duration(input_path, os.path.join(parts_dir, "part_"), video_part_duration)

            # Handle resume logic
            processed_parts = []
            start_part_idx = 0
            total_time_from_previous_parts = 0.0

            if resume_info:
                start_part_idx = resume_info.frame_num  # Using frame_num to store part index
                total_time_from_previous_parts = resume_info.total_processing_time_s
                print(f"Resuming video splitting from part {start_part_idx + 1}")

                # Check for already processed parts from previous runs
                for i in range(len(part_files)):
                    expected_output = os.path.join(parts_dir, f"processed_part_{i+1:03d}.mp4")
                    if os.path.exists(expected_output):
                        processed_parts.append(expected_output)
                        print(f"Found existing processed part: {os.path.basename(expected_output)}")
                    else:
                        print(f"Expected processed part not found: {os.path.basename(expected_output)}")

                # Calculate progress based on actual part durations
                total_video_duration = video_metadata.duration
                completed_duration = 0.0

                # Sum durations of completed parts
                for i in range(start_part_idx):
                    if i < len(part_files):
                        part_metadata = get_video_meta_data(part_files[i])
                        completed_duration += part_metadata.duration

                completed_parts_fraction = completed_duration / total_video_duration if total_video_duration > 0 else 0
                print(f"Resuming with {completed_parts_fraction:.1%} progress already completed (completed {completed_duration:.1f}s of {total_video_duration:.1f}s, total time spent: {total_time_from_previous_parts:.1f}s)")

                # Save resume info for the starting part
                if start_part_idx < len(part_files):
                    resume_info_start = ResumeInformation(0, Fraction(1, 30), start_part_idx, total_time_from_previous_parts if start_part_idx > 0 else 0.0)
                    save_resume_info(input_path, resume_info_start)
                    print(f"Saved resume info for starting part: part_idx={start_part_idx}, total_time={total_time_from_previous_parts if start_part_idx > 0 else 0.0}")

            # Process remaining parts (skip parts that were already processed) - LIGHTWEIGHT LOOP
            for part_idx in range(len(processed_parts), len(part_files)):
                part_path = part_files[part_idx]
                print(f"Processing video part {part_idx + 1}/{len(part_files)}: {os.path.basename(part_path)}")

                part_output_path = os.path.join(parts_dir, f"processed_{os.path.basename(part_path)}")

                # Process this part (same logic as original processing)
                part_success = True
                part_video_metadata = get_video_meta_data(part_path)

                frame_restorer = FrameRestorer(device, part_path, max_clip_length, mosaic_restoration_model_name,
                      mosaic_detection_model, mosaic_restoration_model, preferred_pad_mode)
                part_video_tmp_file_output_path = os.path.join(tempfile.gettempdir(), f"{os.path.basename(os.path.splitext(part_output_path)[0])}.tmp{os.path.splitext(part_output_path)[1]}")
                pathlib.Path(part_output_path).parent.mkdir(exist_ok=True, parents=True)
                try:
                    frame_restorer.start()

                    with VideoWriter(part_video_tmp_file_output_path, part_video_metadata.video_width, part_video_metadata.video_height,
                                  part_video_metadata.video_fps_exact, codec=codec, crf=crf, moov_front=moov_front,
                                  time_base=part_video_metadata.time_base, preset=preset,
                                  custom_encoder_options=custom_encoder_options) as video_writer:
                        frame_restorer_progressbar = utils.Progressbar(part_video_metadata, frame_restorer)
                        for elem in frame_restorer_progressbar:
                            if elem is None:
                                part_success = False
                                print(f"Error on processing part {part_idx + 1}: frame restorer stopped prematurely")
                                break
                            (restored_frame, restored_frame_pts) = elem
                            video_writer.write(restored_frame, restored_frame_pts, bgr2rgb=True)
                            frame_restorer_progressbar.update()
                            frame_restorer_progressbar.update_time_remaining_and_speed()
                except (Exception, KeyboardInterrupt) as e:
                    part_success = False
                    if isinstance(e, KeyboardInterrupt):
                        raise e
                    else:
                        print(f"Error on processing part {part_idx + 1}", e)
                finally:
                    frame_restorer.stop()

                if part_success:
                    # Add audio to the processed part
                    audio_utils.combine_audio_video_files(part_video_metadata, part_video_tmp_file_output_path, part_output_path)
                    processed_parts.append(part_output_path)

                    # Save resume info for next part after successful processing
                    next_part_idx = part_idx + 1
                    if next_part_idx < len(part_files):
                        # Calculate total processing time so far
                        total_processing_time_so_far = 0.0
                        for i in range(next_part_idx):
                            part_metadata = get_video_meta_data(part_files[i])
                            total_processing_time_so_far += part_metadata.duration

                        resume_info_next = ResumeInformation(0, Fraction(1, 30), next_part_idx, total_processing_time_so_far)
                        save_resume_info(input_path, resume_info_next)
                        print(f"Saved resume info for next part: part_idx={next_part_idx}, total_time={total_processing_time_so_far:.1f}s")
                else:
                    print(f"Failed to process part {part_idx + 1}")
                    # Clean up temp file if it exists
                    if os.path.exists(part_video_tmp_file_output_path):
                        os.remove(part_video_tmp_file_output_path)
                    break

            # Merge processed parts if all succeeded
            if len(processed_parts) == len(part_files):
                print("Merging processed parts...")
                video_utils.merge_video_parts(processed_parts, output_path)

                # Add audio
                print(_("Processing audio"))
                audio_utils.combine_audio_video_files(video_metadata, output_path, output_path)

                # Clear resume info since export completed successfully
                clear_resume_info(input_path)
                print("Video splitting completed successfully, cleared resume information")

                success = True
            else:
                success = False

        except Exception as e:
            print(f"Error during video splitting processing: {e}")
            success = False
        finally:
            # Cleanup temporary parts
            try:
                shutil.rmtree(parts_dir)
            except:
                pass
    else:
        # Original processing logic (unchanged)
        frame_restorer = FrameRestorer(device, input_path, max_clip_length, mosaic_restoration_model_name,
                      mosaic_detection_model, mosaic_restoration_model, preferred_pad_mode)
        success = True
        video_tmp_file_output_path = os.path.join(tempfile.gettempdir(), f"{os.path.basename(os.path.splitext(output_path)[0])}.tmp{os.path.splitext(output_path)[1]}")
        pathlib.Path(output_path).parent.mkdir(exist_ok=True, parents=True)
        try:
            frame_restorer.start()

            with VideoWriter(video_tmp_file_output_path, video_metadata.video_width, video_metadata.video_height,
                             video_metadata.video_fps_exact, codec=codec, crf=crf, moov_front=moov_front,
                             time_base=video_metadata.time_base, preset=preset,
                             custom_encoder_options=custom_encoder_options) as video_writer:
                frame_restorer_progressbar = utils.Progressbar(video_metadata, frame_restorer)
                for elem in frame_restorer_progressbar:
                    if elem is None:
                        success = False
                        print("Error on export: frame restorer stopped prematurely")
                        break
                    (restored_frame, restored_frame_pts) = elem
                    video_writer.write(restored_frame, restored_frame_pts, bgr2rgb=True)
                    frame_restorer_progressbar.update()
                    frame_restorer_progressbar.update_time_remaining_and_speed()
        except (Exception, KeyboardInterrupt) as e:
            success = False
            if isinstance(e, KeyboardInterrupt):
                raise e
            else:
                print("Error on export", e)
        finally:
            frame_restorer.stop()

        if success:
            print(_("Processing audio"))
            audio_utils.combine_audio_video_files(video_metadata, video_tmp_file_output_path, output_path)
        else:
            if os.path.exists(video_tmp_file_output_path):
                os.remove(video_tmp_file_output_path)

class ResumeInformation:
    """Class to store resume information for CLI video splitting."""
    def __init__(self, frame_pts, time_base, frame_num, total_processing_time_s=0.0):
        self.frame_pts = frame_pts
        self.time_base = time_base
        self.frame_num = frame_num
        self.total_processing_time_s = total_processing_time_s

    def get_resume_timestamp_ns(self):
        """Get resume timestamp in nanoseconds."""
        return int(self.frame_pts * self.time_base.numerator * 1000000000 / self.time_base.denominator)

def get_resume_info_file_path(input_path: str) -> str:
    """Get the path to the resume info file for a given input video."""
    file_hash = hashlib.md5(input_path.encode()).hexdigest()[:8]
    parts_dir = os.path.join(tempfile.gettempdir(), f"lada_parts_{file_hash}")
    return os.path.join(parts_dir, "resume_info.json")

def check_for_resume_info(input_path: str) -> ResumeInformation | None:
    """Check if there's existing resume information for video splitting."""
    resume_file = get_resume_info_file_path(input_path)

    if os.path.exists(resume_file):
        try:
            with open(resume_file, 'r') as f:
                resume_data = json.load(f)
            resume_info = ResumeInformation(
                resume_data['frame_pts'],
                Fraction(resume_data['time_base_num'], resume_data['time_base_den']),
                resume_data['frame_num'],
                resume_data.get('total_processing_time_s', 0.0)
            )
            return resume_info
        except Exception as e:
            print(f"Failed to load resume info: {e}")
            return None
    return None

def save_resume_info(input_path: str, resume_info: ResumeInformation):
    """Save resume information to disk for video splitting."""
    resume_file = get_resume_info_file_path(input_path)
    parts_dir = os.path.dirname(resume_file)
    os.makedirs(parts_dir, exist_ok=True)

    resume_data = {
        'frame_pts': resume_info.frame_pts,
        'time_base_num': resume_info.time_base.numerator,
        'time_base_den': resume_info.time_base.denominator,
        'frame_num': resume_info.frame_num,
        'total_processing_time_s': resume_info.total_processing_time_s
    }

    try:
        with open(resume_file, 'w') as f:
            json.dump(resume_data, f)
    except Exception as e:
        print(f"Failed to save resume info: {e}")

def clear_resume_info(input_path: str):
    """Clear any existing resume information."""
    resume_file = get_resume_info_file_path(input_path)
    if os.path.exists(resume_file):
        try:
            os.remove(resume_file)
        except Exception as e:
            print(f"Failed to clear resume info: {e}")

def main():
    argparser = setup_argparser()
    args = argparser.parse_args()
    if args.version:
        print("Lada: ", VERSION)
        sys.exit(0)
    if args.list_codecs:
        utils.dump_pyav_codecs()
        sys.exit(0)
    if args.list_mosaic_detection_models:
        utils.dump_available_detection_models()
        sys.exit(0)
    if args.list_mosaic_restoration_models:
        utils.dump_available_restoration_models()
        sys.exit(0)
    if args.list_devices:
        utils.dump_torch_devices()
        sys.exit(0)
    if args.help or not args.input:
        argparser.print_help()
        sys.exit(0)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print(_("GPU {device} selected but CUDA is not available").format(device=args.device))
        sys.exit(1)
    if "{orig_file_name}" not in args.output_file_pattern or "." not in args.output_file_pattern:
        print(_("Invalid file name pattern. It must include the template string '{orig_file_name}' and a file extension"))
        sys.exit(1)
    if os.path.isdir(args.input) and args.output is not None and os.path.isfile(args.output):
        print(_("Invalid output directory. If input is a directory then --output must also be set to a directory"))
        sys.exit(1)
    if not (os.path.isfile(args.input) or os.path.isdir(args.input)):
        print(_("Invalid input. No file or directory at {input_path}").format(input_path=args.input))
        sys.exit(1)

    device = torch.device(args.device)
    mosaic_detection_model, mosaic_restoration_model, preferred_pad_mode = load_models(
        device, args.mosaic_restoration_model, args.mosaic_restoration_model_path, args.mosaic_restoration_config_path,
        args.mosaic_detection_model_path, args.fp16, args.max_clip_length
    )

    input_files, output_files = utils.setup_input_and_output_paths(args.input, args.output, args.output_file_pattern)

    single_file_input = len(input_files) == 1

    for input_path, output_path in zip(input_files, output_files):
        if not single_file_input:
            print(f"{os.path.basename(input_path)}:")
        try:
            process_video_file(input_path=input_path, output_path=output_path, device=device, mosaic_restoration_model=mosaic_restoration_model, mosaic_detection_model=mosaic_detection_model,
                                mosaic_restoration_model_name=args.mosaic_restoration_model, preferred_pad_mode=preferred_pad_mode, max_clip_length=args.max_clip_length,
                                codec=args.codec, crf=args.crf, moov_front=args.moov_front, preset=args.preset, custom_encoder_options=args.custom_encoder_options,
                                video_splitting_enabled=args.video_splitting_enabled, video_part_duration=args.video_part_duration,
                                resume=args.resume, force_fresh_start=args.force_fresh_start)
        except KeyboardInterrupt:
            print(_("Received Ctrl-C, stopping restoration."))
            break

if __name__ == '__main__':
    main()
