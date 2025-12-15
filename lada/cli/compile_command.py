# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import argparse
from gettext import gettext as _
import os
import sys
import textwrap

import torch

from lada import RESTORATION_MODEL_NAMES_TO_FILES
from lada.cli import utils
from lada.utils.os_utils import gpu_has_tensor_cores


def add_compile_subparser(subparsers: argparse._SubParsersAction) -> None:
    examples_header_text = _("Examples:")
    example_text = _("Compile a restoration model to TensorRT:")
    example_command = _("%(prog)s compile --mosaic-restoration-model basicvsrpp-v1.2 --max-clip-size 30")

    parser = subparsers.add_parser(
        "compile",
        help=_("Compile a restoration model to TensorRT. Larger clip sizes will require more VRAM and increase compilation time."),
        description=_("Compile a restoration model to TensorRT"),
        formatter_class=utils.TranslatableHelpFormatter,
        add_help=False,
        epilog=_(
            textwrap.dedent(
                f"""\
                {examples_header_text}
                    * {example_text}
                        {example_command}
                """
            )
        ),
    )

    group_compile = parser.add_argument_group(_("Compile"))
    group_compile.add_argument(
        "--mosaic-restoration-model",
        type=str,
        default="basicvsrpp-v1.2",
        help=_('Name of detection model or path to model weights file. Use "--list-mosaic-restoration-models" to see what\'s available.'),
    )
    group_compile.add_argument(
        "--clip-size",
        type=int,
        default=30,
        help=_("Clip size used for compilation (default: %(default)s)"),
    )
    group_compile.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help=_('Device used for compilation. Use "cuda". If you have multiple GPUs you can select a specific one via index e.g. "cuda:0" (default: %(default)s)'),
    )
    group_compile.add_argument(
        "--fp16",
        action=argparse.BooleanOptionalAction,
        default=gpu_has_tensor_cores(),
        help=_("Enable fp16 compilation when supported (default: %(default)s)"),
    )
    group_compile.add_argument(
        "--mosaic-restoration-config-path",
        type=str,
        default=None,
        help=_("Path to restoration model configuration file. You'll not have to set this unless you're training your own custom models"),
    )
    group_compile.add_argument(
        "--help",
        action="help",
        default=argparse.SUPPRESS,
        help=_("Show this help message and exit"),
    )

    parser.set_defaults(func=compile_main, _parser=parser)


def _resolve_restoration_model(arg: str) -> tuple[str, str, str]:
    if arg in utils.get_available_restoration_models():
        model_name = arg
        model_path = RESTORATION_MODEL_NAMES_TO_FILES[arg]
        out_stem = model_name
        return model_name, model_path, out_stem

    if os.path.isfile(arg):
        model_path = arg
        model_name = "basicvsrpp"
        out_stem = os.path.splitext(os.path.basename(model_path))[0]
        return model_name, model_path, out_stem

    print(_("Invalid mosaic restoration model"))
    sys.exit(1)


def compile_main(args: argparse.Namespace) -> None:
    if args.help:
        args._parser.print_help()
        sys.exit(0)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print(_("GPU {device} selected but CUDA is not available").format(device=args.device))
        sys.exit(1)

    mosaic_restoration_model_name, mosaic_restoration_model_path, output_stem = _resolve_restoration_model(args.mosaic_restoration_model)

    if not mosaic_restoration_model_name.startswith("basicvsrpp"):
        print(_("Only BasicVSR++ restoration models support TensorRT compilation"))
        sys.exit(1)

    device = torch.device(args.device)

    from lada.models.basicvsrpp.inference import load_model
    from lada.restorationpipeline.basicvsrpp_mosaic_restorer import BasicvsrppMosaicRestorer

    model = load_model(args.mosaic_restoration_config_path, mosaic_restoration_model_path, device, args.fp16, args.clip_size)
    restorer = BasicvsrppMosaicRestorer(model, device, args.fp16, args.clip_size)
    output_path = restorer.compile(model_name=output_stem, max_clip_size=args.clip_size)
    print(f"Compilation completed successfully. Saved to {output_path}")
