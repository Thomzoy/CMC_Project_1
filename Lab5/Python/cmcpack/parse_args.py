""" Parse command arguments """

import argparse
from .default import DEFAULT


def parse_args(**kwargs):
    """ Parse arguments """
    course = kwargs.pop("course", "Computational Motor Control")
    lab = kwargs.pop("lab", "Lab")
    comp = kwargs.pop("compatibility", "Python2 and Python3 compatible")
    parser = argparse.ArgumentParser(
        description="{} - {} ({})".format(course, lab, comp),
        usage="python {}".format(__file__)
    )
    parser.add_argument(
        "--save_figures", "-s",
        help="Save all figures",
        dest="save_figures",
        action="store_true"
    )
    extension_support = "png/pdf/ps/eps/svg/..."
    extension_usage = "-e png -e pdf ..."
    parser.add_argument(
        "--extension", "-e",
        help="Output extension (Formats: {}) (Usage: {})".format(
            extension_support,
            extension_usage
        ),
        dest="extension",
        action="append"
    )
    parser.add_argument(
        "--1a",
        help="Run only exercise 1a",
        dest="exo_1a",
        action="store_true"
    )
    parser.add_argument(
        "--1b",
        help="Run only exercise 1b",
        dest="exo_1b",
        action="store_true"
    )    
    parser.add_argument(
        "--1c",
        help="Run only exercise 1c",
        dest="exo_1c",
        action="store_true"
    )     
    parser.add_argument(
        "--1d",
        help="Run only exercise 1d",
        dest="exo_1d",
        action="store_true"   
    )    
    parser.add_argument(
        "--1f",
        help="Run only exercise 1f",
        dest="exo_1f",
        action="store_true"   
    )    
    args = parser.parse_args()
    DEFAULT["save_figures"] = args.save_figures
    DEFAULT["1a"] = args.exo_1a
    DEFAULT["1b"] = args.exo_1b
    DEFAULT["1c"] = args.exo_1c
    DEFAULT["1d"] = args.exo_1d
    DEFAULT["1f"] = args.exo_1f

    if args.extension:
        DEFAULT["save_extensions"] = args.extension
    return args

