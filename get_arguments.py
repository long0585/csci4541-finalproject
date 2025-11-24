# Settings
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--sentence', action='store_true', help="Present problem in sentence format.", default=False)
parser.add_argument('--noprompt', action='store_true', help="Present problem without prompt.", default=False)
parser.add_argument('--replica', action='store_true', help="Replicate results", default=False)
parser.add_argument('--subset', action='store_true', help="Use subset", default=True)
parser.add_argument('--no-subset', dest='subset', action='store_false', help="Do not use subset")

parser.add_argument('--modified', action='store_true', help="Use modified problems", default=False) # todo
parser.add_argument('--no-modified', dest='modified', action='store_false', help="Do not use modified problems")
parser.add_argument('--synthetic', action='store_true', help="Use synthetic alphabet", default=True)
parser.add_argument('--no-synthetic', dest='synthetic', action='store_false', help="Do not use synthetic alphabet")
parser.add_argument('--alphabetprompt', action='store_true', help="Inject alphabet into prompt", default=True)
parser.add_argument('--no-alphabet-prompt', dest='alphabetprompt', action='store_false', help="Do not inject alphabet into prompt")

parser.add_argument('--customalphabet',type=str, default='a b c d e f g h i j k l m n o p q r s t u v w x y z', help='Choose custom alphabet')

args = parser.parse_args()

alphabet_suffix = args.customalphabet.replace(" ", "")

def get_suffix_modified(args):

    if args.synthetic:
        if args.modified is False:
            suffix = '_' + alphabet_suffix + '_interval1'
        else:
            suffix = '_' + alphabet_suffix + '_interval2'
    elif args.alphabetprompt:
        suffix = '_modified_prompt'
    else:
        suffix = '_modified'
    return suffix


def get_suffix(args):
    save_fname = ''
    if args.sentence:
        save_fname += '_sentence'
    if args.noprompt:
        save_fname += '_noprompt'

    if args.modified or args.synthetic:
        save_fname += get_suffix_modified(args)
    elif args.replica:
        save_fname += '_replica'
    if args.subset:
        save_fname += '_subset'

    return save_fname


def get_suffix_problems(args):
    save_fname = ''
    if args.modified:
        save_fname += get_suffix_modified(args)
    return save_fname


def get_prob_types(args, all_prob, prob_types):
    if args.subset and not args.modified:  # todo get subset from modified
        pass  # prob_types = ['succ'] # define subset
    return prob_types


def get_version_dir(args):
    dir_d = {'_modified': '1_real', '_modified_prompt': '2_real_prompt', '_' + alphabet_suffix + '_interval2': 'all_prob_int2',
             '_' + alphabet_suffix + '_interval1': 'all_prob_int1'}
    suffix = get_suffix_modified(args)
    vrs = dir_d[suffix]
    # vrs_dir = os.path.join('GPT3_results_modified_versions', vrs)
    # return vrs_dir
    return vrs
