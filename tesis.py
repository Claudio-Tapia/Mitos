import argparse
import os
import random
import sys

from common import utils
from common.Params import Params as P
from mitosCalsification import Preprocess
from mitos_extract_anotations import candidateSelection as cs

filter = ['*.bmp', '*.png', '*.jpg']

class Mitos_argumment_parser(argparse.ArgumentParser):
    def error(self, message):
        # sys.stderr(message)
        self.print_help()
        sys.exit(-1)

# http://stackoverflow.com/questions/23936145/python-argparse-help-message-disable-metavar-for-short-options
class CustomFormatter(argparse.HelpFormatter):
    def _format_action_invocation(self, action):
        if not action.option_strings:
            metavar, = self._metavar_formatter(action, action.dest)(1)
            return metavar
        else:
            parts = []
            # if the Optional doesn't take a value, format is:
            #    -s, --long
            if action.nargs == 0:
                parts.extend(action.option_strings)

            # if the Optional takes a value, format is:
            #    -s ARGS, --long ARGS
            # change to
            #    -s, --long ARGS
            else:
                default = action.dest.upper()
                args_string = self._format_args(action, default)
                for option_string in action.option_strings:
                    #parts.append('%s %s' % (option_string, args_string))
                    parts.append('%s' % option_string)
                parts[-1] += ' %s'%args_string
            return ', '.join(parts)

def extract_candidates(args):
    """
    Set the params for extracting candidates from the specified folder.
    The candidates are separated in training and testing and saved in their
    corresponding folders
    :param args: namespace that contains the params entered by the user
    """
    if args.custom_folder is None:
        folder_path= P().normHeStainDir
    else:
        folder_path = args.custom_folder
        if not os.path.isdir(folder_path):
            raise FileNotFoundError('The path does not exist')

    # split the files in training and testing
    file_list = utils.listFiles(folder_path, filter)
    train_count = len(file_list)- args.number_test_img
    # train_list = file_list [0:train_count]
    # test_list = file_list [- args.number_test_img:]

    # selects a sample of random files in the list and
    # save  some for validation and the rest for training
    selection_index = random.sample(range(len(file_list)), k=args.number_test_img)
    test_list = [file_list[i] for i in selection_index]
    for i in sorted(selection_index, reverse=True):
        del file_list[i]

    train_params = cs.Candidates_extractor_params(file_list)
    test_params = cs.Candidates_extractor_params(test_list)

    if args.dont_save:
        train_params.write_img_to_disk = False
        test_params.write_img_to_disk = False

    #test_params.write_img_to_disk = False
    if args.save_img_keypoint:
        train_params.bsave_img_keypoints = True
        test_params.bsave_img_keypoints = True

    # specific params for testing
    test_params.save_candidates_dir_path = P().saveTestCandidates
    test_params.save_mitosis_dir_path = P().saveTestMitos
    # test_params.save_mitosis_dir_path = None
    test_params.candidates_json_save_path = P().candidatesTestJsonPath

    train_extractor = cs.Candidates_extractor(train_params)
    test_extractor = cs.Candidates_extractor(test_params)

    train_extractor.extract()
    if args.number_test_img > 0:
        test_extractor.extract()

def pre_process(args):
    Preprocess.pre_process()

def train(args):
    from mitosCalsification.mitosClasificator import train_model
    train_model(args.r, args.R)

def test(args):
    from mitosCalsification.mitosClasificator import test_model
    test_model()

def crossvalidation(args):
    from mitosCalsification.crossval import crossval
    crossval()

def config_extract_cand_parser(parser):
    parser.allow_abbrev = False
    parser.formatter_class = CustomFormatter
    parser.usage= 'python tesis.py extract [options]'
    parser.description='Description: Extract candidates from images'
    parser.add_argument('-c','--custom-folder',
                        help='Use the specified folder,'
                             ' instead of the folder in the config file',
                        action='store', default= None, metavar='<dirPath>')
    parser.add_argument('-n', '--number-test-img', help='number of testing images. Default = 5',
                        action='store', default = 5, type=int, metavar='<Number>')
    parser.add_argument('-d', '--dont-save', help='Do not save the extracted images to disk',
                        action='store_true')
    # parser.add_argument('--cand', help='Extract only candidates,'
    #                                    ' do not validate against annotated mitosis',
    #                     action='store_true')
    parser.add_argument('-k','--save-img-keypoint', help='save to disk the images with printed candidates keypoints',
                        action='store_true')
    parser.set_defaults(func=extract_candidates)

def config_train_parser(parser):
    parser.allow_abbrev = False
    parser.set_defaults(func=train)
    parser.formatter_class = CustomFormatter
    parser.usage = 'python tesis.py train [options]'
    parser.description = 'Description: Train a classificator for classify between mitotic and no mitotic cells'
    parser.add_argument('--custom-folder','-c', help='Use the specified folder, instead of the folder in '
                                                     'the config file', metavar='<dirPath>')
    parser.add_argument('-r', help='Ratio between mitosis class and no mitosis class picked for training.'
                                   ' Default = 1',
                        metavar='<Ratio>', type=float, default= 1)
    parser.add_argument('-R', help='Use all the training samples, ignores the -r flag',
                        action='store_true')

def config_pre_process_parser(parser):
    parser.set_defaults(func=pre_process)
    # TODO: add more options to pre process

def config_test_parser(parser):
    parser.set_defaults(func=test)

parser = Mitos_argumment_parser(formatter_class= CustomFormatter)
parser.usage= 'python %(prog)s <command> [options] '
parser.allow_abbrev= False
subparsers = parser.add_subparsers(title='Commands', metavar='')

extract_cand_parser = subparsers.add_parser('extract', help='Extract candidates from images')
config_extract_cand_parser(extract_cand_parser)

train_parser = subparsers.add_parser('train', help='Train the classificator')
config_train_parser(train_parser)

test_parser = subparsers.add_parser('evaluate', help='Run the current classificator with test samples')
config_test_parser(test_parser)

pre_process_parser = subparsers.add_parser('pretrain', help='Create more training samples rotating the existing ones')
config_pre_process_parser(pre_process_parser)

cross_fold_parser = subparsers.add_parser('crossval', help='10-fold cross validation')
cross_fold_parser.set_defaults(func=crossvalidation)

argc = len(sys.argv)
if argc <= 1:
    parser.print_help()
else:
    args = parser.parse_args()
    args.func(args)

if __name__ == '__main__':
    # pycharm didnt show the option to run script :(
    pass
