import argparse
import os, errno
import random as rnd
import string
import sys
import csv
import cv2

from tqdm import tqdm
from string_generator import (
    create_strings_from_dict,
    create_strings_from_file,
    create_strings_from_wikipedia,
    create_strings_randomly
)
from data_generator import FakeTextDataGenerator
from multiprocessing import Pool


def margins(margin):
    margins = margin.split(',')
    if len(margins) == 1:
        return [margins[0]] * 4
    return [int(m) for m in margins]


def parse_arguments():
    """
        Parse the command line arguments of the program.
    """

    parser = argparse.ArgumentParser(description='Generate synthetic text data for text recognition.')
    parser.add_argument(
        "--output_dir",
        type=str,
        nargs="?",
        help="The output directory",
        default="out/",
    )
    parser.add_argument(
        "-i",
        "--input_file",
        type=str,
        nargs="?",
        help="When set, this argument uses a specified text file as source for the text",
        default=""
    )
    parser.add_argument(
        "-l",
        "--language",
        type=str,
        nargs="?",
        help="The language to use, should be fr (French), en (English), es (Spanish), de (German), or cn (Chinese).",
        default="en"
    )
    parser.add_argument(
        "-c",
        "--count",
        type=int,
        nargs="?",
        help="The number of images to be created.",
        default=1000
    )
    parser.add_argument(
        "-rs",
        "--random_sequences",
        action="store_true",
        help="Use random sequences as the source text for the generation. Set '-let','-num','-sym' to use letters/numbers/symbols. If none specified, using all three.",
        default=False
    )
    parser.add_argument(
        "-let",
        "--include_letters",
        action="store_true",
        help="Define if random sequences should contain letters. Only works with -rs",
        default=False
    )
    parser.add_argument(
        "-num",
        "--include_numbers",
        action="store_true",
        help="Define if random sequences should contain numbers. Only works with -rs",
        default=False
    )
    parser.add_argument(
        "-sym",
        "--include_symbols",
        action="store_true",
        help="Define if random sequences should contain symbols. Only works with -rs",
        default=False
    )
    parser.add_argument(
        "-w",
        "--length",
        type=int,
        nargs="?",
        help="Define how many words should be included in each generated sample. If the text source is Wikipedia, this is the MINIMUM length",
        default=1
    )
    parser.add_argument(
        "-r",
        "--random",
        action="store_true",
        help="Define if the produced string will have variable word count (with --length being the maximum)",
        default=False
    )
    parser.add_argument(
        "-f",
        "--format",
        type=int,
        nargs="?",
        help="Define the height of the produced images if horizontal, else the width",
        default=32,
    )
    parser.add_argument(
        "-t",
        "--thread_count",
        type=int,
        nargs="?",
        help="Define the number of thread to use for image generation",
        default=1,
    )
    parser.add_argument(
        "-e",
        "--extension",
        type=str,
        nargs="?",
        help="Define the extension to save the image with",
        default="jpg",
    )
    parser.add_argument(
        "-k",
        "--skew_angle",
        type=int,
        nargs="?",
        help="Define skewing angle of the generated text. In positive degrees",
        default=0,
    )
    parser.add_argument(
        "-rk",
        "--random_skew",
        action="store_true",
        help="When set, the skew angle will be randomized between the value set with -k and it's opposite",
        default=False,
    )
    parser.add_argument(
        "-wk",
        "--use_wikipedia",
        action="store_true",
        help="Use Wikipedia as the source text for the generation, using this parameter ignores -r, -n, -s",
        default=False,
    )
    parser.add_argument(
        "-bl",
        "--blur",
        type=float,
        nargs="?",
        help="Apply gaussian blur to the resulting sample. Should be an integer defining the blur radius",
        default=0,
    )
    parser.add_argument(
        "-rbl",
        "--random_blur",
        action="store_true",
        help="When set, the blur radius will be randomized between 0 and -bl.",
        default=False,
    )
    parser.add_argument(
        "-b",
        "--background",
        type=int,
        nargs="?",
        help="Define what kind of background to use. 0: Gaussian Noise, 1: Plain white, 2: Quasicrystal, 3: Pictures",
        default=0,
    )
    parser.add_argument(
        "-hw",
        "--handwritten",
        action="store_true",
        help="Define if the data will be \"handwritten\" by an RNN",
    )
    parser.add_argument(
        "-na",
        "--name_format",
        type=int,
        help="Define how the produced files will be named. 0: [TEXT]_[ID].[EXT], 1: [ID]_[TEXT].[EXT] 2: [ID].[EXT] + one file labels.txt containing id-to-label mappings",
        default=0,
    )
    parser.add_argument(
        "-d",
        "--distorsion",
        type=int,
        nargs="?",
        help="Define a distorsion applied to the resulting image. 0: None (Default), 1: Sine wave, 2: Cosine wave, 3: Random",
        default=0
    )
    parser.add_argument(
        "-do",
        "--distorsion_orientation",
        type=int,
        nargs="?",
        help="Define the distorsion's orientation. Only used if -d is specified. 0: Vertical (Up and down), 1: Horizontal (Left and Right), 2: Both",
        default=0
    )
    parser.add_argument(
        "-wd",
        "--width",
        type=int,
        nargs="?",
        help="Define the width of the resulting image. If not set it will be the width of the text + 10. If the width of the generated text is bigger that number will be used",
        default=-1
    )
    parser.add_argument(
        "-al",
        "--alignment",
        type=int,
        nargs="?",
        help="Define the alignment of the text in the image. Only used if the width parameter is set. 0: left, 1: center, 2: right",
        default=1
    )
    parser.add_argument(
        "-or",
        "--orientation",
        type=int,
        nargs="?",
        help="Define the orientation of the text. 0: Horizontal, 1: Vertical",
        default=0
    )
    parser.add_argument(
        "-tc",
        "--text_color",
        type=str,
        nargs="?",
        help="Define the text's color, should be either a single hex color or a range in the ?,? format.",
        default='#282828'
    )
    parser.add_argument(
        "-sw",
        "--space_width",
        type=float,
        nargs="?",
        help="Define the width of the spaces between words. 2.0 means twice the normal space width",
        default=1.0
    )
    parser.add_argument(
        "-m",
        "--margins",
        type=margins,
        nargs="?",
        help="Define the margins around the text when rendered. In pixels",
        default=(5, 5, 5, 5)
    )
    parser.add_argument(
        "-fi",
        "--fit",
        action="store_true",
        help="Apply a tight crop around the rendered text",
        default=False
    )
    parser.add_argument(
        "-ft",
        "--font",
        type=str,
        nargs="?",
        help="Define font to be used"
    )
    parser.add_argument(
        "-ca",
        "--case",
        type=str,
        nargs="?",
        help="Generate upper or lowercase only. arguments: upper or lower. Example: --case upper"
    )
    return parser.parse_args()


def load_dict(lang):
    """
        Read the dictionary file and returns all words in it.
    """

    lang_dict = []
    with open(os.path.join('dicts', lang + '.txt'), 'r', encoding="utf8", errors='ignore') as d:
        lang_dict = [l for l in d.read().splitlines() if len(l) > 0]
    return lang_dict


def extract_from_dic(lang_dict, word_vec):
    """
        Das
    """

    # sorted_dic = sorted(lang_dict, key=lambda line: len(line), reverse=False)
    shuffled_dic = lang_dict[:]
    rnd.shuffle(shuffled_dic)

    sentence = ' '
    for _ in word_vec:
        for w in shuffled_dic:
            if len(w) is _:
                sentence += w
                break

    return sentence


def load_fonts(lang):
    """
        Load all fonts in the fonts directories
    """

    if lang == 'cn':
        return [os.path.join('fonts/cn', font) for font in os.listdir('fonts/cn')]
    else:
        return [os.path.join('fonts/latin', font) for font in os.listdir('fonts/latin')]


def alpha_sort(file):
    """
        Rewrites csv sorting row according to the GT values, alphabetically.
    """
    with open(file, encoding="utf8", errors='ignore') as csvFile:
        reader = csv.reader(csvFile)
        headers = next(reader, None)

        sorted_list = sorted(reader, key=lambda row: row[0].lower(), reverse=False)
        # for index, column in enumerate(sorted_list):
        # print(column)

    with open(file, 'w', encoding="utf8", errors='ignore') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(headers)
        writer.writerows(sorted_list)

    return


def sentence2wordlen(sentences):
    """
        1. Takes in the sentence and outputs a vector with the length of every word in it.
        2. Splits target and from sentence and returns mod version.
    """

    wordlen, targets = [], []
    sentencesNT = []  # sentences with no target
    for sentence in sentences:
        word_vec = sentence.split()  # split to words
        targets.append(word_vec.pop())  # set last word as target
        word_len1sen = []

        sentence = ' '.join(word_vec)  # reassemble the word w/o target
        for word in word_vec:
            word_len1sen.append(len(word))
        wordlen.append(word_len1sen)
        sentencesNT.append(sentence)

    fileout = 'Wordcount_record.csv'
    with open(fileout, 'w', encoding="utf-8_sig") as csvFile:
        writer = csv.writer(csvFile)

        for _ in wordlen:
            sentence = sentencesNT[wordlen.index(_)]
            _.insert(0, sentence)
            _.insert(1, targets[wordlen.index(_)])
            _.insert(2, len(sentence))
            writer.writerow(_)

    return sentencesNT


def randomize_sentence(file):
    """
        As name implies it generates a random sentence of a given length
    """

    with open(file, 'r', encoding="utf-8_sig") as csvFile:
        read_dt = list(csv.reader(csvFile))
        random_sentences = []
        target_strings = []
        for row in read_dt:
            target_strings.append(row[1])  # take last word for now => rnd.randint
            print(row[0], ' ', len(row[0]), ' target lc', row[1][-1])
            sentence = ''
            for i in range(3, len(row)):
                wl = int(row[i])
                res = ''.join(rnd.choices(string.ascii_uppercase + string.ascii_lowercase + string.digits, k=wl))
                sentence += res
                sentence += ' '
            sentence = sentence[:-1]  # remove the last space
            random_sentences.append(sentence)
            print(sentence, ' ', len(sentence), os.linesep)

    return random_sentences, target_strings


def generate_random_get_targets(sentences):
    """
        1. Takes in the sentence and outputs a vector with the length of every word in it.
        2. Splits target and from sentence and returns both; mod version and target array.
    """

    lang_dic = []
    with open(os.path.join('dicts', 'google_en.txt'), 'r', encoding="utf8", errors='ignore') as d:
        lang_dic = [l for l in d.read().splitlines() if len(l) > 0]

    # shuffle_dict = lang_dict[:]
    wordlen, targets = [], []
    sentencesNT = []  # sentences with no target
    for sentence in sentences:
        word_vec = sentence.split()  # split to words
        targets.append(' ' + word_vec.pop())  # set last word as target
        word_len1sen = []

        sentence = ' '.join(word_vec)  # reassemble the word w/o target
        for word in word_vec:
            word_len1sen.append(len(word))
        wordlen.append(word_len1sen)
        sentencesNT.append(sentence)

    random_sentences = []  # random sentences
    for sentence in sentencesNT:

        index = sentencesNT.index(sentence)
        word_len1sen = wordlen[index]

        rnd_sentence = ''

        for _ in word_len1sen:   # shuffle_dict:
            # rnd.shuffle(shuffle_dict)
            rnd_word = rnd.choice(lang_dic)

            while len(rnd_word) != _:
                rnd_word = rnd.choice(lang_dic)

            rnd_sentence += rnd_word + ' '

        #    wl = int(_)    # consider this part if using random strings instead of words
        #    res = ''.join(rnd.choices(string.ascii_uppercase + string.ascii_lowercase + string.digits, k=wl))
        #    rnd_sentence += res + ' '

        rnd_sentence = rnd_sentence[:-1]  # remove the last space
        random_sentences.append(rnd_sentence)

    fileout = 'Wordcount_record.csv'
    with open(fileout, 'w', encoding="utf-8_sig") as csvFile:
        writer = csv.writer(csvFile)

        for _ in wordlen:
            sentence = sentencesNT[wordlen.index(_)]
            _.insert(0, sentence)
            _.insert(1, targets[wordlen.index(_)])
            _.insert(2, len(sentence))
            writer.writerow(_)

    return sentencesNT, random_sentences, targets


def determine_out_dir(index, strings_last, out_dir):
    if index >= strings_last:
        out_dir = "out_target/"
    return out_dir


def concatenate_images():
    """
        rtert
    """

    # Folders: (sentence, target) => (output)
    sent_folder = os.path.join(os.getcwd(), 'out/')
    target_folder = os.path.join(os.getcwd(), 'out_target/')
    random_folder = os.path.join(os.getcwd(), 'out_random/')

    try:
        os.makedirs(os.path.join(os.getcwd(), 'out_concat/'))
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    concat_folder = os.path.join(os.getcwd(), 'out_concat/')

    try:
        os.makedirs(os.path.join(os.getcwd(), 'out_concat_rand/'))
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    concat_rand_folder = os.path.join(os.getcwd(), 'out_concat_rand/')

    sent_files = [f for f in os.listdir(sent_folder) if not f.startswith('.')]
    target_files = [f for f in os.listdir(target_folder) if not f.startswith('.')]
    random_files = [f for f in os.listdir(random_folder) if not f.startswith('.')]

    # sort by number @ '#.jpg'  --> key: lastchar: lastchar[-5:] when name is used
    sorted_sent = sorted(sent_files, key=lambda w: int(w[:-4]), reverse=False)
    sorted_target = sorted(target_files, key=lambda w: int(w[:-4]), reverse=False)
    sorted_random = sorted(random_files, key=lambda w: int(w[:-4]), reverse=False)

    Sort_sent_paths = [sent_folder + file for file in sorted_sent]
    Sort_target_paths = [target_folder + file for file in sorted_target]
    Sort_random_paths = [random_folder + file for file in sorted_random]

    # print(sorted_randFil[0], os.linesep, sorted_randFil[len(sent_files)-1])
    # input("debug")

    for _ in range(len(sent_files)):
        # sentence stems, (normal and random)
        img_left = cv2.imread(Sort_sent_paths[_])
        img_left_rand = cv2.imread(Sort_random_paths[_])

        # targets
        img_right = cv2.imread(Sort_target_paths[_])

        # horizontal concatenate
        img_h = cv2.hconcat([img_left, img_right])
        img_h_random = cv2.hconcat([img_left_rand, img_right])

        cv2.imwrite(concat_folder + "concat_{}.jpg".format(_), img_h)
        cv2.imwrite(concat_rand_folder + "concat_random_{}.jpg".format(_), img_h_random)

    return


def main():
    """
        Description: Main function
    """

    # Argument parsing
    args = parse_arguments()

    # Create the directory if it does not exist.
    try:
        os.makedirs(args.output_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    # Creating word list
    lang_dict = load_dict(args.language)

    # Create font (path) list
    if not args.font:
        fonts = load_fonts(args.language)
    else:
        if os.path.isfile(args.font):
            fonts = [args.font]
        else:
            sys.exit("Cannot open font")

    # Creating synthetic sentences (or word)
    strings, targets = [], []
    random_sentence, orig = [], []
    inp = 'n'

    if args.use_wikipedia:
        strings = create_strings_from_wikipedia(args.length, args.count, args.language)
        print("sourcing from random wiki page...", args.use_wikipedia)
        input("Press enter...")
    elif args.input_file != '':
        strings = create_strings_from_file(args.input_file, args.count)
        print("sourcing from input file: ", args.input_file)

        print(args.language)
        if args.language == 'cn':

            KanjiArray, StrokeArray = [], []
            for strobj in strings:
                kanji, stroke = strobj.split(",")
                KanjiArray.append(kanji)
                StrokeArray.append(stroke)

            strings = KanjiArray[:]
            print("entered ", KanjiArray[len(KanjiArray)-1], strings[len(KanjiArray)-1])

        inp = input("NLP test? (y/n) ")
        if inp is 'y':
            # file = os.path.join(os.getcwd(), 'Wordcount_record.csv')

            print('Starting NLP-test' + os.linesep + '    ** generating sentences...')

            strings, random_sentence, targets = generate_random_get_targets(strings)

            print(strings[0] + targets[0])
            print(random_sentence[0] + targets[0])
            print('    ** generating random sentences and retrieving targets...')
            string_last = len(strings)

            input("Press enter to generate images...")

    elif args.random_sequences:
        strings = create_strings_randomly(args.length, args.random, args.count,
                                          args.include_letters, args.include_numbers, args.include_symbols,
                                          args.language)
        # Set a name format compatible with special characters automatically if they are used
        if args.include_symbols or True not in (args.include_letters, args.include_numbers, args.include_symbols):
            args.name_format = 2
    else:
        strings = create_strings_from_dict(args.length, args.random, args.count, lang_dict)

    if args.case == 'upper':
        strings = [x.upper() for x in strings]
    if args.case == 'lower':
        strings = [x.lower() for x in strings]

    string_count = len(strings)

    # Store random values used at FakeTextDataGenerator Class
    RandNums = []

    #  Apply effect onto targets first then clean args
    if inp is 'y':
        orig = strings
        strings = targets
        args.output_dir = "out_target/"
        args.margins = (5, 0, 5, 5)
        args.name_format = 2
        try:
            os.makedirs(args.output_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        print('    ** applying effects onto targets...')

    p = Pool(args.thread_count)
    for _ in tqdm(p.imap_unordered(
            FakeTextDataGenerator.generate_from_tuple,
            zip(
                [i for i in range(0, string_count)],
                strings,
                [fonts[rnd.randrange(0, len(fonts))] for _ in range(0, string_count)],
                # [determine_out_dir(i, string_last, args.output_dir) for i in range(0, string_count)],
                [args.output_dir] * string_count,
                [args.format] * string_count,
                [args.extension] * string_count,
                [args.skew_angle] * string_count,
                [args.random_skew] * string_count,
                [args.blur] * string_count,
                [args.random_blur] * string_count,
                [args.background] * string_count,
                [args.distorsion] * string_count,
                [args.distorsion_orientation] * string_count,
                [args.handwritten] * string_count,
                [args.name_format] * string_count,
                [args.width] * string_count,
                [args.alignment] * string_count,
                [args.text_color] * string_count,
                [args.orientation] * string_count,
                [args.space_width] * string_count,
                [args.margins] * string_count,
                [args.fit] * string_count
            )
    ), total=args.count):
        RandNums.append(_)
        pass
    p.terminate()

    if inp is 'y':
        strings = orig
        print('    ** generating clean images for sentences without target...')
        args.output_dir = "out/"
        args.margins = (5, 5, 5, 0)

        # store and reset effect variables at args
        orig_blur = args.blur
        orig_random_blur = args.random_blur
        args.blur = 0
        args.background = 1
        args.random_blur = False

        p = Pool(args.thread_count)
        for _ in tqdm(p.imap_unordered(
                FakeTextDataGenerator.generate_from_tuple,
                zip(
                    [i for i in range(0, string_count)],
                    strings,
                    [fonts[rnd.randrange(0, len(fonts))] for _ in range(0, string_count)],
                    # [determine_out_dir(i, string_last, args.output_dir) for i in range(0, string_count)],
                    [args.output_dir] * string_count,
                    [args.format] * string_count,
                    [args.extension] * string_count,
                    [args.skew_angle] * string_count,
                    [args.random_skew] * string_count,
                    [args.blur] * string_count,
                    [args.random_blur] * string_count,
                    [args.background] * string_count,
                    [args.distorsion] * string_count,
                    [args.distorsion_orientation] * string_count,
                    [args.handwritten] * string_count,
                    [args.name_format] * string_count,
                    [args.width] * string_count,
                    [args.alignment] * string_count,
                    [args.text_color] * string_count,
                    [args.orientation] * string_count,
                    [args.space_width] * string_count,
                    [args.margins] * string_count,
                    [args.fit] * string_count
                )
        ), total=args.count):
            # RandNums.append(_)
            pass
        p.terminate()

        rand_sent = random_sentence[:]
        strings = random_sentence
        print('    ** generating clean images out of random sentences...')
        args.output_dir = "out_random/"
        try:
            os.makedirs(args.output_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        p = Pool(args.thread_count)
        for _ in tqdm(p.imap_unordered(
                FakeTextDataGenerator.generate_from_tuple,
                zip(
                    [i for i in range(0, string_count)],
                    strings,
                    [fonts[rnd.randrange(0, len(fonts))] for _ in range(0, string_count)],
                    # [determine_out_dir(i, string_last, args.output_dir) for i in range(0, string_count)],
                    [args.output_dir] * string_count,
                    [args.format] * string_count,
                    [args.extension] * string_count,
                    [args.skew_angle] * string_count,
                    [args.random_skew] * string_count,
                    [args.blur] * string_count,
                    [args.random_blur] * string_count,
                    [args.background] * string_count,
                    [args.distorsion] * string_count,
                    [args.distorsion_orientation] * string_count,
                    [args.handwritten] * string_count,
                    [args.name_format] * string_count,
                    [args.width] * string_count,
                    [args.alignment] * string_count,
                    [args.text_color] * string_count,
                    [args.orientation] * string_count,
                    [args.space_width] * string_count,
                    [args.margins] * string_count,
                    [args.fit] * string_count
                )
        ), total=args.count):
            # RandNums.append(_)
            pass
        p.terminate()

        args.blur = orig_blur
        args.random_blur = orig_random_blur

        # concatenate images and go back to original string
        concatenate_images()
        for index in range(len(strings)):
            rand_sent[index] += targets[index]
            strings[index] = orig[index] + targets[index]

    # print(RandNums, os.linesep)

    if args.name_format == 2:
        # Create file with filename-to-label connections
        with open(os.path.join(args.output_dir, "labels.txt"), 'w', encoding="utf-8_sig") as f:
            for i in range(string_count):
                file_name = str(i) + "." + args.extension
                f.write("{} {}\n".format(file_name, strings[i]))

    # Write to Csv
    tag_array = []
    for arg in vars(args):
        tag_array.append(arg)
    tag_array.insert(0, "Content")
    tag_array.insert(2, "File name")
    if args.language == 'cn':
        tag_array.insert(3, "Num of strokes")

    fileout = 'PyCsvExp.csv'
    with open(fileout, 'w', encoding="utf-8_sig") as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(tag_array)

        row_keeper = []
        for i in range(string_count):
            attr_array = []

            # retrieve rand from class
            # print(FakeTextDataGenerator.generate_from_tuple())

            for arg in vars(args):
                if arg is 'blur':
                    if args.random_blur is False:
                        RandNums[i][0] = args.blur
                    attr_array.append(RandNums[i][0])
                elif arg is 'length':
                    attr_array.append(len(strings[i]))
                else:
                    attr_array.append(getattr(args, arg))
                # print(arg, getattr(args, arg))
            attr_array.insert(0, strings[i])
            if inp is 'y':
                attr_array.insert(2, "concat_" + RandNums[i][2])
            else:
                attr_array.insert(2, RandNums[i][2])

            if args.language == 'cn':
                attr_array.insert(3, StrokeArray[i])

            # writer.writerow(attr_array)
            row_keeper.append(attr_array)

        if inp is 'y':
            sorted_rows = sorted(row_keeper, key=lambda row: int(row[2][:-4].strip("concat_")),
                                 reverse=False)  # row[0].lower()
        else:
            sorted_rows = sorted(row_keeper, key=lambda row: int(row[2].strip(".jpg")),
                                 reverse=False)  # row[0].lower()

        writer.writerows(sorted_rows)
        csvFile.close()

    if inp is 'y':
        # Write to Csv
        tag_array = []
        for arg in vars(args):
            tag_array.append(arg)
        tag_array.insert(0, "Content")
        tag_array.insert(2, "file name")

        fileout = 'PyCsvExpRandom.csv'
        with open(fileout, 'w', encoding="utf-8_sig") as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(tag_array)

            row_keeper = []
            for i in range(string_count):
                attr_array = []

                # retrieve rand from class
                # print(FakeTextDataGenerator.generate_from_tuple())

                for arg in vars(args):
                    if arg is 'blur':
                        if args.random_blur is False:
                            RandNums[i][0] = args.blur
                        attr_array.append(RandNums[i][0])
                    elif arg is 'length':
                        attr_array.append(len(rand_sent[i]))
                    else:
                        attr_array.append(getattr(args, arg))
                    # print(arg, getattr(args, arg))
                attr_array.insert(0, rand_sent[i])
                attr_array.insert(2, "concat_random_" + RandNums[i][2])
                # writer.writerow(attr_array)
                row_keeper.append(attr_array)

            sorted_rows = sorted(row_keeper, key=lambda row: int(row[2][:-4].strip("concat_random_")), reverse=False)  # row[0].lower()
            writer.writerows(sorted_rows)
            csvFile.close()


if __name__ == '__main__':
    main()
