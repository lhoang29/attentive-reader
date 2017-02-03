#!/usr/bin/env python

""" QA Preprocessing """

import numpy as np
import h5py
import argparse
import sys
import re
import codecs
import copy
import operator
import json
import glob, os
import csv
import re

GLOVE_PATH = "data/glove.6B.50d.txt"
START_TOKEN = "<s>"
END_TOKEN = "</s>"
PAD_TOKEN = "PADDING"
RARE_TOKEN = "RARE"

MAX_STORY = 0
MAX_QUESTION = 0
MAX_FACT = 0

NUM_TASKS = 20

args = {}

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def get_suffix(w):
    if len(w) < 2:
        return w
    return w[-2:]


def get_prefix(w):
    if len(w) < 2:
        return w
    return w[:2]


def get_cap_index(word):
    cap_idx = 6

    if word.islower(): # all low caps
        cap_idx = 2
    elif word.isupper(): # all upper caps
        cap_idx = 3
    elif word[0].isupper(): # first letter cap
        cap_idx = 4
    elif sum(int(c.isupper()) for c in word) == 1: # one cap
        cap_idx = 5
    else: # all other cases
        cap_idx = 6
    return cap_idx


def init_vocab():
    return { RARE_TOKEN : 1, PAD_TOKEN : 2, START_TOKEN: 3, END_TOKEN: 4 }


def write_dict(file, dict):
    sorted_dict = sorted(dict.items(), key=operator.itemgetter(1))
    writer = csv.writer(open(file, 'wb'))
    for key, value in sorted_dict:
       writer.writerow([key, value])


def get_vocab(folder, word_to_idx = {}):
    if len(word_to_idx) == 0:
        word_to_idx = init_vocab()

    idx = len(word_to_idx) + 1
    owd = os.getcwd()
    os.chdir(folder)
    for infile in glob.glob("*.txt"):
        with open(infile) as inf:
            for line in inf:
                parts = line.lower().replace('.','').strip().split('?')
                tokens = parts[0].split(' ')
                for i in np.arange(1, len(tokens)):
                    if tokens[i] not in word_to_idx and not is_number(tokens[i]):
                        word_to_idx[tokens[i]] = idx
                        idx += 1
                if len(parts) > 1:
                    answer = parts[1].strip().split('\t')[0]
                    if answer not in word_to_idx:
                        word_to_idx[answer.lower()] = idx
                        idx += 1
    os.chdir(owd)
    return word_to_idx


# might be useful later
def get_vocab_embedding(vocab_size = 500000):
    word_to_idx = init_vocab()
    idx = 5

    suffix_to_idx = { get_suffix(
        RARE_TOKEN) : 1, get_suffix(PAD_TOKEN) : 2, get_suffix(
        START_TOKEN): 3, get_suffix(END_TOKEN) : 4 }
    idx_suf = 5

    prefix_to_idx = { get_prefix(
        RARE_TOKEN) : 1, get_prefix(PAD_TOKEN) : 2, get_prefix(
        START_TOKEN): 3, get_prefix(END_TOKEN) : 4 }
    idx_pre = 5

    embeddings = [np.zeros(50) for _ in range(4)]
    with codecs.open(GLOVE_PATH, "r", encoding="utf-8") as gf:
        for line in gf:
            tokens = line.split(' ')
            word = tokens[0]
            embedding = np.array(tokens[1:]).astype(float)
            embeddings.append(embedding)
            word_to_idx[word] = idx
            suf = get_suffix(word)
            if suf not in suffix_to_idx:
                suffix_to_idx[suf] = idx_suf
                idx_suf += 1
            pre = get_prefix(word)
            if pre not in prefix_to_idx:
                prefix_to_idx[pre] = idx_pre
                idx_pre += 1
            if idx - 2 >= vocab_size: # don't count special words
                break
            idx += 1
    return word_to_idx, suffix_to_idx, prefix_to_idx, np.array(embeddings)


def normalize(data, word_to_idx):
    for t in np.arange(NUM_TASKS) + 1:
        stories = data[t]['stories']
        markers = data[t]['markers']
        questions = data[t]['questions']
        answers = data[t]['answers']
        facts = data[t]['facts']

        pad_idx = word_to_idx[PAD_TOKEN]
        norm_stories = np.ones((len(stories), MAX_STORY)) * pad_idx
        norm_markers = np.zeros((len(stories), MAX_STORY))
        norm_questions = np.ones((len(questions), MAX_QUESTION)) * pad_idx
        norm_answers = np.array(answers)
        norm_facts = np.zeros((len(facts), MAX_FACT))

        for i in range(len(stories)):
            norm_stories[i, : len(stories[i])] = stories[i]
            norm_markers[i, : len(markers[i])] = markers[i]
        for i in range(len(questions)):
            norm_questions[i, : len(questions[i])] = questions[i]
        for i in range(len(facts)):
            norm_facts[i, : len(facts[i])] = facts[i]

        data[t]['stories'] = norm_stories
        data[t]['markers'] = norm_markers
        data[t]['questions'] = norm_questions
        data[t]['answers'] = norm_answers
        data[t]['facts'] = norm_facts


def process_answer(label, task_no, word_to_idx):
    '''
    Process output label for each sentence depending on the task number.
    '''
    parts = label.strip().split('\t')
    # TODO: ignore caps in answer?
    return word_to_idx[parts[0].lower()], parts[1].split(' ')


def process(file, task_no, word_to_idx):
    '''
    Process one single QA file (either train or test).
    '''
    global MAX_STORY
    global MAX_QUESTION
    global MAX_FACT

    all_stories = []
    all_markers = [] # line number for each word in story
    all_questions = []
    all_answers = []
    all_facts = []

    current_story = []
    current_markers = []

    with open(file) as inf:
        for line in inf:
            line_info = re.match('([0-9].*?)\ (.*)', line)
            line_no = int(line_info.group(1)) # line number
            line_data = line_info.group(2) # rest of line

            parts = line_data.split('?')

            # parse the first part, either statement or question
            statement = parts[0].strip().replace('.', '').split(' ')
            words = [START_TOKEN] + [w.lower() for w in statement] + [END_TOKEN]

            if line_no == 1: # start of new story
                if len(current_story) > 0 and len(current_markers) > 0:
                    MAX_STORY = max(MAX_STORY, len(current_story))
                    current_story = [] # end of story, start over
                    current_markers = [] # start over

            if len(parts) > 1: # is a question
                all_stories.append(copy.copy(current_story))
                all_markers.append(copy.copy(current_markers))
                    
                MAX_QUESTION = max(MAX_QUESTION, len(words))
                all_questions.append([word_to_idx[w] for w in words]) # append to question list

                answer, facts = process_answer(parts[1], task_no, word_to_idx)
                all_answers.append(answer)

                MAX_FACT = max(MAX_FACT, len(facts))
                all_facts.append(facts)
            else: # is not a question
                for w in words:
                    current_story.append(word_to_idx[w]) # append to story
                    current_markers.append(line_no)

    return {
    'stories': all_stories, 'markers': all_markers, 'questions': all_questions,
    'answers': all_answers, 'facts': all_facts }


def process_files(folder, word_to_idx):
    '''
    Process all QA files in specified folder.
    '''
    trains = {}
    tests = {}
    tasks = ['' for t in range(NUM_TASKS)] # task names

    owd = os.getcwd()
    os.chdir(folder)
    for infile in glob.glob("*.txt"):
        file_info = re.match('qa(.*)_(.*)_(.*).txt', infile)
        task_no = int(file_info.group(1)) # from 1 to 20
        task_name = file_info.group(2) # e.g. yes-no-questions, basic-induction
        task_data_type = file_info.group(3) # train or test

        # process data
        processed_data = process(infile, task_no, word_to_idx)
        if task_data_type == 'train':
            trains[task_no] = processed_data
        else:
            tests[task_no] = processed_data

        # add the task name
        tasks[task_no - 1] = task_name

    os.chdir(owd)

    return trains, tests, tasks


def main(arguments):
    global args
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-vocabsize', help="vocabsize",
                        type=long,default=500000,required=False)
    parser.add_argument('-dir', help="data directory",
                        type=str,default='babi_data/en/',required=False)
    args = parser.parse_args(arguments)

    # get embeddings
    # word_to_idx, suffix_to_idx, prefix_to_idx, embeddings = get_vocab_embedding(args.vocabsize)

    word_to_idx = get_vocab(args.dir)
    write_dict('word_to_idx.csv', word_to_idx) # for debugging purposes

    trains, tests, tasks = process_files(args.dir, word_to_idx)
    normalize(trains, word_to_idx)
    normalize(tests, word_to_idx)

    for t in np.arange(NUM_TASKS) + 1:
        filename = 'qa{0:02d}.hdf5'.format(t)
        with h5py.File(filename, "w") as f:
            f['train_stories'] = trains[t]['stories']
            f['train_markers'] = trains[t]['markers']
            f['train_questions'] = trains[t]['questions']
            f['train_answers'] = trains[t]['answers']
            f['train_facts'] = trains[t]['facts']

            f['test_stories'] = tests[t]['stories']
            f['test_markers'] = tests[t]['markers']
            f['test_questions'] = tests[t]['questions']
            f['test_answers'] = tests[t]['answers']
            f['test_facts'] = tests[t]['facts']

            f['nwords'] = np.array([len(word_to_idx)], dtype=np.int32)
            # f['word_embeddings'] = embeddings

            f['idx_pad'] = np.array([word_to_idx[PAD_TOKEN]], dtype=np.int32)
            f['idx_rare'] = np.array([word_to_idx[RARE_TOKEN]], dtype=np.int32)
            f['idx_start'] = np.array([word_to_idx[START_TOKEN]], dtype=np.int32)
            f['idx_end'] = np.array([word_to_idx[END_TOKEN]], dtype=np.int32)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
