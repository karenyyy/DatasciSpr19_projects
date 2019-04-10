def read_files(filepaths):
    sentence_list = []
    file_list = []

    for filepath in filepaths:
        file_list.append(open(filepath))

    for lines in zip(*file_list):
        for i, line in enumerate(lines):
            line = line.rstrip("\n").split(" ")
            sentence_list.append(line)

        yield sentence_list
        sentence_list = []


def gen_vocab(filepaths, vocab_file):
    vocab = {}
    for sentences in read_files(filepaths):
        for sentence in sentences:
            for word in sentence:
                if word in vocab:
                    vocab[word] += 1
                else:
                    vocab[word] = 1

    with open(vocab_file, "w") as v:
        for word, count in sorted(vocab.items(), key=lambda x: x[1],
                               reverse=True):
            v.write("{}\n".format(word))
