#!/usr/bin/env python
# coding: utf-8

import os
import bz2
import json
import argparse
import logging
import functools

from concurrent.futures import ThreadPoolExecutor

from revscoring.features import wikitext
from revscoring.datasources.meta import mappers, vectorizers
from revscoring.datasources import revision_oriented
from revscoring.dependencies import solve
from revscoring.features.meta import aggregators

import mwxml

class CleanAndVectorize(object):

    def __init__(self, en_kvs_path, **kwargs):
        self.tokenizer = mappers.lower_case(wikitext.revision.datasources.words)
        self.vectorizer = self.load_vectorizer(en_kvs_path)
        self.workers = kwargs.get('workers',1)
        self.save_text = kwargs.get('save_text',False)
        self.save_tokens = kwargs.get('save_tokens', False)
        self.debug = kwargs.get('debug', False)

    def load_vectorizer(self, enwiki_kvs_path):
        enwiki_kvs = vectorizers.word2vec.load_gensim_kv(
            path=enwiki_kvs_path,
            mmap="r"
        )

        vectorize_words = functools.partial(vectorizers.word2vec.vectorize_words, enwiki_kvs)

        revision_text_vectors = vectorizers.word2vec(
            mappers.lower_case(wikitext.revision.datasources.words),
            vectorize_words,
            name="revision.text.en_vectors")

        w2v = aggregators.mean(
            revision_text_vectors,
            vector=True,
            name="revision.text.en_vectors_mean"
        )

        return w2v

    def vectorize(self, filestream):
        dump = mwxml.Dump.from_file(filestream)
        for i, page in enumerate(dump):
            for rev in page:
                observation = {
                    'title':page.title,
                    'page_id':page.id,
                    'rev_id':rev.id,
                    'redirect':page.redirect
                }

                cache = {}

                cache[revision_oriented.revision.text] = rev.text
                tokenized_text = solve(self.tokenizer, cache=cache, context=None)

                cache[self.tokenizer] = tokenized_text
                observation['feature_vector'] = solve(self.vectorizer, cache=cache, context=None)

                if self.save_text:
                    observation['text'] = rev.text

                if self.save_tokens:
                    observation['tokenized_text'] = tokenized_text

                yield observation

            if self.debug:
                if i == self.debug:
                    break

    def _process_dump(self,infile_path,outfile_path):
        with bz2.open(infile_path) as filestream:
            with open(outfile_path,'w') as outfile:
                outfile.write('[')
                for i, obs in enumerate(self.vectorize(filestream)):
                    if i > 0:
                        outfile.write(',')
                        if i % 10 == 0:
                            print('{0}, {1}, {2}'.format(infile_path,outfile_path,i))
                    json.dump(obs,outfile)
                outfile.write(']')

    def process(self,files_to_process,files_to_write):
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            executor.map(self._process_dump,
                         files_to_process,
                         files_to_write)

def main():
    parser = argparse.ArgumentParser(description='Convert .bz2 compressed xml dumps of Wikipedia articles to feature vectors.')
    parser.add_argument('indir',
                        help='a directory to read compressed dump files')
    parser.add_argument('outdir',
                        help='a directory to write feature vectors')
    parser.add_argument('embedding_file',
                        help='a kvs embedding layer file')
    parser.add_argument('-w', '--workers', type=int, default=1)
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-d', '--debug', type=int, default=False)
    parser.add_argument('--save_text', action='store_true')
    parser.add_argument('--save_tokens', action='store_true')
    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    handler = logging.StreamHandler()

    formatter = logging.Formatter(fmt='[%(levelname)s %(asctime)s] %(message)s',
                                  datefmt='%m/%d/%Y %I:%M:%S %p')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    if args.verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    cv = CleanAndVectorize(en_kvs_path=args.embedding_file,
                           workers=args.workers,
                           debug=args.debug,
                           save_text=args.save_text,
                           save_tokens=args.save_tokens)

    files_to_process = [os.path.join(args.indir,f) for f in os.listdir(args.indir)
                        if f.find('enwiki-latest-pages-articles') >= 0
                        and f.find('multistream') < 0
                        and f.find('rss') < 0
                        and f.find('enwiki-latest-pages-articles.xml.bz2') < 0]

    files_to_write = [os.path.join(args.outdir,'{0}.json'.format(os.path.basename(f).rstrip('.bz2'))) for f in files_to_process]

    cv.process(files_to_process,files_to_write)


if __name__ == "__main__":
    main()




