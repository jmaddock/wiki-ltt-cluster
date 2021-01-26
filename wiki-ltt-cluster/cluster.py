#!/usr/bin/env python
# coding: utf-8

import faiss
import json
import os
import argparse
import logging
import numpy as np
from sklearn import metrics

def read_json_files(indir,file_list):
    observation_list = []
    for filename in file_list:
        filepath = os.path.join(indir,filename)
        with open(filepath,'r') as filestream:
            observation_list += json.load(filestream)
            
    return observation_list


def get_feature_vectors(observation_list):
    return np.asarray([obs['feature_vector'] for obs in observation_list],dtype=np.float32)


def cluster(feature_vector_list, k):
    c = faiss.Kmeans(feature_vector_list.shape[1],k,spherical = True)
    c.train(feature_vector_list)
    sims, cluster_assignments = c.assign(feature_vector_list)
    return cluster_assignments


def determine_k(feature_vector_list, k_to_try):
    max_score = 0
    best_k = None
    best_cluster_assignment = None
    scores = {}
    for k in k_to_try:
        cluster_assignments = cluster(feature_vector_list,k)
        score = metrics.silhouette_score(feature_vector_list,
                                         cluster_assignments,
                                         metric='euclidean')
        scores[k] = score
        if score > max_score:
            max_score = score
            best_k = k
            best_cluster_assignment = cluster_assignments
            
    
    result = {
        'scores':scores,
        'max_score':max_score,
        'best_k':best_k,
        'best_cluster_assignment':best_cluster_assignment
    }
    
    return result


def main():
    parser = argparse.ArgumentParser(description='Determine the optimal number of clusters from a range and return cluster assignments for all observations.')
    parser.add_argument('indir',
                        help='a directory to read json files with feature vectors')
    parser.add_argument('outfile',
                        help='a file path to write cluster assignments')
    parser.add_argument('-v', '--verbose',
                        action='store_true')
    parser.add_argument('-k',
                        type=int,
                        nargs=3,
                        help='3 integers describing a range of k values to try, specified by min max step')

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

    files_to_process = [x for x in os.listdir(args.indir) if '.json' in x]
    observation_list = read_json_files(args.indir,files_to_process)
    feature_vector_list = get_feature_vectors(observation_list)
    
    k_to_try = range(args.k[0],args.k[1],args.k[2])
    
    result = determine_k(feature_vector_list,k_to_try)

    with open(args.outfile,'w') as outfile:
        outfile.write('[')
        for i, obs in enumerate(observation_list):
            json.dump({
                'title':obs['title'],
                'page_id':obs['page_id'],
                'rev_id':obs['rev_id'],
                'redirect':obs['redirect'],
                'cluster_assignment':int(result['best_cluster_assignment'][i])
            },outfile)
        
        outfile.write(']')
        
if __name__ == "__main__":
    main()