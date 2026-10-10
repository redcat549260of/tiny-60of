#!/usr/bin/env python3
"""
tiny embedding similarity search utility
"""

import sys, math, argparse

def cosine(v, u):
    dot = sum(a*b for a,b in zip(v,u))
    norm_v = math.sqrt(sum(a*a for a in v))
    norm_u = math.sqrt(sum(b*b for b in u))
    return dot/(norm_v*norm_u) if norm_v and norm_u else 0.0

def parse_vector(s):
    return [float(x) for x in s.strip().split()]

def read_candidates(path):
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts: continue
            cid, vec = parts[0], [float(x) for x in parts[1:]]
            yield cid, vec

def main():
    parser = argparse.ArgumentParser(description='Search embeddings by cosine similarity.')
    parser.add_argument('query', help='Query embedding (space-separated floats)')
    parser.add_argument('-f', '--file', help='File with candidate embeddings', default=None)
    parser.add_argument('-n', '--top', type=int, default=3, help='Top N results')
    args = parser.parse_args()

    query_vec = parse_vector(args.query)

    candidates = []
    if args.file:
        for cid, vec in read_candidates(args.file):
            candidates.append((cid, vec))
    else:
        default = {'a':[0.1,0.2,0.3], 'b':[0.4,0.5,0.6], 'c':[0.7,0.8,0.9]}
        for cid, vec in default.items():
            candidates.append((cid, vec))

    results = []
    for cid, vec in candidates:
        results.append((cosine(query_vec, vec), cid, vec))
    results.sort(reverse=True)

    for