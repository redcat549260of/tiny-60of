"""
Tiny embedding similarity search utility.
Usage: python search.py data.txt --query 0.1,0.2,0.3 --top 5
"""

import argparse, math, heapq, sys

def load_embeddings(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 2: continue
            id_ = parts[0]
            vec = [float(v) for v in parts[1:]]
            norm = math.sqrt(sum(v*v for v in vec))
            data.append((id_, vec, norm))
    return data

def cosine(q, qnorm, vec, vnorm):
    dot = sum(a*b for a,b in zip(q, vec))
    return dot/(qnorm*vnorm) if vnorm else 0.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('file')
    parser.add_argument('--query', required=True)
    parser.add_argument('--top', type=int, default=5)
    args = parser.parse_args()

    q = [float(v) for v in args.query.split(',')]
    qnorm = math.sqrt(sum(v*v for v in q))

    data = load_embeddings(args.file)
    sims = [(cosine(q, qnorm, vec, norm), id_) for id_, vec, norm in data]
    top = heapq.nlargest(args.top, sims)

    for sim, id_ in top:
        print(f'{id_}\t{sim:.4f}')

if __name__ == '__main__':
    main()