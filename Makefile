all: vectorize cluster

vectorize: word2vec/enwiki-20200501-learned_vectors.50_cell.10k.kv
    python ./wiki-ltt-cluster/clean_and_vectorize \
        /mnt/data/xmldatadumps/public/enwiki/latest \ # this shouldn't be hardcoded
        ./datasets/vectorized \
        ./word2vec/enwiki-20200501-learned_vectors.50_cell.10k.kv

cluster:
    python ./wiki-ltt-cluster/cluster \
        ./datasets/vectorized \
        datasets/clustered/enwiki_clustered_26-01-2021.json \
        -k # need to update this with reasonable value of k

word2vec/enwiki-20200501-learned_vectors.50_cell.10k.kv:
	wget https://analytics.wikimedia.org/datasets/archive/public-datasets/all/ores/topic/vectors/enwiki-20200501-learned_vectors.50_cell.10k.kv -qO- > $@

