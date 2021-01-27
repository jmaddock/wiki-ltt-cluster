all: vectorize cluster

# shouldn't hardcode input dir
vectorize: word2vec/enwiki-20200501-learned_vectors.50_cell.10k.kv
	python ./wiki-ltt-cluster/clean_and_vectorize.py \
		~/research_data/wmf_knowledge_graph/raw_xml_dumps/enwiki/ \
		./datasets/vectorized \
		./word2vec/enwiki-20200501-learned_vectors.50_cell.10k.kv \
		--debug 100

# need to update this with reasonable value of k
cluster:
	python ./wiki-ltt-cluster/cluster.py \
		./datasets/vectorized \
		datasets/clustered/enwiki_clustered_26-01-2021.json \
		-k 2 5 1 

word2vec/enwiki-20200501-learned_vectors.50_cell.10k.kv:
	wget https://analytics.wikimedia.org/datasets/archive/public-datasets/all/ores/topic/vectors/enwiki-20200501-learned_vectors.50_cell.10k.kv -qO- > $@
