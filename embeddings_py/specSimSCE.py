# Fine tune Bert-like model weights using homemade sentence dataset
# Original Source for model-fine_tuning: https://github.com/UKPLab/sentence-transformers/blob/master/examples/unsupervised_learning/SimCSE/train_simcse_from_file.py
from torch.utils.data import DataLoader
import math
from sentence_transformers import models, losses
from sentence_transformers import LoggingHandler, SentenceTransformer, InputExample
import logging
from datetime import datetime
import gzip
from tqdm import tqdm
from transformers import AutoTokenizer

from embeddings.sentence_embeddings2 import *
from embeddings.sentence_transformer_models import create_author_embeddings, get_custom_model, get_embedding
from embeddings.utils import find_author_rank, mkdirs, get_positions


def preprocess_sentence(sentence: str):
    """
    Function to preprocess sentence string before using it as model training input.
    Remove unecessary newlines and '...' that usually exist at the end of the sentence.
    :param sentence: str
    :return: res: str
    """

    res = re.sub("\n", " ", sentence)  # remove newlines
    res = re.sub("…", "", res)         # remove dots in sentence end if exist

    return res + "\n"


def create_scientific_sentences_dataset(input_name: str,
                                        output_name: str,
                                        write_mode: str,
                                        include_abstract=True):
    """
    Creates a dataset .txt file with all sentences from input_name file
    with total token coynt less than 100 with scibert tokenizer seperated using newline.
    Accepted Write modes are "a" and "w"
    :param input_name: str
    :param output_name: str
    :param write_mode: str
    :param include_abstract:
    :return:
    """
    print(f'Creating Dataset from Author Publications - Input file "{input_name}"')

    max_token_len = 100
    tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')

    with open(output_name, write_mode, encoding="utf-8") as f:
        with open(input_name, encoding='utf-8') as input_file:
            objects = ijson.items(input_file, 'item')
            objects = islice(objects, 500)
            for author in tqdm(objects):
                if 'Publications' in author:
                    for paper in author.get('Publications'):
                        if "Title" in paper:
                            text = preprocess_sentence(paper["Title"])
                            text_tokenized = tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=512)
                            token_num = text_tokenized.data["input_ids"].size(1)
                            if token_num >= max_token_len: continue
                            f.write(text)


def fine_tune_bertlike_model(sentences_path,
                             model_name: str,
                             train_batch_size: int,
                             num_epochs: int):
    max_seq_length = 100

    #### Just some code to print debug information to stdout
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])
    #### /print debug information to stdout

    filepath = sentences_path
    # Save path to store our model
    model_output_path = 'models/train_simcse-{}'.format(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)

    # Apply mean pooling to get one fixed sized sentence vector
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    ################# Read the train corpus  #################
    train_samples = []
    with gzip.open(filepath, 'rt', encoding='utf8') if filepath.endswith('.gz') else open(filepath, encoding='utf8') as fIn:
        for line in tqdm(fIn, desc='Read file'):
            line = line.strip()
            if len(line) >= 10:
                train_samples.append(InputExample(texts=[line, line]))

    logging.info("Train sentences: {}".format(len(train_samples)))

    # We train our model using the MultipleNegativesRankingLoss
    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size, drop_last=True)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)  # 10% of train data for warm-up
    logging.info("Warmup-steps: {}".format(warmup_steps))

    # Train the model
    model.fit(train_objectives=[(train_dataloader, train_loss)],
              epochs=num_epochs,
              warmup_steps=warmup_steps,
              optimizer_params={'lr': 3e-5},
              checkpoint_path=model_output_path,
              show_progress_bar=True,
              use_amp=False)  # Set to True, if your GPU supports FP16 cores)


def sentence_transformers_fine_tuning_main(dataset_path: str, batch_size: int, num_epochs: int):
    fname_in = r'..\json_files\csd_in_with_abstract\csd_in_specter_no_greek_rank.json'
    fname_out = r'..\json_files\csd_out_with_abstract\csd_out_completed_missing_2_no_greek_rank_no_nan.json'

    create_scientific_sentences_dataset(input_name=fname_in, output_name=dataset_path, write_mode="w")
    create_scientific_sentences_dataset(input_name=fname_out, output_name=dataset_path, write_mode="a")

    # Fine Tune a pretrained sentence transformer model (from huggingface available pretrained models)
    fine_tune_bertlike_model(sentences_path=dataset_path,
                             model_name='allenai/specter',
                             train_batch_size=batch_size,
                             num_epochs=num_epochs)


def create_author_embeddings_all(csd_list, model_name, model, tokenizer, in_or_out):
    for i, csd_file in enumerate(csd_list):
        with open(csd_file, encoding='utf-8') as input_file:
            objects = ijson.items(input_file, 'item')
            objects = islice(objects, 500)
            for author in tqdm(objects):
                create_author_embeddings(author, model_name=model_name, model=model, tokenizer=tokenizer, in_or_out=in_or_out[i])


def create_author_aggregations_all(csd_list: list, model_name: str, in_or_out: list):
    for i, csd_file in enumerate(csd_list):
        with open(csd_file, encoding='utf-8') as input_file:
            objects = ijson.items(input_file, 'item')
            objects = islice(objects, 500)
            for author in tqdm(objects):
                create_author_aggregations(author, model_name=model_name, in_or_out=in_or_out[i])


if __name__ == "__main__":

    fname_in = r'..\json_files\csd_in_with_abstract\csd_in_specter_no_greek_rank.json'
    fname_out = r'..\json_files\csd_out_with_abstract\csd_out_completed_missing_2_no_greek_rank_no_nan.json'

    # Create Sentences Dataset from Csd in and out author publications
    learning_rate = "5e-5"
    batch_size = 16
    max_tokens = 100
    n_epochs = 1

    model_str = f"{max_tokens}_{batch_size}_{learning_rate}" if n_epochs == 1 else f"{max_tokens}_{batch_size}_{learning_rate}_{n_epochs}epochs"
    model_name = f"specter_simcse_{model_str}"
    custom_model_dir=f"./models/train_simcse-{model_str}/model_folder"
    csd_list = [fname_in, fname_out]
    in_or_out = ["in", "out"]

    model, tokenizer = get_custom_model(custom_model_dir)
    create_author_embeddings_all(csd_list, model_name, model, tokenizer, in_or_out)
    create_author_aggregations_all(csd_list, model_name, in_or_out)

    titles, descriptions, authors_targets_in, authors_targets_in_standby, authors_targets_out, authors_targets_out_standby, position_ranks = get_positions(r'.\positions\test_apella_data.json')
    # Evaluate Embeddings of the fine tuned model
    main_ranking_authors_aggregations(fname_in, model_name, titles, descriptions, authors_targets_in, authors_targets_in_standby, position_ranks,
                                      in_or_out="in", custom_model_dir=custom_model_dir)
    main_ranking_authors_aggregations(fname_out, model_name, titles, descriptions, authors_targets_out, authors_targets_out_standby, position_ranks,
                                      in_or_out="out", custom_model_dir=custom_model_dir)

    print_final_results(titles)
