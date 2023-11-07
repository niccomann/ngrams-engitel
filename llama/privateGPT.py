from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp
import os
import argparse
import time
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


load_dotenv()

embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get('PERSIST_DIRECTORY')
model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = os.environ.get('MODEL_N_CTX')
model_n_batch = int(os.environ.get('MODEL_N_BATCH', 8))
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS', 4))

import os
from dotenv import load_dotenv
from chromadb.config import Settings

load_dotenv()

# Define the folder for storing database
PERSIST_DIRECTORY = os.environ.get('PERSIST_DIRECTORY')

# Define the Chroma settings
CHROMA_SETTINGS = Settings(chroma_db_impl='duckdb+parquet', persist_directory=PERSIST_DIRECTORY,
                           anonymized_telemetry=False)


def evaluate_prompt(query):

    print(ROOT_DIR)
    os.chdir(ROOT_DIR+"/llama") ## Perchè se lo chiamo da fuori fa confusione con i path

    # Parse the command line arguments
    args = parse_arguments()
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)

    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})

    # activate/deactivate the streaming StdOut callback for LLMs
    callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]

    # Prepare the LLM
    match model_type:
        case "LlamaCpp":


            llm = LlamaCpp(model_path=model_path, n_ctx=model_n_ctx, n_batch=model_n_batch, callbacks=callbacks,
                               verbose=False)

        case "GPT4All":

            llm = GPT4All(model=model_path, n_ctx=model_n_ctx, backend='gptj', n_batch=model_n_batch,
                              callbacks=callbacks, verbose=False)

        case _default:
            print(f"Model {model_type} not supported!")
            exit;
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever,
                                     return_source_documents=not args.hide_source)
    # Interactive questions and answers

    # Get the answer from the chain
    start = time.time()
    res = qa(query)
    answer, docs = res['result'], [] if args.hide_source else res['source_documents']
    end = time.time()

    # Print the result
    print("\n\n> Question:")
    print(query)
    print(f"\n> Answer (took {round(end - start, 2)} s.):")
    print(answer)

    # Print the relevant sources used for the answer
    for document in docs:
        pass
        # print("\n> " + document.metadata["source"] + ":")
        # print(document.page_content)

    other_info = document.metadata["source"] + ":" + document.page_content

    return answer, other_info


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='llama: Ask questions to your documents without an internet connection, '
                    'using the power of LLMs.')
    parser.add_argument("--hide-source", "-S", action='store_true',
                        help='Use this flag to disable printing of source documents used for answers.')

    parser.add_argument("--mute-stream", "-M",
                        action='store_true',
                        help='Use this flag to disable the streaming StdOut callback for LLMs.')

    return parser.parse_args()


if __name__ == "__main__":
    #
    # embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    #
    # db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    # retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
    #
    # llm = LlamaCpp(model_path=model_path, n_ctx=model_n_ctx, n_batch=model_n_batch,
    #                verbose=False)
    #
    # grossa_richiesta = ''
    # list_of_requests = get_bodies_from_bigquery()
    #

    # def crea_file_csv(lista_stringhe, nome_file):
    #     with open(nome_file, 'w', newline='') as file_csv:
    #         writer = csv.writer(file_csv)
    #         for stringa in lista_stringhe:
    #             writer.writerow([stringa])
    #
    #
    # crea_file_csv(list_of_requests, 'source_documents/prova.csv')
    #





    # for request in list_of_requests[:500]:
    #     grossa_richiesta += request + ';    '
    # inizio = time.time()
    #
    # print(grossa_richiesta)
    #
    # qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    # query = "Can you tell me if the following http requests (separated by the ';' character) are cyber attacks :" + grossa_richiesta
    # res = qa(query)
    #
    # answer = res['result']
    # fine = time.time()
    #
    # print(answer)
    # print(fine - inizio)

    # list_of_requests = get_bodies_from_bigquery()
    # grossa_richiesta = ''
    # for request in list_of_requests[:1000]:
    #     grossa_richiesta += request + ';    '
    #
    # print(grossa_richiesta)
    #
    # # Calcola il tempi che ci mette a eseguire la funzione ev
    # inizio = time.time()
    #
    # answer, other_info = evaluate_prompt(
    #     "Can you tell me if the following http requests (separated by the ';' character):" + grossa_richiesta + "Are they cyber attacks? If you find an attack tell me which request is an attack")
    #
    # fine = time.time()
    #
    # print(answer)
    #
    # print(fine - inizio)

    #answer, other_info = evaluate_prompt("Mi puoi dire qualcosa sui file che ho caricato ?")

    answer, other_info = evaluate_prompt("Can you analyze the http requests of http requests of July 3, 2023 and tell me if one of them is a cyber attack? If yes show me the line which is a cyber attack")
   # answer, other_info = evaluate_prompt("Are there any cyberattacks like the sql injection in the csv lines representing http requests? If yes show me the line which is a sql injection")


