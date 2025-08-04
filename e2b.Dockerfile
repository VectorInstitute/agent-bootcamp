FROM e2bdev/code-interpreter:latest

# All downloaded files will be available under /data
WORKDIR /data

# # Example: download given link to publicly-shared Google Drive file and unzip.
# # Set permission to open for the sandbox user.
# RUN python3 -m pip install gdown && \
# python3 -m gdown -O local_sqlite.zip "EXAMPLE_FILE_ID" && \
# unzip local_sqlite.zip && \
# chmod -R a+wr /data && \
# rm -v local_sqlite.zip

# Example: download file from public URL- e.g., HuggingFacek
RUN wget --content-disposition \
"https://huggingface.co/datasets/vector-institute/hotpotqa/resolve/main/data/validation-00000-of-00001.parquet"
