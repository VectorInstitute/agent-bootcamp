FROM e2bdev/code-interpreter:latest 

# All downloaded files will be available under /data
WORKDIR /data

# Example: download given link to publicly-shared Google Drive file and unzip.
# Set permission to open for the sandbox user.
RUN python3 -m pip install gdown && \
python3 -m gdown -O local_sqlite.zip "1coEVsCZq-Xvj9p2TnhBFoFTsY-UoYGmG" && \
unzip local_sqlite.zip && \
chmod -R a+wr /data && \
rm -v local_sqlite.zip

# Example: download file from public URL- e.g., HuggingFacek
RUN wget --content-disposition \
"https://huggingface.co/datasets/aki005/Recipes_json_vector_nbarbcone/resolve/main/recipes.json"