FROM python:3.10

WORKDIR /usr/app/src

COPY dataset.py ./
COPY markets.py ./
COPY tools.py ./
COPY contracts ./contracts

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir requests pandas tqdm web3

# using a public RPC
CMD [ "python", "./dataset.py", "https://lb.nodies.app/v1/406d8dcc043f4cb3959ed7d6673d311a"]
