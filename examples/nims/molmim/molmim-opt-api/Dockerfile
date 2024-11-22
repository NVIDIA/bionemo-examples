# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
FROM python:3.11

WORKDIR /code

RUN pip install --no-cache-dir --upgrade fastapi[standard] requests rdkit cmaes

COPY ./*.py /code/

CMD ["fastapi", "run", "/code/main.py", "--port", "80"]
