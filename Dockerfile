FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

WORKDIR /workspace/KDS-Former
COPY requirements.txt .
RUN grep -v -E '^torch(==|>=|<=|~=|>|<|$)' requirements.txt > /tmp/requirements-docker.txt \
    && pip install --no-cache-dir -r /tmp/requirements-docker.txt

COPY . .
