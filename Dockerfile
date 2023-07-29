FROM python:3.8-slim-buster AS compile-image

# Install Dependencies
RUN apt-get -y update && apt-get -y upgrade


WORKDIR app

# Install dependencies
RUN python -m venv /opt/venv
RUN /opt/venv/bin/python -m pip install --upgrade pip setuptools

# Make sure we use the virtualenv:
ENV PATH="/opt/venv/bin:$PATH"

COPY setup.py .
COPY requirements.txt .
RUN pip install -r requirements.txt --use-pep517

# Setup find custom package and copy app
COPY . .
RUN pip install -e .


FROM python:3.8-slim-buster AS build-image
COPY --from=compile-image /opt/venv /opt/venv
COPY --from=compile-image app app

WORKDIR app
ENV PATH="/opt/venv/bin:$PATH"

CMD ["python", "./cmd/delete_data/main.py"]
