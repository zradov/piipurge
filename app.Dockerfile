FROM python:3.12-slim-bookworm

WORKDIR /app

RUN bash -c "echo -e 'deb https://notesalexp.org/tesseract-ocr5/bookworm/ bookworm main' | tee /etc/apt/sources.list.d/notesalexp.list"
RUN apt-get update -y -o Acquire::AllowInsecureRepositories=true
RUN apt-get install -y --no-install-recommends --allow-unauthenticated -o Acquire::AllowInsecureRepositories=true \
    notesalexp-keyring apt-transport-https build-essential git ffmpeg libsm6 libxext6 tesseract-ocr=5.5.0-1
RUN rm -rf /var/lib/apt/lists/

COPY src/ /app/src
COPY models/ /app/models
COPY requirements/base.txt /app

RUN pip install --upgrade pip && \
    pip install -r base.txt

RUN mkdir -p /app/output

EXPOSE 8501

CMD ["streamlit", "run", "src/frontend/app.py", "--server.port=8501", "--server.address=0.0.0.0"]


