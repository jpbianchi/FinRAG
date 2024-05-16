FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE 1  
# ^ saves space by not writing .pyc files
ENV PYTHONUNBUFFERED 1  
# ^ ensures that the output from the Python app is sent straight to the terminal without being buffered -> real time monitoring

ENV ENVIRONMENT=dev 

COPY ./app /app
WORKDIR /app
RUN mkdir /data

RUN pip install --no-cache-dir --upgrade -r requirements.txt
# ^ no caching of the packages to save space

RUN python -c "import nltk; nltk.download('stopwords')"
# ^ to fix runtime error, see https://github.com/run-llama/llama_index/issues/10681

RUN chmod -R 777 /usr/local/lib/python3.10/site-packages

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]