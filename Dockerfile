FROM tiangolo/uwsgi-nginx-flask:python3.7
ENV STATIC_URL /static
ENV STATIC_PATH /var/www/app/static
RUN    apt-get update \
    && apt-get install openssl \
    && apt-get install ca-certificates
COPY ./requirements.txt /var/www/requirements.txt
RUN pip install --upgrade pip
RUN pip install -r /var/www/requirements.txt
COPY ./nltk_pkg.py /var/www/nltk_pkg.py
RUN python /var/www/nltk_pkg.py
ENV UWSGI_CHEAPER 0
ENV UWSGI_PROCESSES 1
