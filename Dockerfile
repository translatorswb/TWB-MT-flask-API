FROM tiangolo/uwsgi-nginx-flask:python3.7
ENV STATIC_URL /static
ENV STATIC_PATH /var/www/app/static
COPY ./requirements.txt /var/www/requirements.txt
RUN pip install --upgrade pip
RUN pip install -r /var/www/requirements.txt
ENV UWSGI_CHEAPER 0
ENV UWSGI_PROCESSES 1
